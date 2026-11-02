// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use imaging::ShaderHandle;
use imaging_shader_ext::{
    ShaderBindings, ShaderChildDesc, ShaderDesc, ShaderError, ShaderFactory, ShaderKind,
    ShaderLayout, ShaderProgram, ShaderSource, UniformDesc, UniformType, UniformValue,
};
use skia_safe::{
    self as sk,
    runtime_effect::{ChildPtr, ChildType, uniform::Type},
};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

/// Shared owner of compiled `SkSL` programs and immutable shader/filter instances.
///
/// Clones share one registry. Attach it to a [`crate::SkiaConfig`],
/// [`crate::SkCanvasSink`], or [`crate::SkPictureRecorderSink`] before rendering.
/// Recorded imaging scenes hold identities only, so resources must remain registered
/// until replay. Native Skia pictures and bound parent shaders retain their resources.
#[derive(Clone, Debug, Default)]
pub struct SkiaShaderRegistry(Rc<RefCell<Registry>>);

#[derive(Debug, Default)]
struct Registry {
    programs: HashMap<ShaderProgram, Program>,
    shaders: HashMap<ShaderHandle, BoundEffect>,
}

#[derive(Debug)]
enum BoundEffect {
    Shader(sk::Shader),
    ColorFilter(sk::ColorFilter),
    ImageFilter(sk::ImageFilter),
}

#[derive(Debug)]
struct Program {
    effect: sk::RuntimeEffect,
    layout: ShaderLayout,
}

impl SkiaShaderRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a native shader as a new immutable instance.
    ///
    /// Any images or other resources referenced by the native shader must be usable
    /// on the target canvas. This does not transfer GPU resources between devices.
    pub fn register_shader(&self, shader: sk::Shader) -> ShaderHandle {
        self.register(BoundEffect::Shader(shader))
    }

    /// Register an immutable native color filter for use in `Filter::Shader`.
    pub fn register_color_filter(&self, filter: sk::ColorFilter) -> ShaderHandle {
        self.register(BoundEffect::ColorFilter(filter))
    }

    /// Register an immutable native image filter for use in `Filter::Shader`.
    /// Implicit source inputs receive the preceding filter result or group content.
    pub fn register_image_filter(&self, filter: sk::ImageFilter) -> ShaderHandle {
        self.register(BoundEffect::ImageFilter(filter))
    }

    fn register(&self, effect: BoundEffect) -> ShaderHandle {
        let handle = ShaderHandle::new();
        self.0.borrow_mut().shaders.insert(handle, effect);
        handle
    }

    /// Remove a shader or filter binding. Future semantic-scene replay using it fails.
    /// Existing native pictures and bound parent shaders remain valid.
    pub fn remove_shader(&self, handle: &ShaderHandle) -> bool {
        self.0.borrow_mut().shaders.remove(handle).is_some()
    }

    /// Remove compiled code. Previously bound shader instances remain valid.
    pub fn remove_program(&self, program: &ShaderProgram) -> bool {
        self.0.borrow_mut().programs.remove(program).is_some()
    }

    pub(crate) fn resolve(&self, handle: ShaderHandle) -> Result<sk::Shader, crate::Error> {
        match self.0.borrow().shaders.get(&handle) {
            Some(BoundEffect::Shader(shader)) => Ok(shader.clone()),
            Some(_) => Err(crate::Error::InvalidShaderUsage {
                shader: handle,
                usage: "a brush",
            }),
            None => Err(crate::Error::UnboundShader(handle)),
        }
    }

    pub(crate) fn filter_is_spatial(&self, handle: ShaderHandle) -> Result<bool, crate::Error> {
        match self.0.borrow().shaders.get(&handle) {
            Some(BoundEffect::ColorFilter(_)) => Ok(false),
            Some(BoundEffect::ImageFilter(_)) => Ok(true),
            Some(_) => Err(crate::Error::InvalidShaderUsage {
                shader: handle,
                usage: "a filter",
            }),
            None => Err(crate::Error::UnboundShader(handle)),
        }
    }

    pub(crate) fn apply_filter(
        &self,
        handle: ShaderHandle,
        input: Option<sk::ImageFilter>,
    ) -> Result<sk::ImageFilter, crate::Error> {
        let registry = self.0.borrow();
        let result = match registry.shaders.get(&handle) {
            Some(BoundEffect::ColorFilter(filter)) => {
                sk::image_filters::color_filter(filter.clone(), input, None)
            }
            Some(BoundEffect::ImageFilter(filter)) => match input {
                Some(input) => sk::image_filters::compose(filter.clone(), input),
                None => Some(filter.clone()),
            },
            Some(_) => {
                return Err(crate::Error::InvalidShaderUsage {
                    shader: handle,
                    usage: "a filter",
                });
            }
            None => return Err(crate::Error::UnboundShader(handle)),
        };
        result.ok_or(crate::Error::UnsupportedFilter)
    }
}

impl ShaderFactory for SkiaShaderRegistry {
    fn create_program(&self, desc: ShaderDesc<'_>) -> Result<ShaderProgram, ShaderError> {
        let source = match desc.source {
            ShaderSource::SkSl(source) => source,
            _ => return Err(ShaderError::UnsupportedSource),
        };
        let options = sk::runtime_effect::Options {
            name: desc.label.unwrap_or_default(),
            ..sk::runtime_effect::Options::default()
        };
        let effect = match desc.kind {
            ShaderKind::Shader | ShaderKind::ImageFilter => {
                sk::RuntimeEffect::make_for_shader(source, Some(&options))
            }
            ShaderKind::ColorFilter => {
                sk::RuntimeEffect::make_for_color_filter(source, Some(&options))
            }
            _ => return Err(ShaderError::UnsupportedSource),
        }
        .map_err(ShaderError::Compilation)?;
        let uniforms = effect
            .uniforms()
            .iter()
            .map(|uniform| {
                if desc.kind == ShaderKind::ImageFilter && uniform.is_array() {
                    return Err(ShaderError::InvalidUniform {
                        name: uniform.name().into(),
                        reason: "image-filter uniform arrays are not supported by this backend",
                    });
                }
                Ok(UniformDesc {
                    name: uniform.name().into(),
                    ty: uniform_type(uniform.ty()),
                    count: usize::try_from(uniform.count())
                        .expect("Skia uniform counts are nonnegative"),
                    is_array: uniform.is_array(),
                    is_color: uniform.is_color(),
                })
            })
            .collect::<Result<_, _>>()?;
        let children: Vec<ShaderChildDesc> = effect
            .children()
            .iter()
            .map(|child| {
                let kind = match child.ty() {
                    ChildType::Shader => ShaderKind::Shader,
                    ChildType::ColorFilter => ShaderKind::ColorFilter,
                    ChildType::Blender => {
                        return Err(ShaderError::InvalidChild {
                            name: child.name().into(),
                            reason: "blender children are not supported",
                        });
                    }
                };
                Ok(ShaderChildDesc {
                    name: child.name().into(),
                    kind,
                })
            })
            .collect::<Result<_, _>>()?;
        if desc.kind == ShaderKind::ImageFilter
            && (children.len() != 1 || children[0].kind != ShaderKind::Shader)
        {
            return Err(ShaderError::InvalidImageFilter(
                "exactly one shader child is required",
            ));
        }
        let program = ShaderProgram::new();
        self.0.borrow_mut().programs.insert(
            program,
            Program {
                effect,
                layout: ShaderLayout {
                    kind: desc.kind,
                    uniforms,
                    children,
                },
            },
        );
        Ok(program)
    }

    fn layout(&self, program: &ShaderProgram) -> Result<ShaderLayout, ShaderError> {
        self.0
            .borrow()
            .programs
            .get(program)
            .map(|program| program.layout.clone())
            .ok_or(ShaderError::UnboundProgram)
    }

    fn bind_shader(
        &self,
        program: &ShaderProgram,
        bindings: ShaderBindings<'_>,
    ) -> Result<ShaderHandle, ShaderError> {
        let shader = {
            let registry = self.0.borrow();
            let program = registry
                .programs
                .get(program)
                .ok_or(ShaderError::UnboundProgram)?;
            let effect = &program.effect;
            let mut data = vec![0_u8; effect.uniform_size()];
            let mut seen = HashSet::new();
            for &(name, value) in bindings.uniforms {
                let invalid = |reason| ShaderError::InvalidUniform {
                    name: name.into(),
                    reason,
                };
                if !seen.insert(name) {
                    return Err(invalid("duplicate binding"));
                }
                let uniform = effect
                    .find_uniform(name)
                    .ok_or_else(|| invalid("unknown name"))?;
                let is_int = matches!(
                    uniform.ty(),
                    Type::Int | Type::Int2 | Type::Int3 | Type::Int4
                );
                let count = match value {
                    UniformValue::Float(values) if !is_int => values.len(),
                    UniformValue::Int(values) if is_int => values.len(),
                    _ => return Err(invalid("numeric type does not match declaration")),
                };
                if count != uniform.size_in_bytes() / 4 {
                    return Err(invalid("element count does not match declaration"));
                }
                let bytes = &mut data[uniform.offset()..uniform.offset() + uniform.size_in_bytes()];
                match value {
                    UniformValue::Float(values) => {
                        for (dst, value) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                            dst.copy_from_slice(&value.to_ne_bytes());
                        }
                    }
                    UniformValue::Int(values) => {
                        for (dst, value) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                            dst.copy_from_slice(&value.to_ne_bytes());
                        }
                    }
                }
            }
            for uniform in effect.uniforms() {
                if !seen.contains(uniform.name()) {
                    return Err(ShaderError::InvalidUniform {
                        name: uniform.name().into(),
                        reason: "missing binding",
                    });
                }
            }

            if program.layout.kind == ShaderKind::ImageFilter {
                let input = bindings
                    .image_filter
                    .ok_or(ShaderError::InvalidImageFilter(
                        "missing dynamic input configuration",
                    ))?;
                if !input.sample_radius.is_finite() || input.sample_radius < 0.0 {
                    return Err(ShaderError::InvalidImageFilter(
                        "sample radius must be finite and nonnegative",
                    ));
                }
                if input.input != program.layout.children[0].name {
                    return Err(ShaderError::InvalidImageFilter(
                        "input must name the declared shader child",
                    ));
                }
                if !bindings.children.is_empty() {
                    return Err(ShaderError::InvalidImageFilter(
                        "the input child is supplied automatically",
                    ));
                }
                let mut builder = sk::runtime_effect::RuntimeShaderBuilder::new(effect.clone());
                // Names, numeric types, and lengths were checked above before calling the
                // native builder, whose setters assume well-formed uniform data.
                for &(name, value) in bindings.uniforms {
                    let result = match value {
                        UniformValue::Float(values) => builder.set_uniform_float(name, values),
                        UniformValue::Int(values) => builder.set_uniform_int(name, values),
                    };
                    result.map_err(|_| ShaderError::InvalidUniform {
                        name: name.into(),
                        reason: "unsupported image-filter uniform size",
                    })?;
                }
                let filter = sk::image_filters::runtime_shader_with_options(
                    &builder,
                    input.sample_radius,
                    input.input,
                    None,
                    false,
                )
                .ok_or(ShaderError::CreateShader)?;
                drop(registry);
                return Ok(self.register_image_filter(filter));
            }
            if bindings.image_filter.is_some() {
                return Err(ShaderError::InvalidImageFilter(
                    "only image filters accept dynamic input configuration",
                ));
            }

            let mut children = vec![None; effect.children().len()];
            for &(name, handle) in bindings.children {
                let invalid = |reason| ShaderError::InvalidChild {
                    name: name.into(),
                    reason,
                };
                let child = effect
                    .find_child(name)
                    .ok_or_else(|| invalid("unknown name"))?;
                let slot = &mut children[child.index()];
                if slot.is_some() {
                    return Err(invalid("duplicate binding"));
                }
                let shader = registry
                    .shaders
                    .get(&handle)
                    .ok_or(ShaderError::UnboundShader(handle))?;
                *slot = Some(match (child.ty(), shader) {
                    (ChildType::Shader, BoundEffect::Shader(shader)) => {
                        ChildPtr::Shader(shader.clone())
                    }
                    (ChildType::ColorFilter, BoundEffect::ColorFilter(filter)) => {
                        ChildPtr::ColorFilter(filter.clone())
                    }
                    _ => return Err(invalid("instance kind does not match child declaration")),
                });
            }
            for child in effect.children() {
                if children[child.index()].is_none() {
                    return Err(ShaderError::InvalidChild {
                        name: child.name().into(),
                        reason: "missing binding",
                    });
                }
            }
            let children = children.into_iter().flatten().collect::<Vec<_>>();
            let data = sk::Data::new_copy(&data);
            match program.layout.kind {
                ShaderKind::Shader => BoundEffect::Shader(
                    effect
                        .make_shader(data, &children, None)
                        .ok_or(ShaderError::CreateShader)?,
                ),
                ShaderKind::ColorFilter => BoundEffect::ColorFilter(
                    effect
                        .make_color_filter(data, children.as_slice())
                        .ok_or(ShaderError::CreateShader)?,
                ),
                _ => return Err(ShaderError::UnsupportedSource),
            }
        };
        Ok(self.register(shader))
    }
}

fn uniform_type(ty: Type) -> UniformType {
    match ty {
        Type::Float => UniformType::Float,
        Type::Float2 => UniformType::Float2,
        Type::Float3 => UniformType::Float3,
        Type::Float4 => UniformType::Float4,
        Type::Float2x2 => UniformType::Float2x2,
        Type::Float3x3 => UniformType::Float3x3,
        Type::Float4x4 => UniformType::Float4x4,
        Type::Int => UniformType::Int,
        Type::Int2 => UniformType::Int2,
        Type::Int3 => UniformType::Int3,
        Type::Int4 => UniformType::Int4,
    }
}
