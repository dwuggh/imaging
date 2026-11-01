// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use imaging::ShaderHandle;
use imaging_shader_ext::{
    ShaderBindings, ShaderDesc, ShaderError, ShaderFactory, ShaderLayout, ShaderProgram,
    ShaderSource, UniformDesc, UniformType, UniformValue,
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

/// Shared owner of compiled `SkSL` programs and immutable shader instances.
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
    shaders: HashMap<ShaderHandle, sk::Shader>,
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
        let handle = ShaderHandle::new();
        self.0.borrow_mut().shaders.insert(handle.clone(), shader);
        handle
    }

    /// Remove a shader binding. Future semantic-scene replay using it fails.
    /// Existing native pictures and bound parent shaders remain valid.
    pub fn remove_shader(&self, handle: &ShaderHandle) -> bool {
        self.0.borrow_mut().shaders.remove(handle).is_some()
    }

    /// Remove compiled code. Previously bound shader instances remain valid.
    pub fn remove_program(&self, program: &ShaderProgram) -> bool {
        self.0.borrow_mut().programs.remove(program).is_some()
    }

    pub(crate) fn resolve(&self, handle: &ShaderHandle) -> Option<sk::Shader> {
        self.0.borrow().shaders.get(handle).cloned()
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
        let effect = sk::RuntimeEffect::make_for_shader(source, Some(&options))
            .map_err(ShaderError::Compilation)?;
        let uniforms = effect
            .uniforms()
            .iter()
            .map(|uniform| UniformDesc {
                name: uniform.name().into(),
                ty: uniform_type(uniform.ty()),
                count: usize::try_from(uniform.count())
                    .expect("Skia uniform counts are nonnegative"),
                is_array: uniform.is_array(),
                is_color: uniform.is_color(),
            })
            .collect();
        let children = effect
            .children()
            .iter()
            .map(|child| {
                if child.ty() != ChildType::Shader {
                    return Err(ShaderError::InvalidChild {
                        name: child.name().into(),
                        reason: "only shader children are supported",
                    });
                }
                Ok(child.name().into())
            })
            .collect::<Result<_, _>>()?;
        let program = ShaderProgram::new();
        self.0.borrow_mut().programs.insert(
            program.clone(),
            Program {
                effect,
                layout: ShaderLayout { uniforms, children },
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
                    .get(handle)
                    .ok_or_else(|| ShaderError::UnboundShader(handle.clone()))?;
                *slot = Some(ChildPtr::Shader(shader.clone()));
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
            effect
                .make_shader(sk::Data::new_copy(&data), &children, None)
                .ok_or(ShaderError::CreateShader)?
        };
        Ok(self.register_shader(shader))
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
