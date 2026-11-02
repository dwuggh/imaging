// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shader creation interfaces for `imaging` backends.
//!
//! Registries own compiled programs and immutable bindings. Draws use
//! [`imaging::ShaderBrush`] or [`imaging::Filter::Shader`] to reference the resulting
//! [`ShaderHandle`].

#![no_std]

extern crate alloc;

use alloc::{string::String, vec::Vec};
use core::fmt;
pub use imaging::{ShaderBrush, ShaderHandle};

/// Source accepted by a shader compiler.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum ShaderSource<'a> {
    /// Skia runtime source. The entry point depends on [`ShaderDesc::kind`].
    SkSl(&'a str),
}

/// Role of a compiled program and its bound instances.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShaderKind {
    /// A brush color source, with a `half4 main(float2)` entry point in `SkSL`.
    #[default]
    Shader,
    /// A color transformation, with a `half4 main(half4)` entry point in `SkSL`.
    ColorFilter,
    /// An image transformation using a shader entry point and a dynamic shader child.
    ImageFilter,
}

/// Description of a shader program to compile.
#[derive(Clone, Copy, Debug)]
pub struct ShaderDesc<'a> {
    /// Optional diagnostic name.
    pub label: Option<&'a str>,
    /// Program source; uniform declarations are reflected after compilation.
    pub source: ShaderSource<'a>,
    /// How instances of this program will be used.
    pub kind: ShaderKind,
}

/// Opaque identity of compiled code owned by a registry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ShaderProgram(ShaderHandle);

impl ShaderProgram {
    /// Allocate an unbound program identity for a registry implementation.
    #[must_use]
    pub fn new() -> Self {
        Self(ShaderHandle::new())
    }
}

/// Reflected uniform element type. Arrays repeat this element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniformType {
    /// One floating-point value.
    Float,
    /// Two floating-point values.
    Float2,
    /// Three floating-point values.
    Float3,
    /// Four floating-point values.
    Float4,
    /// A column-major 2 by 2 matrix.
    Float2x2,
    /// A column-major 3 by 3 matrix.
    Float3x3,
    /// A column-major 4 by 4 matrix.
    Float4x4,
    /// One signed integer.
    Int,
    /// Two signed integers.
    Int2,
    /// Three signed integers.
    Int3,
    /// Four signed integers.
    Int4,
}

/// A uniform declaration reflected from compiled source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniformDesc {
    /// Source-level name.
    pub name: String,
    /// Type of each element.
    pub ty: UniformType,
    /// Number of elements; one for non-array uniforms.
    pub count: usize,
    /// Whether the declaration is an array, including arrays of length one.
    pub is_array: bool,
    /// Whether values are unpremultiplied sRGB colors (`layout(color)` in `SkSL`).
    pub is_color: bool,
}

/// Bindings required by a compiled shader program.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderLayout {
    /// Role of this program.
    pub kind: ShaderKind,
    /// Required numeric uniforms.
    pub uniforms: Vec<UniformDesc>,
    /// Required children and their roles, in declaration order.
    pub children: Vec<ShaderChildDesc>,
}

/// A reflected child effect declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderChildDesc {
    /// Source-level name.
    pub name: String,
    /// Required instance kind; image filters cannot be used as children.
    pub kind: ShaderKind,
}

/// Borrowed numeric values, copied when a shader is bound.
#[derive(Clone, Copy, Debug)]
pub enum UniformValue<'a> {
    /// Float/half scalars, vectors, matrices, or arrays. Matrices are column-major.
    Float(&'a [f32]),
    /// Signed integer scalars, vectors, or arrays.
    Int(&'a [i32]),
}

/// Values used to create an immutable shader instance.
///
/// Supply every reflected uniform and child exactly once, except the automatic
/// image-filter input. Children must be bound in the same registry. Numeric slices
/// are never retained by reference.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShaderBindings<'a> {
    /// Uniform names and values.
    pub uniforms: &'a [(&'a str, UniformValue<'a>)],
    /// Child effect names and handles.
    pub children: &'a [(&'a str, ShaderHandle)],
    /// Dynamic input configuration, required only for image-filter instances.
    pub image_filter: Option<ImageFilterBinding<'a>>,
}

/// Input supplied automatically when an image filter is applied to a group.
#[derive(Clone, Copy, Debug)]
pub struct ImageFilterBinding<'a> {
    /// Name of the shader child receiving the preceding filter result or group content.
    pub input: &'a str,
    /// Maximum absolute sampling offset in either axis, in filter-local coordinates.
    /// Must be finite and nonnegative; the caller must bound all input samples by it.
    pub sample_radius: f32,
}

/// Compilation or binding failure.
#[derive(Clone, Debug)]
pub enum ShaderError {
    /// Native compiler diagnostics.
    Compilation(String),
    /// The registry cannot compile this source language.
    UnsupportedSource,
    /// The program is absent from this registry.
    UnboundProgram,
    /// A child effect is absent from this registry.
    UnboundShader(ShaderHandle),
    /// Invalid uniform name, type, count, or duplicate/missing binding.
    InvalidUniform {
        /// Uniform name.
        name: String,
        /// Explanation of the failure.
        reason: &'static str,
    },
    /// Invalid child declaration or duplicate/missing binding.
    InvalidChild {
        /// Child name.
        name: String,
        /// Explanation of the failure.
        reason: &'static str,
    },
    /// Missing, unexpected, or invalid image-filter input configuration.
    InvalidImageFilter(&'static str),
    /// The native backend could not instantiate the shader or filter.
    CreateShader,
}

impl fmt::Display for ShaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compilation(message) => write!(f, "shader compilation failed: {message}"),
            Self::UnsupportedSource => f.write_str("unsupported shader source language"),
            Self::UnboundProgram => f.write_str("program is not bound in this shader registry"),
            Self::UnboundShader(handle) => write!(f, "child shader is not bound: {handle:?}"),
            Self::InvalidUniform { name, reason } => write!(f, "uniform {name:?}: {reason}"),
            Self::InvalidChild { name, reason } => write!(f, "child {name:?}: {reason}"),
            Self::InvalidImageFilter(reason) => write!(f, "invalid image filter: {reason}"),
            Self::CreateShader => f.write_str("failed to create native shader or filter"),
        }
    }
}

impl core::error::Error for ShaderError {}

/// Compiler and binding interface implemented by a shader registry.
pub trait ShaderFactory {
    /// Compile reusable code and reflect its bindings.
    fn create_program(&self, desc: ShaderDesc<'_>) -> Result<ShaderProgram, ShaderError>;
    /// Inspect the declarations required by a program in this registry.
    fn layout(&self, program: &ShaderProgram) -> Result<ShaderLayout, ShaderError>;
    /// Copy bindings and create a new immutable shader or filter instance.
    fn bind_shader(
        &self,
        program: &ShaderProgram,
        bindings: ShaderBindings<'_>,
    ) -> Result<ShaderHandle, ShaderError>;
}
