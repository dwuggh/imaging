// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::sync::Arc;
use core::{
    fmt,
    hash::{Hash, Hasher},
};

/// Identity of an immutable shader instance owned by a separate shader registry.
///
/// Clones identify the same shader. The handle contains no shader code or bindings and
/// does not keep registry resources alive. Rendering an unbound handle is an error.
#[derive(Clone)]
pub struct ShaderHandle(Arc<()>);

impl ShaderHandle {
    /// Create a new, unbound shader identity.
    #[must_use]
    pub fn new() -> Self {
        Self(Arc::new(()))
    }
}

impl Default for ShaderHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ShaderHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ShaderHandle")
            .field(&Arc::as_ptr(&self.0))
            .finish()
    }
}

impl PartialEq for ShaderHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for ShaderHandle {}
impl Hash for ShaderHandle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::ptr::hash(Arc::as_ptr(&self.0), state);
    }
}

/// Shader color source for a fill, stroke, or glyph run.
///
/// Use the draw's brush transform to position the shader and its composite alpha to
/// modulate opacity. Uniforms and child shaders are bound in the shader registry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderBrush {
    /// Bound shader instance to evaluate.
    pub shader: ShaderHandle,
}
