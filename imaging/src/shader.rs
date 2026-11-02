// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::{
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering},
};

/// Identity of an immutable shader or filter owned by a separate shader registry.
///
/// Copies identify the same instance. The handle contains no shader code or bindings and
/// does not keep registry resources alive. Rendering an unbound handle is an error.
/// Identities are process-local and never reused; they are not serialization keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShaderHandle(NonZeroUsize);

static NEXT_SHADER_ID: AtomicUsize = AtomicUsize::new(1);

impl ShaderHandle {
    /// Create a new, unbound shader identity.
    ///
    /// # Panics
    /// Panics if the process has exhausted the available identities.
    #[must_use]
    pub fn new() -> Self {
        let id = NEXT_SHADER_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("shader identities exhausted");
        Self(NonZeroUsize::new(id).expect("shader identities start at one"))
    }
}

impl Default for ShaderHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// Shader color source for a fill, stroke, or glyph run.
///
/// Use the draw's brush transform to position the shader and its composite alpha to
/// modulate opacity. Uniforms and child shaders are bound in the shader registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderBrush {
    /// Bound shader instance to evaluate.
    pub shader: ShaderHandle,
}
