// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use kurbo::{Affine, Vec2};

/// Description of a filter effect applied to an isolated group.
///
/// Coordinates and standard deviations use the same `f64` precision as [`kurbo`] geometry.
/// Offsets must be finite; standard deviations must be finite and nonnegative.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Filter {
    /// Runtime color or image filter bound in the renderer's shader registry.
    ///
    /// The backend checks the instance kind. Image filters sample the previous filter
    /// result (or group content) in coordinates defined by `GroupRef::filter_transform`.
    Shader(crate::ShaderHandle),
    /// Fill the group output with a solid color (aka `feFlood`).
    ///
    /// This ignores the group's source content. It is still affected by the group's isolated clip.
    Flood {
        /// Flood color.
        color: peniko::Color,
    },
    /// Gaussian blur with separate X/Y standard deviation values in user space.
    ///
    /// [`crate::GroupRef::filter_transform`] maps the blur kernel to destination coordinates.
    Blur {
        /// Standard deviation along the X axis (in user space units).
        std_deviation_x: f64,
        /// Standard deviation along the Y axis (in user space units).
        std_deviation_y: f64,
    },
    /// Drop shadow under the source content.
    DropShadow {
        /// Shadow offset along the X axis (in user space units).
        dx: f64,
        /// Shadow offset along the Y axis (in user space units).
        dy: f64,
        /// Blur standard deviation along the X axis (in user space units).
        std_deviation_x: f64,
        /// Blur standard deviation along the Y axis (in user space units).
        std_deviation_y: f64,
        /// Shadow color.
        color: peniko::Color,
    },
    /// Translate the group output by a vector (aka `feOffset`).
    ///
    /// [`crate::GroupRef::filter_transform`] maps this vector to destination coordinates.
    /// Translation in that transform does not affect the vector.
    Offset {
        /// Offset along the X axis (in user space units).
        dx: f64,
        /// Offset along the Y axis (in user space units).
        dy: f64,
    },
}

impl Filter {
    /// Create a flood filter.
    #[inline]
    pub const fn flood(color: peniko::Color) -> Self {
        Self::Flood { color }
    }

    /// Create a uniform Gaussian blur filter.
    #[inline]
    pub const fn blur(sigma: f64) -> Self {
        Self::Blur {
            std_deviation_x: sigma,
            std_deviation_y: sigma,
        }
    }

    /// Create a Gaussian blur filter with separate X/Y sigma values.
    #[inline]
    pub const fn blur_xy(std_deviation_x: f64, std_deviation_y: f64) -> Self {
        Self::Blur {
            std_deviation_x,
            std_deviation_y,
        }
    }

    /// Create an offset/translation filter.
    #[inline]
    pub const fn offset(dx: f64, dy: f64) -> Self {
        Self::Offset { dx, dy }
    }

    /// Express this effect in destination coordinates using the transform's linear part.
    ///
    /// Returns `None` if the transformed blur kernel is not axis-aligned. A rotated
    /// anisotropic blur or a sheared blur can require a non-axis-aligned kernel.
    /// Backends with native affine filter support may apply those kernels directly instead.
    /// Opaque shader filters always return `None`; only the backend can transform them.
    ///
    /// Inputs must pass [`crate::record::Scene::validate`] or
    /// [`crate::validation::ValidatingSink`] before conversion.
    #[must_use]
    pub fn in_device_space(self, transform: Affine) -> Option<Self> {
        let [a, b, c, d, ..] = transform.as_coeffs();
        let offset = |dx, dy| (a * dx + c * dy, b * dx + d * dy);
        let blur = |sx, sy| {
            // Rows of A * diag(sx, sy). Their dot product is the off-diagonal
            // covariance; their lengths are the destination standard deviations.
            let x = Vec2::new(a * sx, c * sy);
            let y = Vec2::new(b * sx, d * sy);
            let (sx, sy) = (x.hypot(), y.hypot());
            // Allow cancellation error when a rotation leaves the covariance diagonal.
            if sx != 0.0 && sy != 0.0 && (x / sx).dot(y / sy).abs() > 32.0 * f64::EPSILON {
                return None;
            }
            Some((sx, sy))
        };
        Some(match self {
            Self::Shader(_) => return None,
            Self::Flood { .. } => self,
            Self::Offset { dx, dy } => {
                let (dx, dy) = offset(dx, dy);
                Self::Offset { dx, dy }
            }
            Self::Blur {
                std_deviation_x,
                std_deviation_y,
            } => {
                let (std_deviation_x, std_deviation_y) = blur(std_deviation_x, std_deviation_y)?;
                Self::Blur {
                    std_deviation_x,
                    std_deviation_y,
                }
            }
            Self::DropShadow {
                dx,
                dy,
                std_deviation_x,
                std_deviation_y,
                color,
            } => {
                let (dx, dy) = offset(dx, dy);
                let (std_deviation_x, std_deviation_y) = blur(std_deviation_x, std_deviation_y)?;
                Self::DropShadow {
                    dx,
                    dy,
                    std_deviation_x,
                    std_deviation_y,
                    color,
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_filter_identity_survives_replay_and_nested_masks() {
        use crate::{GroupRef, MaskMode, MaskRef, Painter, ShaderBrush, ShaderHandle, record};

        fn require_copy<T: Copy>(value: T) -> T {
            value
        }
        let handle = require_copy(ShaderHandle::new());
        let brush = require_copy(ShaderBrush { shader: handle });
        let filter = require_copy(Filter::Shader(handle));
        assert_eq!(brush.shader, handle);
        assert_ne!(handle, ShaderHandle::new());
        assert_eq!(filter.in_device_space(Affine::translate((3.0, 4.0))), None);

        let mut mask = record::Scene::new();
        Painter::new(&mut mask).with_group(GroupRef::new().with_filters(&[filter]), |p| {
            p.fill_rect(kurbo::Rect::new(0.0, 0.0, 8.0, 8.0), peniko::Color::WHITE);
        });
        assert!(mask.has_shader_filters());
        assert!(!mask.has_shader_brushes());
        let mut scene = record::Scene::new();
        Painter::new(&mut scene).with_group(
            GroupRef::new().with_mask(MaskRef::new(MaskMode::Alpha, &mask)),
            |p| p.fill_rect(kurbo::Rect::new(0.0, 0.0, 8.0, 8.0), peniko::Color::WHITE),
        );
        assert!(scene.has_shader_filters());
        scene.validate().unwrap();
        let mut copy = record::Scene::new();
        record::replay(&scene, &mut copy);
        assert_eq!(scene, copy);
        assert!(copy.has_shader_filters());
        let record::Command::PushGroup(id) = mask.commands()[0] else {
            panic!("expected group")
        };
        assert_eq!(mask.group(id).filters, [Filter::Shader(handle)]);
    }

    #[test]
    fn offsets_use_vectors_and_blurs_use_covariance() {
        let transform = Affine::new([0.0, 2.0, -3.0, 0.0, 100.0, 200.0]);
        assert_eq!(
            Filter::offset(4.0, 5.0).in_device_space(transform),
            Some(Filter::offset(-15.0, 8.0))
        );
        assert_eq!(
            Filter::blur_xy(4.0, 5.0).in_device_space(transform),
            Some(Filter::blur_xy(15.0, 8.0))
        );
        assert_eq!(
            Filter::blur(2.0).in_device_space(Affine::rotate(0.7)),
            Some(Filter::blur(2.0))
        );
        assert_eq!(
            Filter::blur_xy(2.0, 3.0).in_device_space(Affine::scale_non_uniform(-2.0, 0.0)),
            Some(Filter::blur_xy(4.0, 0.0))
        );
    }

    #[test]
    fn shadow_transforms_offset_and_kernel() {
        let shadow = Filter::DropShadow {
            dx: 1.0,
            dy: 2.0,
            std_deviation_x: 3.0,
            std_deviation_y: 4.0,
            color: peniko::Color::BLACK,
        };
        assert_eq!(
            shadow.in_device_space(Affine::scale(2.0)),
            Some(Filter::DropShadow {
                dx: 2.0,
                dy: 4.0,
                std_deviation_x: 6.0,
                std_deviation_y: 8.0,
                color: peniko::Color::BLACK,
            })
        );
    }

    #[test]
    fn non_axis_aligned_kernels_are_not_representable() {
        assert_eq!(
            Filter::blur_xy(2.0, 1.0).in_device_space(Affine::rotate(0.7)),
            None
        );
        assert_eq!(
            Filter::blur(1.0).in_device_space(Affine::skew(0.5, 0.0)),
            None
        );
    }

    #[test]
    fn destination_parameters_keep_f64_range() {
        let large = f64::from(f32::MAX);
        assert_eq!(
            Filter::offset(large, 0.0).in_device_space(Affine::scale(2.0)),
            Some(Filter::offset(large * 2.0, 0.0))
        );
        assert_eq!(
            Filter::blur(large).in_device_space(Affine::scale(2.0)),
            Some(Filter::blur(large * 2.0))
        );
    }
}
