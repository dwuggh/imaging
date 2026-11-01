// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Compile `SkSL`, bind uniforms, and paint shader brushes without a GPU or window.
//!
//! Run: `cargo run -p imaging_skia --example sksl -- /tmp/imaging-sksl.png`

use imaging::{Brush, Composite, Painter, ShaderBrush, record::Scene};
use imaging_shader_ext::{
    ShaderBindings, ShaderDesc, ShaderError, ShaderFactory, ShaderSource, UniformValue,
};
use imaging_skia::{SkiaConfig, SkiaCpuRenderer, SkiaShaderRegistry};
use kurbo::{Affine, Rect, RoundedRect, Stroke};
use peniko::Color;
use skia_safe as sk;
use std::{error::Error, io, path::PathBuf};

const STRIPES: &str = r#"
    uniform float frequency;
    uniform float phase;
    layout(color) uniform half4 color_a;
    layout(color) uniform half4 color_b;

    half4 main(float2 p) {
        float t = 0.5 + 0.5 *
            sin((p.x + p.y * 0.5) * frequency + phase);
        half4 color = mix(color_a, color_b, t);
        // Color uniforms are unpremultiplied; shader output must be premultiplied.
        return half4(color.rgb * color.a, color.a);
    }
"#;

fn main() -> Result<(), Box<dyn Error>> {
    let output = std::env::args_os()
        .nth(1)
        .map_or_else(|| PathBuf::from("sksl.png"), PathBuf::from);
    let (width, height) = (720_u16, 360_u16);

    // Compile once. The separate registry owns the compiled program and shaders.
    let shaders = SkiaShaderRegistry::new();
    let program = shaders.create_program(ShaderDesc {
        label: Some("diagonal stripes"),
        source: ShaderSource::SkSl(STRIPES),
    })?;

    // Bind two immutable instances of the same program. Values are copied here;
    // creating another instance cannot change a brush already recorded in a scene.
    let bind = |phase| -> Result<Brush, ShaderError> {
        let shader = shaders.bind_shader(
            &program,
            ShaderBindings {
                uniforms: &[
                    ("frequency", UniformValue::Float(&[0.055])),
                    ("phase", UniformValue::Float(&[phase])),
                    ("color_a", UniformValue::Float(&[0.08, 0.65, 0.95, 1.0])),
                    ("color_b", UniformValue::Float(&[0.95, 0.25, 0.40, 1.0])),
                ],
                children: &[],
            },
        )?;
        Ok(Brush::Shader(ShaderBrush { shader }))
    };
    let fill = bind(0.0)?;
    let stroke = bind(2.4)?;

    let mut scene = Scene::new();
    {
        let mut painter = Painter::new(&mut scene);
        painter.fill_rect(
            Rect::new(0.0, 0.0, f64::from(width), f64::from(height)),
            Color::from_rgb8(0x10, 0x14, 0x20),
        );
        painter
            .fill(RoundedRect::new(24.0, 24.0, 344.0, 336.0, 24.0), &fill)
            .brush_transform(Some(Affine::translate((24.0, 24.0))))
            .draw();
        painter
            .stroke(
                RoundedRect::new(392.0, 48.0, 672.0, 312.0, 36.0),
                &Stroke::new(28.0),
                &stroke,
            )
            .brush_transform(Some(Affine::translate((392.0, 48.0))))
            .composite(Composite {
                alpha: 0.85,
                ..Composite::default()
            })
            .draw();
    }

    // The scene owns handles only. Give its registry to the renderer for lookup.
    let config = SkiaConfig::new().with_shader_registry(shaders);
    let mut renderer = SkiaCpuRenderer::new_with_config(config);
    let pixels = renderer.render_scene(&scene, width, height)?;

    // Encode the renderer's unpremultiplied RGBA8 output with Skia's PNG encoder.
    let info = sk::ImageInfo::new(
        (i32::from(width), i32::from(height)),
        sk::ColorType::RGBA8888,
        sk::AlphaType::Unpremul,
        None,
    );
    let image = sk::images::raster_from_data(
        &info,
        sk::Data::new_copy(&pixels.data),
        usize::from(width) * 4,
    )
    .ok_or_else(|| io::Error::other("could not create PNG source image"))?;
    let png = image
        .encode(None, sk::EncodedImageFormat::PNG, None)
        .ok_or_else(|| io::Error::other("could not encode PNG"))?;
    std::fs::write(&output, png.as_bytes())?;
    println!("Wrote {}", output.display());
    Ok(())
}
