// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Compare a shader brush, a color filter, an image filter, and a composed filter chain.
//!
//! Run: `cargo run -p imaging_skia --example sksl_filters -- /tmp/sksl-filters.png`
//! Everything is rendered through imaging; the bundled font needs no system installation.

use imaging::{
    Filter, GroupRef, Painter, ShaderBrush,
    record::{self, Glyph, Scene},
};
use imaging_shader_ext::{
    ImageFilterBinding, ShaderBindings, ShaderDesc, ShaderFactory, ShaderKind, ShaderSource,
    UniformValue,
};
use imaging_skia::{SkiaConfig, SkiaCpuRenderer, SkiaShaderRegistry};
use kurbo::{Affine, Circle, Line, Rect, RoundedRect, Stroke};
use peniko::{Blob, Color, Fill, FontData, Style};
use skia_safe as sk;
use std::{error::Error, io, path::PathBuf, sync::Arc};

const PAINT: &str = r#"
    uniform float phase;
    half4 main(float2 p) {
        float t = 0.5 + 0.5 * sin(p.x * 0.018 - p.y * 0.025 + phase);
        half3 color = mix(half3(0.04, 0.68, 0.90), half3(0.96, 0.17, 0.38), t);
        return half4(color, 1);
    }
"#;

// A color filter receives a premultiplied color, not a position or image.
// Preserve alpha while remapping luminance to two color-managed palette uniforms.
const DUOTONE: &str = r#"
    layout(color) uniform half4 dark;
    layout(color) uniform half4 light;
    uniform float strength;
    half4 main(half4 color) {
        if (color.a == 0) { return half4(0); }
        half3 rgb = color.rgb / color.a;
        half luminance = dot(rgb, half3(0.2126, 0.7152, 0.0722));
        half3 graded = mix(dark.rgb, light.rgb, luminance);
        return half4(mix(rgb, graded, strength) * color.a, color.a);
    }
"#;

// The backend binds the group's pixels (or preceding filter output) to scene_input.
// Both axes shift by at most amplitude; channel separation adds at most split.
const WARP: &str = r#"
    uniform shader scene_input;
    uniform float amplitude;
    uniform float frequency;
    uniform float phase;
    uniform float split;
    half4 main(float2 p) {
        float2 q = p + amplitude * float2(
            sin(p.y * frequency + phase),
            cos(p.x * frequency * 0.65 + phase));
        half4 r = scene_input.eval(q + float2(split, 0));
        half4 g = scene_input.eval(q);
        half4 b = scene_input.eval(q - float2(split, 0));
        half alpha = max(r.a, max(g.a, b.a));
        return half4(r.r, g.g, b.b, alpha);
    }
"#;

fn main() -> Result<(), Box<dyn Error>> {
    let output = std::env::args_os()
        .nth(1)
        .map_or_else(|| PathBuf::from("sksl-filters.png"), PathBuf::from);
    let shaders = SkiaShaderRegistry::new();
    let paint_program = shaders.create_program(ShaderDesc {
        label: Some("procedural ribbon"),
        source: ShaderSource::SkSl(PAINT),
        kind: ShaderKind::Shader,
    })?;
    let color_program = shaders.create_program(ShaderDesc {
        label: Some("duotone grade"),
        source: ShaderSource::SkSl(DUOTONE),
        kind: ShaderKind::ColorFilter,
    })?;
    let image_program = shaders.create_program(ShaderDesc {
        label: Some("wave and chromatic split"),
        source: ShaderSource::SkSl(WARP),
        kind: ShaderKind::ImageFilter,
    })?;

    // Every bind returns the same Copy handle type. The registry retains its native kind.
    let paint = shaders.bind_shader(
        &paint_program,
        ShaderBindings {
            uniforms: &[("phase", UniformValue::Float(&[0.6]))],
            ..ShaderBindings::default()
        },
    )?;
    let grade = shaders.bind_shader(
        &color_program,
        ShaderBindings {
            uniforms: &[
                ("dark", UniformValue::Float(&[0.08, 0.10, 0.26, 1.0])),
                ("light", UniformValue::Float(&[1.0, 0.80, 0.32, 1.0])),
                ("strength", UniformValue::Float(&[1.0])),
            ],
            ..ShaderBindings::default()
        },
    )?;
    let bind_warp = |phase| {
        shaders.bind_shader(
            &image_program,
            ShaderBindings {
                uniforms: &[
                    ("amplitude", UniformValue::Float(&[14.0])),
                    ("frequency", UniformValue::Float(&[0.045])),
                    ("phase", UniformValue::Float(&[phase])),
                    ("split", UniformValue::Float(&[4.0])),
                ],
                image_filter: Some(ImageFilterBinding {
                    input: "scene_input",
                    sample_radius: 18.0,
                }),
                ..ShaderBindings::default()
            },
        )
    };
    let warp = bind_warp(0.3)?;
    // A second immutable instance shares the compiled program, without changing warp.
    let second_warp = bind_warp(1.6)?;
    let labels = Labels::new()?;
    let artwork = artwork(&labels, ShaderBrush { shader: paint });
    let chain = [
        Filter::Shader(grade),
        Filter::Shader(second_warp),
        Filter::DropShadow {
            dx: 0.0,
            dy: 8.0,
            std_deviation_x: 8.0,
            std_deviation_y: 8.0,
            color: Color::from_rgba8(155, 114, 255, 150),
        },
    ];
    let panels: [(&str, &str, &[Filter]); 4] = [
        ("01 / ORIGINAL", "Procedural brush + paths + glyphs", &[]),
        (
            "02 / COLOR FILTER",
            "Luminance remapped to a duotone palette",
            &[Filter::Shader(grade)],
        ),
        (
            "03 / IMAGE FILTER",
            "Wave displacement + RGB channel separation",
            &[Filter::Shader(warp)],
        ),
        (
            "04 / COMPOSED",
            "Duotone > second wave instance > drop shadow",
            &chain,
        ),
    ];

    let (width, height) = (1120_u16, 940_u16);
    let mut scene = Scene::new();
    let mut p = Painter::new(&mut scene);
    p.fill_rect(
        Rect::new(0.0, 0.0, f64::from(width), f64::from(height)),
        Color::from_rgb8(12, 16, 25),
    );
    labels.draw(
        &mut p,
        "SkSL / One scene. Four outputs.",
        (32.0, 58.0),
        32.0,
        Color::WHITE,
    );
    labels.draw(
        &mut p,
        "Paint colors, transform pixels, then compose effects on the whole group.",
        (32.0, 92.0),
        17.0,
        Color::from_rgb8(155, 171, 195),
    );
    for (index, (title, description, filters)) in panels.iter().enumerate() {
        let x = if index % 2 == 0 { 32.0 } else { 572.0 };
        let y = if index < 2 { 126.0 } else { 506.0 };
        p.fill(
            RoundedRect::new(x, y, x + 516.0, y + 356.0, 16.0),
            Color::from_rgb8(23, 30, 43),
        )
        .draw();
        labels.draw(
            &mut p,
            title,
            (x + 20.0, y + 30.0),
            17.0,
            Color::from_rgb8(226, 233, 245),
        );
        labels.draw(
            &mut p,
            description,
            (x + 20.0, y + 52.0),
            13.0,
            Color::from_rgb8(149, 165, 188),
        );
        let placement = Affine::translate((x + 18.0, y + 74.0));
        // Artwork and filter coordinates move together. Filters run in the given order,
        // before the group's mask, isolated clip, opacity, and final blending.
        p.with_group(
            GroupRef::new()
                .with_filters(filters)
                .with_filter_transform(placement),
            |p| {
                record::replay_transformed(&artwork, p.sink_mut(), placement);
            },
        );
    }
    labels.draw(
        &mut p,
        "One Copy handle type. Three effect kinds. Immutable bindings, reusable programs.",
        (32.0, 906.0),
        16.0,
        Color::from_rgb8(155, 171, 195),
    );

    let config = SkiaConfig::new().with_shader_registry(shaders);
    let pixels = SkiaCpuRenderer::new_with_config(config).render_scene(&scene, width, height)?;
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
    .ok_or_else(|| io::Error::other("could not create PNG source"))?;
    let png = image
        .encode(None, sk::EncodedImageFormat::PNG, None)
        .ok_or_else(|| io::Error::other("could not encode PNG"))?;
    std::fs::write(&output, png.as_bytes())?;
    println!("Wrote {}", output.display());
    Ok(())
}

fn artwork(labels: &Labels, brush: ShaderBrush) -> Scene {
    let mut scene = Scene::new();
    let mut p = Painter::new(&mut scene);
    p.fill(RoundedRect::new(24.0, 14.0, 456.0, 244.0, 22.0), &brush)
        .draw();
    // Fine lines and rings make spatial distortion visible across multiple draw commands.
    for i in 0..13 {
        let x = 36.0 + f64::from(i) * 34.0;
        p.stroke(
            Line::new((x, 28.0), (x, 230.0)),
            &Stroke::new(1.0),
            Color::from_rgba8(255, 255, 255, 55),
        )
        .draw();
    }
    for radius in [72.0, 56.0, 40.0, 24.0] {
        p.stroke(
            Circle::new((348.0, 126.0), radius),
            &Stroke::new(5.0),
            Color::from_rgba8(255, 255, 255, 215),
        )
        .draw();
    }
    labels.draw(&mut p, "SkSL", (46.0, 139.0), 74.0, Color::WHITE);
    labels.draw(
        &mut p,
        "SHAPE / COLOR / MOTION",
        (49.0, 170.0),
        12.0,
        Color::WHITE,
    );
    for i in 0..10 {
        let x = 49.0 + f64::from(i) * 18.0;
        p.fill_rect(
            Rect::new(x, 192.0, x + 10.0, 217.0),
            if i % 2 == 0 {
                Color::WHITE
            } else {
                Color::from_rgb8(27, 37, 61)
            },
        );
    }
    scene
}

struct Labels {
    data: FontData,
    typeface: sk::Typeface,
}

impl Labels {
    fn new() -> Result<Self, io::Error> {
        let bytes: &[u8] = include_bytes!("../../test_assets/fonts/NotoSans-Regular.ttf");
        let typeface = sk::FontMgr::default()
            .new_from_data(sk::Data::new_copy(bytes), None)
            .ok_or_else(|| io::Error::other("could not load bundled font"))?;
        Ok(Self {
            data: FontData::new(Blob::new(Arc::new(bytes)), 0),
            typeface,
        })
    }

    fn draw(
        &self,
        p: &mut Painter<'_, Scene>,
        text: &str,
        origin: (f64, f64),
        size: f32,
        color: Color,
    ) {
        // ASCII labels need only glyph lookup and advances; this is not a text-shaping API.
        let font = sk::Font::from_typeface(self.typeface.clone(), size);
        let ids = font.str_to_glyphs_vec(text);
        let mut widths = vec![0.0; ids.len()];
        font.get_widths(&ids, &mut widths);
        let mut x = 0.0;
        let glyphs = ids.into_iter().zip(widths).map(|(id, width)| {
            let glyph = Glyph {
                id: u32::from(id),
                x,
                y: 0.0,
            };
            x += width;
            glyph
        });
        p.glyphs(&self.data, color)
            .font_size(size)
            .transform(Affine::translate(origin))
            .draw(&Style::Fill(Fill::NonZero), glyphs);
    }
}
