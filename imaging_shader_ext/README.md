# Imaging Shader Ext

Shader creation and binding interfaces for `imaging` backends.

`ShaderFactory::create_program` compiles a `ShaderDesc`; `layout` exposes its reflected
uniforms and typed children. `ShaderDesc::kind` selects `Shader`, `ColorFilter`, or
`ImageFilter`. `bind_shader` copies named bindings into an immutable instance and
returns the same `Copy` `imaging::ShaderHandle` for `Brush::Shader` or `Filter::Shader`.
The backend checks that the instance kind matches its use.

For an image filter, supply `ShaderBindings::image_filter` with an
`ImageFilterBinding { input, sample_radius }`. The named shader child receives the
previous filter result, or the group's content when the filter is first. Do not
bind that child manually. The radius is the maximum absolute sampling offset in
either axis in filter-local coordinates; callers must bound their samples by it.

The Skia implementation currently supports one dynamic image-filter input and
non-array scalar/vector/matrix uniforms. Brush and color-filter programs also
support uniform arrays and shader/color-filter children, where Skia allows them.
Image filters cannot be child effects. New uniform values create new instances.

The registry owns native resources. Recorded scenes retain handle identities only,
so keep their shader bindings registered until replay. New uniform values create a
new instance without changing earlier recordings.

`imaging_skia::SkiaShaderRegistry` provides the SkSL implementation. See the
[runnable example](../imaging_skia/examples/sksl.rs):

```sh
cargo run -p imaging_skia --example sksl -- /tmp/imaging-sksl.png
cargo run -p imaging_skia --example sksl_filters -- /tmp/sksl-filters.png
```

Disable default features and enable `libm` to use the interfaces without `std`.

Licensed under Apache-2.0 OR MIT, like the rest of the workspace.
