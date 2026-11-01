# Imaging Shader Ext

Shader creation and binding interfaces for `imaging` backends.

`ShaderFactory::create_program` compiles a `ShaderDesc`; `layout` exposes its reflected
uniforms and child shaders. `bind_shader` copies named bindings into an immutable
instance and returns an `imaging::ShaderHandle` for use in `Brush::Shader`.

The registry owns native resources. Recorded scenes retain handle identities only,
so keep their shader bindings registered until replay. New uniform values create a
new instance without changing earlier recordings.

`imaging_skia::SkiaShaderRegistry` provides the SkSL implementation. See the
[runnable example](../imaging_skia/examples/sksl.rs):

```sh
cargo run -p imaging_skia --example sksl -- /tmp/imaging-sksl.png
```

Disable default features and enable `libm` to use the interfaces without `std`.

Licensed under Apache-2.0 OR MIT, like the rest of the workspace.
