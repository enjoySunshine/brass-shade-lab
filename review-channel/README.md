# review-channel

A tiny recorded draw channel with two submission targets.

The channel records a flat list of commands — viewport, scissor, blend,
depth, draw-arrays, draw-indexed. Vertices are 20 bytes each (`x, y, z, w` as
little-endian `f32`, then `r, g, b, a` as `u8`); indices are 16-bit unsigned;
the color attachment is RGBA8 and the depth attachment is `f32`. Every draw
is a triangle list, consumed three vertices at a time.

- **Channels that are not under review** forward the recording to the
  graphics card through the `GpuSink` trait, unchanged.
- **The review machine has no GPU.** `SoftwareChannel` executes the very same
  recording on the CPU — no llvmpipe / swiftshader / mesa, no window — and
  keeps the color and depth attachments for readback (`read_color`,
  `read_depth`).

Both sides consume the same `Recording`, so editing the recording (for
example shrinking the scissor rect) is visible to the review readback *and*
to the record handed to the GPU.

## Conventions

- Clip space: bottom-left is `(-1, -1)`. Framebuffer: top-left origin, y down.
- The viewport maps clip space onto the framebuffer; depth comes from the
  viewport's near/far: `depth = near + (z / w) * (far - near)`.
- A pixel is covered when its center falls inside the triangle. Two triangles
  sharing an edge: a pixel on the edge belongs to exactly one of them
  (top-left rule).
- State recorded by a command (viewport, scissor, blend, depth) stays in
  effect for the rest of the submission.
- A draw whose buffers do not hold the bytes it asks for is dropped as a
  whole: no half-written colors, previous pixels stay.

## Review scenes

`cargo run -p review-channel -- <output-dir>` renders the 8x8 review scenes
and writes, per scene:

- `<scene>.ppm` — the color attachment as an ASCII (`P3`) PPM, pixel (0, 0)
  at the top left first (PPM carries no alpha; use `read_color` for RGBA);
- `<scene>.depth.txt` — the depth attachment, one pixel per line, same order.

`cargo test -p review-channel` checks the scenes against the expected colors
and depths, including the 1024x1024 / 200-triangle submission that must come
back within 200 ms in release mode (`cargo test --release -p review-channel`).
