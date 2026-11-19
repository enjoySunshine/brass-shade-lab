//! The review scenes: the exact submissions used to check the channel.
//!
//! Coordinates are clip space (bottom-left is `(-1, -1)`), colors are
//! `r, g, b, a` bytes. All scenes target an 8x8 attachment pair.

use std::sync::Arc;

use crate::{push_vertex, Blend, Compare, DepthState, Recording, Scissor, Viewport};

/// Packs `(position, color)` vertices into a vertex buffer.
fn vertex_buffer(vertices: &[([f32; 4], [u8; 4])]) -> Arc<[u8]> {
    let mut buffer = Vec::with_capacity(vertices.len() * crate::VERTEX_STRIDE);
    for (position, color) in vertices {
        push_vertex(&mut buffer, *position, *color);
    }
    buffer.into()
}

/// Two triangles covering the whole clip space, one flat color, depth `z`.
pub fn fullscreen_quad(color: [u8; 4], z: f32) -> Arc<[u8]> {
    vertex_buffer(&[
        ([-1.0, 1.0, z, 1.0], color),
        ([1.0, 1.0, z, 1.0], color),
        ([-1.0, -1.0, z, 1.0], color),
        ([1.0, 1.0, z, 1.0], color),
        ([1.0, -1.0, z, 1.0], color),
        ([-1.0, -1.0, z, 1.0], color),
    ])
}

fn viewport_8x8() -> Viewport {
    Viewport {
        x: 0.0,
        y: 0.0,
        width: 8.0,
        height: 8.0,
        near: 0.0,
        far: 1.0,
    }
}

fn scissor_full() -> Scissor {
    Scissor {
        x: 0,
        y: 0,
        width: 8,
        height: 8,
    }
}

fn depth_less_write() -> DepthState {
    DepthState {
        compare: Compare::Less,
        write: true,
    }
}

/// The base scene: additive blending over two full-screen triangles plus one
/// small triangle in the top-left corner.
///
/// Expected result: every pixel is `(10, 0, 0, 10)` at depth `0.5`, except
/// pixel (0, 0) which is `(20, 0, 0, 20)` at depth `0.25`. No pixel may be
/// `0` or `20` anywhere else — the shared edge of the two big triangles is
/// owned by exactly one of them.
pub fn base() -> Recording {
    let mut recording = Recording::new();
    recording
        .clear([0, 0, 0, 0], 1.0)
        .set_viewport(viewport_8x8())
        .set_scissor(scissor_full())
        .set_depth(depth_less_write())
        .set_blend(Blend::Add);

    let color = [10, 0, 0, 10];

    // First triangle, drawn by vertices.
    let first = vertex_buffer(&[
        ([-1.0, 1.0, 0.5, 1.0], color),
        ([1.0, 1.0, 0.5, 1.0], color),
        ([-1.0, -1.0, 0.5, 1.0], color),
    ]);
    recording.draw_arrays(first, 0, 3);

    // Second triangle, drawn by 16-bit indices. Blend and depth carry over.
    let second = vertex_buffer(&[
        ([1.0, 1.0, 0.5, 1.0], color),
        ([1.0, -1.0, 0.5, 1.0], color),
        ([-1.0, -1.0, 0.5, 1.0], color),
    ]);
    recording.draw_indexed(second, vec![0u16, 1, 2].into(), 3);

    // Small triangle in the top-left corner; blend is still additive.
    let corner = vertex_buffer(&[
        ([-1.0, 1.0, 0.25, 1.0], color),
        ([-0.6875, 1.0, 0.25, 1.0], color),
        ([-1.0, 0.6875, 0.25, 1.0], color),
    ]);
    recording.draw_arrays(corner, 0, 3);

    recording
}

/// The color scene: replace blending, interpolated vertex colors, and a
/// farther full-screen quad that must not cover nearer pixels.
///
/// Expected result: pixel (2, 1) is `(140, 40, 20, 255)` at depth `0.5`,
/// pixel (3, 3) is `(255, 255, 255, 255)` at depth `0.25`, pixel (0, 0) is
/// `(255, 0, 0, 255)` at depth `0.75`.
pub fn color() -> Recording {
    let mut recording = Recording::new();
    recording
        .clear([0, 0, 0, 0], 1.0)
        .set_viewport(viewport_8x8())
        .set_scissor(scissor_full())
        .set_blend(Blend::Replace)
        .set_depth(depth_less_write());

    // Interpolated colors: red, green and blue corners.
    let gradient = vertex_buffer(&[
        ([-0.75, 0.75, 0.5, 1.0], [200, 0, 0, 255]),
        ([0.75, 0.75, 0.5, 1.0], [0, 200, 0, 255]),
        ([0.0, -0.5, 0.5, 1.0], [0, 0, 200, 255]),
    ]);
    recording.draw_arrays(gradient, 0, 3);

    // Nearer white triangle.
    let white = [255, 255, 255, 255];
    let near = vertex_buffer(&[
        ([-0.5, 0.5, 0.25, 1.0], white),
        ([0.5, 0.5, 0.25, 1.0], white),
        ([0.0, -0.25, 0.25, 1.0], white),
    ]);
    recording.draw_arrays(near, 0, 3);

    // Farther red full-screen quad; blend still Replace, depth still Less.
    recording.draw_arrays(fullscreen_quad([255, 0, 0, 255], 0.75), 0, 6);

    recording
}

/// The scissor scene: only the scissor rect is set (4 pixels wide, full
/// height); viewport, blend and depth stay at the channel defaults
/// (full-target viewport, Replace, Less + write).
///
/// Expected result: pixel (1, 1) is `(0, 200, 0, 255)` at depth `0.25`;
/// pixel (6, 1) is still the clear value `(0, 0, 0, 0)` at depth `1.0`.
pub fn scissor() -> Recording {
    let mut recording = Recording::new();
    recording.clear([0, 0, 0, 0], 1.0).set_scissor(Scissor {
        x: 0,
        y: 0,
        width: 4,
        height: 8,
    });
    recording.draw_arrays(fullscreen_quad([200, 0, 0, 255], 0.5), 0, 6);
    // Scissor unchanged: the second quad only lands inside the same rect.
    recording.draw_arrays(fullscreen_quad([0, 200, 0, 255], 0.25), 0, 6);
    recording
}

/// The malformed-draw scene: a red quad, then a draw that asks for 3 vertices
/// while its buffer holds only the first 20 bytes (one vertex).
///
/// Expected result: the bad draw is dropped as a whole. Every pixel stays
/// `(200, 0, 0, 255)` at depth `0.5` — no green mixed in, no half-written
/// colors.
pub fn truncated() -> Recording {
    let mut recording = Recording::new();
    recording
        .clear([0, 0, 0, 0], 1.0)
        .set_scissor(scissor_full());
    recording.draw_arrays(fullscreen_quad([200, 0, 0, 255], 0.5), 0, 6);

    // Only one vertex (20 bytes) is actually in the buffer.
    let short = vertex_buffer(&[([-1.0, 1.0, 0.25, 1.0], [0, 200, 0, 255])]);
    recording.draw_arrays(short, 0, 3);

    recording
}
