//! A tiny recorded draw channel with two submission targets.
//!
//! The channel records a flat list of commands: viewport, scissor, blend,
//! depth, draw-arrays and draw-indexed. Every draw is a triangle list and
//! consumes its vertices three at a time. Vertices are 20 bytes each
//! (`x, y, z, w` as little-endian `f32`, then `r, g, b, a` as `u8`), indices
//! are 16-bit unsigned. The bytes in a buffer are exactly the bytes that were
//! written into it; the color attachment is 8-bit unsigned per channel and
//! the depth attachment is 32-bit float.
//!
//! * Channels that are **not** under review forward the recording to the
//!   graphics card through [`GpuSink`], unchanged.
//! * The review machine has no GPU, so [`SoftwareChannel`] executes the very
//!   same recording itself — no llvmpipe / swiftshader / mesa, no window —
//!   and keeps the resulting color and depth attachments for readback.
//!
//! Because both sides consume the same [`Recording`], editing the recording
//! (for example shrinking the scissor rect) is visible to the review readback
//! *and* to the record handed to the GPU.
//!
//! Coordinate conventions:
//!
//! * clip space: bottom-left is `(-1, -1)`;
//! * framebuffer: top-left is the origin, `y` grows downward;
//! * the viewport maps clip space onto the framebuffer, and its `near`/`far`
//!   define the depth range: `depth = near + (z / w) * (far - near)`;
//! * a pixel is covered when its center falls inside the triangle;
//! * when two triangles share an edge, a pixel on the edge belongs to exactly
//!   one of them (top-left rule).

use std::sync::Arc;

mod output;
mod software;

pub mod scenes;

pub use output::{write_depth, write_ppm};
pub use software::SoftwareChannel;

/// Number of bytes per vertex: `x, y, z, w` (`f32`) followed by `r, g, b, a` (`u8`).
pub const VERTEX_STRIDE: usize = 20;

/// Appends one vertex to a vertex buffer: four little-endian `f32`
/// (`x, y, z, w`) followed by four bytes (`r, g, b, a`), 20 bytes in total.
pub fn push_vertex(buffer: &mut Vec<u8>, position: [f32; 4], color: [u8; 4]) {
    let [x, y, z, w] = position;
    buffer.extend_from_slice(&x.to_le_bytes());
    buffer.extend_from_slice(&y.to_le_bytes());
    buffer.extend_from_slice(&z.to_le_bytes());
    buffer.extend_from_slice(&w.to_le_bytes());
    buffer.extend_from_slice(&color);
}

/// Maps clip space onto the framebuffer and defines the depth range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    /// Left edge of the viewport, in framebuffer pixels.
    pub x: f32,
    /// Top edge of the viewport, in framebuffer pixels.
    pub y: f32,
    /// Width of the viewport, in framebuffer pixels.
    pub width: f32,
    /// Height of the viewport, in framebuffer pixels.
    pub height: f32,
    /// Depth that clip-space `z / w == 0` maps to.
    pub near: f32,
    /// Depth that clip-space `z / w == 1` maps to.
    pub far: f32,
}

/// Axis-aligned scissor rectangle, in framebuffer pixels. Pixels outside it
/// are left untouched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scissor {
    /// Left edge of the rectangle.
    pub x: u32,
    /// Top edge of the rectangle.
    pub y: u32,
    /// Width of the rectangle, in pixels.
    pub width: u32,
    /// Height of the rectangle, in pixels.
    pub height: u32,
}

/// How a new fragment combines with the color already in the attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blend {
    /// The new color simply covers the old one.
    Replace,
    /// Source and destination are added together per channel; results above
    /// 255 are clamped to 255.
    Add,
}

/// Depth comparison used for the depth test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compare {
    /// Never passes.
    Never,
    /// Passes when the new depth is smaller.
    Less,
    /// Passes when the depths are equal.
    Equal,
    /// Passes when the new depth is smaller or equal.
    LessEqual,
    /// Passes when the new depth is larger.
    Greater,
    /// Passes when the depths differ.
    NotEqual,
    /// Passes when the new depth is larger or equal.
    GreaterEqual,
    /// Always passes.
    Always,
}

impl Compare {
    pub(crate) fn passes(self, new: f32, current: f32) -> bool {
        match self {
            Compare::Never => false,
            Compare::Less => new < current,
            Compare::Equal => new == current,
            Compare::LessEqual => new <= current,
            Compare::Greater => new > current,
            Compare::NotEqual => new != current,
            Compare::GreaterEqual => new >= current,
            Compare::Always => true,
        }
    }
}

/// Depth test and write state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthState {
    /// Comparison a fragment must pass to be written.
    pub compare: Compare,
    /// Whether a passing fragment also writes its depth into the attachment.
    pub write: bool,
}

/// Draws `vertex_count` vertices from a byte buffer, starting at
/// `first_vertex`. The count must be a multiple of three (triangle list).
#[derive(Clone, Debug, PartialEq)]
pub struct DrawArrays {
    /// Vertex bytes, exactly as written; [`VERTEX_STRIDE`] bytes per vertex.
    pub vertices: Arc<[u8]>,
    /// Index of the first vertex to use.
    pub first_vertex: u32,
    /// Number of vertices to draw.
    pub vertex_count: u32,
}

/// Draws `index_count` 16-bit indices into a vertex byte buffer. The count
/// must be a multiple of three (triangle list).
#[derive(Clone, Debug, PartialEq)]
pub struct DrawIndexed {
    /// Vertex bytes, exactly as written; [`VERTEX_STRIDE`] bytes per vertex.
    pub vertices: Arc<[u8]>,
    /// 16-bit indices into the vertex buffer.
    pub indices: Arc<[u16]>,
    /// Number of indices to draw.
    pub index_count: u32,
}

/// One recorded command of the channel.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// Clears the color attachment to `color` and the depth attachment to `depth`.
    Clear {
        /// RGBA color to clear to.
        color: [u8; 4],
        /// Depth value to clear to.
        depth: f32,
    },
    /// Sets the viewport (and with it the depth range).
    SetViewport(Viewport),
    /// Sets the scissor rectangle.
    SetScissor(Scissor),
    /// Sets the blend mode.
    SetBlend(Blend),
    /// Sets the depth state.
    SetDepth(DepthState),
    /// Draws vertices directly.
    DrawArrays(DrawArrays),
    /// Draws vertices through 16-bit indices.
    DrawIndexed(DrawIndexed),
}

/// The recorded commands of one submission.
///
/// This is the single source of truth for a submission: the review machine
/// executes it in software, a channel that is not under review hands this
/// exact recording to the GPU. Recording another state command (for example
/// a smaller scissor rect) changes what both sides see, because later
/// commands override earlier ones.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Recording {
    commands: Vec<Command>,
}

impl Recording {
    /// Starts an empty recording.
    pub fn new() -> Self {
        Self::default()
    }

    /// The recorded commands, in submission order.
    pub fn commands(&self) -> &[Command] {
        &self.commands
    }

    /// Records a clear of both attachments.
    pub fn clear(&mut self, color: [u8; 4], depth: f32) -> &mut Self {
        self.commands.push(Command::Clear { color, depth });
        self
    }

    /// Records a viewport change.
    pub fn set_viewport(&mut self, viewport: Viewport) -> &mut Self {
        self.commands.push(Command::SetViewport(viewport));
        self
    }

    /// Records a scissor change.
    pub fn set_scissor(&mut self, scissor: Scissor) -> &mut Self {
        self.commands.push(Command::SetScissor(scissor));
        self
    }

    /// Records a blend mode change.
    pub fn set_blend(&mut self, blend: Blend) -> &mut Self {
        self.commands.push(Command::SetBlend(blend));
        self
    }

    /// Records a depth state change.
    pub fn set_depth(&mut self, depth: DepthState) -> &mut Self {
        self.commands.push(Command::SetDepth(depth));
        self
    }

    /// Records a draw of `vertex_count` vertices from `vertices`.
    pub fn draw_arrays(
        &mut self,
        vertices: Arc<[u8]>,
        first_vertex: u32,
        vertex_count: u32,
    ) -> &mut Self {
        self.commands.push(Command::DrawArrays(DrawArrays {
            vertices,
            first_vertex,
            vertex_count,
        }));
        self
    }

    /// Records a draw of `index_count` 16-bit indices into `vertices`.
    pub fn draw_indexed(
        &mut self,
        vertices: Arc<[u8]>,
        indices: Arc<[u16]>,
        index_count: u32,
    ) -> &mut Self {
        self.commands.push(Command::DrawIndexed(DrawIndexed {
            vertices,
            indices,
            index_count,
        }));
        self
    }
}

/// The GPU-facing end of a channel that is **not** under review.
///
/// Implementations hand the recorded viewport, scissor and draw commands to
/// the graphics card exactly as recorded. The review machine has no GPU, so
/// it uses [`SoftwareChannel`] instead; both consume the same [`Recording`].
pub trait GpuSink {
    /// Forwards one submitted recording to the graphics card.
    fn submit(&self, recording: &Recording);
}
