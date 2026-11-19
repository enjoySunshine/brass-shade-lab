//! The review-machine path: executes a [`Recording`] on the CPU and keeps the
//! color and depth attachments around for readback.

use crate::{
    Blend, Command, Compare, DepthState, DrawArrays, DrawIndexed, Recording, Scissor, Viewport,
    VERTEX_STRIDE,
};

/// A recorded channel rendered by the CPU into owned color/depth attachments.
///
/// This is what the review machine (no GPU) uses: `submit` executes the
/// recording itself — no llvmpipe / swiftshader / mesa, no window — and the
/// resulting attachments can be read back with [`SoftwareChannel::read_color`]
/// and [`SoftwareChannel::read_depth`].
///
/// State set by a command (viewport, scissor, blend, depth) stays in effect
/// for the rest of the submission until it is recorded again. Each
/// [`submit`](SoftwareChannel::submit) starts from the default state:
/// full-target viewport with depth range `0..=1`, full-target scissor,
/// [`Blend::Replace`], and depth compare [`Compare::Less`] with writes on.
#[derive(Debug)]
pub struct SoftwareChannel {
    width: u32,
    height: u32,
    /// RGBA8, row-major, pixel (0, 0) at the top left.
    color: Vec<u8>,
    /// `f32`, row-major, pixel (0, 0) at the top left.
    depth: Vec<f32>,
}

/// State that persists across the draws of one submission.
#[derive(Clone, Copy, Debug)]
struct State {
    viewport: Viewport,
    scissor: Scissor,
    blend: Blend,
    depth: DepthState,
}

/// One vertex decoded from its 20 recorded bytes.
#[derive(Clone, Copy, Debug)]
struct Vertex {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
    color: [u8; 4],
}

impl Vertex {
    fn parse(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), VERTEX_STRIDE);
        let f32_at = |offset: usize| {
            f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        };
        Vertex {
            x: f32_at(0),
            y: f32_at(4),
            z: f32_at(8),
            w: f32_at(12),
            color: [bytes[16], bytes[17], bytes[18], bytes[19]],
        }
    }
}

/// A vertex after the viewport transform, in `f64` so that edge functions of
/// `f32` inputs keep exact signs.
#[derive(Clone, Copy, Debug)]
struct Transformed {
    /// Framebuffer x (pixels, y grows downward).
    sx: f64,
    /// Framebuffer y (pixels).
    sy: f64,
    /// Clip-space `z / w`.
    nz: f64,
    /// `1 / w`, for perspective-correct attribute interpolation.
    inv_w: f64,
    /// Vertex color channels as `f64` in `0..=255`.
    color: [f64; 4],
}

impl Transformed {
    fn new(vertex: &Vertex, viewport: &Viewport) -> Option<Self> {
        if vertex.w == 0.0 {
            return None;
        }
        let inv_w = 1.0 / f64::from(vertex.w);
        let nx = f64::from(vertex.x) * inv_w;
        let ny = f64::from(vertex.y) * inv_w;
        let nz = f64::from(vertex.z) * inv_w;
        Some(Transformed {
            // Clip space is y-up with (-1, -1) at the bottom left; the
            // framebuffer is y-down with the origin at the top left.
            sx: f64::from(viewport.x) + (nx + 1.0) * 0.5 * f64::from(viewport.width),
            sy: f64::from(viewport.y) + (1.0 - ny) * 0.5 * f64::from(viewport.height),
            nz,
            inv_w,
            color: [
                f64::from(vertex.color[0]),
                f64::from(vertex.color[1]),
                f64::from(vertex.color[2]),
                f64::from(vertex.color[3]),
            ],
        })
    }
}

/// Twice the signed area of the triangle `a`, `b`, `(px, py)`.
fn edge(a: &Transformed, b: &Transformed, px: f64, py: f64) -> f64 {
    (b.sx - a.sx) * (py - a.sy) - (b.sy - a.sy) * (px - a.sx)
}

/// Pixel-center coverage for one edge: strictly inside, or exactly on a
/// top/left edge (so a shared edge is owned by exactly one of two triangles).
fn covered(e: f64, top_left: bool) -> bool {
    e > 0.0 || (e == 0.0 && top_left)
}

impl SoftwareChannel {
    /// Creates a channel with `width * height` color and depth attachments.
    pub fn new(width: u32, height: u32) -> Self {
        let pixels = width as usize * height as usize;
        SoftwareChannel {
            width,
            height,
            color: vec![0; pixels * 4],
            depth: vec![1.0; pixels],
        }
    }

    /// Width of the attachments, in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height of the attachments, in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Executes one recorded submission synchronously.
    ///
    /// A draw whose buffers do not hold the bytes it asks for (or whose
    /// vertex/index count is not a whole number of triangles) is dropped as a
    /// whole: it leaves no half-written colors behind, and whatever previous
    /// draws produced stays untouched.
    pub fn submit(&mut self, recording: &Recording) {
        let mut state = State {
            viewport: Viewport {
                x: 0.0,
                y: 0.0,
                width: self.width as f32,
                height: self.height as f32,
                near: 0.0,
                far: 1.0,
            },
            scissor: Scissor {
                x: 0,
                y: 0,
                width: self.width,
                height: self.height,
            },
            blend: Blend::Replace,
            depth: DepthState {
                compare: Compare::Less,
                write: true,
            },
        };
        for command in recording.commands() {
            match command {
                Command::Clear { color, depth } => {
                    for px in self.color.chunks_exact_mut(4) {
                        px.copy_from_slice(color);
                    }
                    self.depth.fill(*depth);
                }
                Command::SetViewport(viewport) => state.viewport = *viewport,
                Command::SetScissor(scissor) => state.scissor = *scissor,
                Command::SetBlend(blend) => state.blend = *blend,
                Command::SetDepth(depth) => state.depth = *depth,
                Command::DrawArrays(draw) => self.draw_arrays(&state, draw),
                Command::DrawIndexed(draw) => self.draw_indexed(&state, draw),
            }
        }
    }

    /// The color attachment: RGBA8, row-major, pixel (0, 0) at the top left.
    pub fn color(&self) -> &[u8] {
        &self.color
    }

    /// The depth attachment: `f32`, row-major, pixel (0, 0) at the top left.
    pub fn depth(&self) -> &[f32] {
        &self.depth
    }

    /// Reads back the whole color attachment.
    pub fn read_color(&self) -> Vec<u8> {
        self.color.clone()
    }

    /// Reads back the whole depth attachment.
    pub fn read_depth(&self) -> Vec<f32> {
        self.depth.clone()
    }

    /// The RGBA color of one pixel.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let idx = (y * self.width + x) as usize * 4;
        [
            self.color[idx],
            self.color[idx + 1],
            self.color[idx + 2],
            self.color[idx + 3],
        ]
    }

    /// The depth of one pixel.
    pub fn depth_at(&self, x: u32, y: u32) -> f32 {
        self.depth[(y * self.width + x) as usize]
    }

    fn draw_arrays(&mut self, state: &State, draw: &DrawArrays) {
        let first = draw.first_vertex as usize;
        let count = draw.vertex_count as usize;
        // Triangle list: only whole triangles can be drawn.
        if count == 0 || !count.is_multiple_of(3) {
            return;
        }
        let Some(vertex_end) = first.checked_add(count) else {
            return;
        };
        let Some(byte_end) = vertex_end.checked_mul(VERTEX_STRIDE) else {
            return;
        };
        if byte_end > draw.vertices.len() {
            // The buffer does not hold the bytes this draw asks for. Drop the
            // whole draw rather than leaving half a color behind.
            return;
        }
        let vertices: Vec<Vertex> = (first..vertex_end)
            .map(|i| Vertex::parse(&draw.vertices[i * VERTEX_STRIDE..(i + 1) * VERTEX_STRIDE]))
            .collect();
        for triangle in vertices.chunks_exact(3) {
            self.rasterize(state, [triangle[0], triangle[1], triangle[2]]);
        }
    }

    fn draw_indexed(&mut self, state: &State, draw: &DrawIndexed) {
        let count = draw.index_count as usize;
        if count == 0 || !count.is_multiple_of(3) || count > draw.indices.len() {
            return;
        }
        let indices = &draw.indices[..count];
        let available = draw.vertices.len() / VERTEX_STRIDE;
        if indices.iter().any(|&i| usize::from(i) >= available) {
            // An index points at a vertex that was never written: drop the draw.
            return;
        }
        let vertices: Vec<Vertex> = (0..available)
            .map(|i| Vertex::parse(&draw.vertices[i * VERTEX_STRIDE..(i + 1) * VERTEX_STRIDE]))
            .collect();
        for triangle in indices.chunks_exact(3) {
            self.rasterize(
                state,
                [
                    vertices[usize::from(triangle[0])],
                    vertices[usize::from(triangle[1])],
                    vertices[usize::from(triangle[2])],
                ],
            );
        }
    }

    fn rasterize(&mut self, state: &State, triangle: [Vertex; 3]) {
        let viewport = state.viewport;
        let Some(t0) = Transformed::new(&triangle[0], &viewport) else {
            return;
        };
        let Some(t1) = Transformed::new(&triangle[1], &viewport) else {
            return;
        };
        let Some(t2) = Transformed::new(&triangle[2], &viewport) else {
            return;
        };
        let mut v = [t0, t1, t2];

        // Normalize the winding so the interior is on the non-negative side of
        // every edge function (culling is off, so both windings are drawn).
        let area = edge(&v[0], &v[1], v[2].sx, v[2].sy);
        if area == 0.0 {
            return;
        }
        if area < 0.0 {
            v.swap(1, 2);
        }

        // Top-left rule: a pixel center exactly on an edge is covered only
        // when the edge is a top edge or a left edge. With the winding
        // normalized above, that is `dy < 0 || (dy == 0 && dx > 0)` along each
        // directed edge (framebuffer y grows downward).
        let mut top_left = [false; 3];
        for (i, flag) in top_left.iter_mut().enumerate() {
            let a = &v[(i + 1) % 3];
            let b = &v[(i + 2) % 3];
            let dx = b.sx - a.sx;
            let dy = b.sy - a.sy;
            *flag = dy < 0.0 || (dy == 0.0 && dx > 0.0);
        }

        let min_x = v[0].sx.min(v[1].sx).min(v[2].sx);
        let max_x = v[0].sx.max(v[1].sx).max(v[2].sx);
        let min_y = v[0].sy.min(v[1].sy).min(v[2].sy);
        let max_y = v[0].sy.max(v[1].sy).max(v[2].sy);

        // Intersect the bounding box with the scissor rect and the target.
        let scissor = state.scissor;
        let clip_x0 = i64::from(scissor.x.min(self.width));
        let clip_y0 = i64::from(scissor.y.min(self.height));
        let clip_x1 = i64::from(scissor.x.saturating_add(scissor.width).min(self.width));
        let clip_y1 = i64::from(scissor.y.saturating_add(scissor.height).min(self.height));

        // A pixel can only be covered when its center lies inside the bbox.
        let x_begin = ((min_x - 0.5).ceil() as i64).max(clip_x0);
        let x_end = ((max_x - 0.5).floor() as i64 + 1).min(clip_x1);
        let y_begin = ((min_y - 0.5).ceil() as i64).max(clip_y0);
        let y_end = ((max_y - 0.5).floor() as i64 + 1).min(clip_y1);
        if x_begin >= x_end || y_begin >= y_end {
            return;
        }

        let near = f64::from(viewport.near);
        let far = f64::from(viewport.far);

        // Flat-colored triangle: the interpolated color is the same at every
        // pixel, so it can be shaded once up front.
        let uniform_color = (v[0].color == v[1].color && v[1].color == v[2].color).then(|| {
            [
                v[0].color[0] as u8,
                v[0].color[1] as u8,
                v[0].color[2] as u8,
                v[0].color[3] as u8,
            ]
        });

        // Edge-function increments for stepping one pixel to the right.
        let inc0 = -(v[2].sy - v[1].sy);
        let inc1 = -(v[0].sy - v[2].sy);
        let inc2 = -(v[1].sy - v[0].sy);

        let width = self.width as usize;
        let color = &mut self.color;
        let depth_buffer = &mut self.depth;

        for py in y_begin..y_end {
            let cy = py as f64 + 0.5;
            let x_start = x_begin as f64 + 0.5;
            let mut e0 = edge(&v[1], &v[2], x_start, cy);
            let mut e1 = edge(&v[2], &v[0], x_start, cy);
            let mut e2 = edge(&v[0], &v[1], x_start, cy);

            let row = py as usize * width + x_begin as usize;
            let span = (x_end - x_begin) as usize;
            let depth_row = &mut depth_buffer[row..row + span];
            let color_row = &mut color[row * 4..(row + span) * 4];

            for (i, _px) in (x_begin..x_end).enumerate() {
                let inside = covered(e0, top_left[0])
                    && covered(e1, top_left[1])
                    && covered(e2, top_left[2]);
                if inside {
                    // Depth interpolates linearly in screen space; the edge
                    // functions sum to twice the triangle area.
                    let esum = e0 + e1 + e2;
                    let nz = (e0 * v[0].nz + e1 * v[1].nz + e2 * v[2].nz) / esum;
                    let depth = (near + nz * (far - near)) as f32;

                    if state.depth.compare.passes(depth, depth_row[i]) {
                        if state.depth.write {
                            depth_row[i] = depth;
                        }
                        let src = match uniform_color {
                            Some(c) => c,
                            None => {
                                // Perspective-correct weights for the color.
                                let cw0 = e0 * v[0].inv_w;
                                let cw1 = e1 * v[1].inv_w;
                                let cw2 = e2 * v[2].inv_w;
                                let csum = cw0 + cw1 + cw2;
                                if csum <= 0.0 {
                                    e0 += inc0;
                                    e1 += inc1;
                                    e2 += inc2;
                                    continue;
                                }
                                let inv_sum = 1.0 / csum;
                                let mut s = [0u8; 4];
                                for (k, sk) in s.iter_mut().enumerate() {
                                    let c = (cw0 * v[0].color[k]
                                        + cw1 * v[1].color[k]
                                        + cw2 * v[2].color[k])
                                        * inv_sum;
                                    *sk = c.round().clamp(0.0, 255.0) as u8;
                                }
                                s
                            }
                        };
                        let dst = &mut color_row[i * 4..i * 4 + 4];
                        match state.blend {
                            Blend::Replace => dst.copy_from_slice(&src),
                            Blend::Add => dst
                                .iter_mut()
                                .zip(src)
                                .for_each(|(d, s)| *d = d.saturating_add(s)),
                        }
                    }
                }
                e0 += inc0;
                e1 += inc1;
                e2 += inc2;
            }
        }
    }
}
