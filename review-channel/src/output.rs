//! Readback output formats: PPM for color, one value per line for depth.

use std::fs;
use std::io;
use std::path::Path;

/// Writes the color attachment as an ASCII (`P3`) PPM image.
///
/// Pixels are written row by row starting with pixel (0, 0) at the top left,
/// so the first RGB triple in the file is the top-left pixel. PPM carries no
/// alpha channel; the alpha values are still available through
/// [`crate::SoftwareChannel::read_color`].
pub fn write_ppm(path: &Path, width: u32, height: u32, color_rgba: &[u8]) -> Result<(), io::Error> {
    assert_eq!(
        color_rgba.len(),
        width as usize * height as usize * 4,
        "color buffer size must match width * height * 4"
    );
    let mut text = String::new();
    text.push_str(&format!("P3\n{width} {height}\n255\n"));
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize * 4;
            text.push_str(&format!(
                "{} {} {}",
                color_rgba[idx],
                color_rgba[idx + 1],
                color_rgba[idx + 2]
            ));
            if x + 1 < width {
                text.push(' ');
            }
        }
        text.push('\n');
    }
    fs::write(path, text)
}

/// Writes the depth attachment as text: one pixel per line, in the same
/// order as the PPM (row by row, starting with pixel (0, 0) at the top left).
pub fn write_depth(path: &Path, width: u32, height: u32, depth: &[f32]) -> Result<(), io::Error> {
    assert_eq!(
        depth.len(),
        width as usize * height as usize,
        "depth buffer size must match width * height"
    );
    let mut text = String::new();
    for y in 0..height {
        for x in 0..width {
            text.push_str(&format!("{}\n", depth[(y * width + x) as usize]));
        }
    }
    fs::write(path, text)
}
