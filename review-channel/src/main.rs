//! Renders the 8x8 review scenes on the CPU and writes the attachments out:
//! `<scene>.ppm` for color (top-left pixel first) and `<scene>.depth.txt`
//! with one depth value per line in the same order.
//!
//! Usage: `review-channel [output-dir]` (defaults to the current directory).

use std::io;
use std::path::Path;

use review_channel::{scenes, write_depth, write_ppm, SoftwareChannel};

fn main() -> io::Result<()> {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| String::from("."));
    std::fs::create_dir_all(&out_dir)?;

    let scene_list = [
        ("base", scenes::base()),
        ("color", scenes::color()),
        ("scissor", scenes::scissor()),
        ("truncated", scenes::truncated()),
    ];

    for (name, recording) in scene_list {
        let mut channel = SoftwareChannel::new(8, 8);
        channel.submit(&recording);

        let ppm_path = Path::new(&out_dir).join(format!("{name}.ppm"));
        let depth_path = Path::new(&out_dir).join(format!("{name}.depth.txt"));
        write_ppm(&ppm_path, 8, 8, channel.color())?;
        write_depth(&depth_path, 8, 8, channel.depth())?;
        eprintln!("wrote {} and {}", ppm_path.display(), depth_path.display());
    }

    Ok(())
}
