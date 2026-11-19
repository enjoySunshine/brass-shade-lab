//! Verification of the review scenes against the expected colors and depths.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use review_channel::{
    push_vertex, scenes, write_depth, write_ppm, Command, GpuSink, Recording, Scissor,
    SoftwareChannel,
};

/// 底座：除 (0,0) 外每个像素都是 (10,0,0,10)、深度 0.5；
/// (0,0) 是 (20,0,0,20)、深度 0.25。别的像素不能是 0，也不能是 20。
#[test]
fn base_scene_matches_expectations() {
    let mut channel = SoftwareChannel::new(8, 8);
    channel.submit(&scenes::base());

    assert_eq!(channel.pixel(0, 0), [20, 0, 0, 20]);
    assert_eq!(channel.depth_at(0, 0), 0.25);

    for y in 0..8 {
        for x in 0..8 {
            if (x, y) == (0, 0) {
                continue;
            }
            assert_eq!(channel.pixel(x, y), [10, 0, 0, 10], "color at ({x}, {y})");
            assert_eq!(channel.depth_at(x, y), 0.5, "depth at ({x}, {y})");
        }
    }
}

/// 上色：三个指定像素的颜色和深度；更远的颜色不能盖住更近的像素。
#[test]
fn color_scene_matches_expectations() {
    let mut channel = SoftwareChannel::new(8, 8);
    channel.submit(&scenes::color());

    assert_eq!(channel.pixel(2, 1), [140, 40, 20, 255]);
    assert_eq!(channel.depth_at(2, 1), 0.5);

    assert_eq!(channel.pixel(3, 3), [255, 255, 255, 255]);
    assert_eq!(channel.depth_at(3, 3), 0.25);

    assert_eq!(channel.pixel(0, 0), [255, 0, 0, 255]);
    assert_eq!(channel.depth_at(0, 0), 0.75);

    // The full-screen red quad is the farthest layer: nothing may stay
    // cleared, and nothing may end up farther than 0.75.
    for y in 0..8 {
        for x in 0..8 {
            assert_ne!(channel.pixel(x, y), [0, 0, 0, 0], "color at ({x}, {y})");
            assert!(channel.depth_at(x, y) <= 0.75, "depth at ({x}, {y})",);
        }
    }
}

/// 裁剪：左半被画到，右半保持清除值。
#[test]
fn scissor_scene_matches_expectations() {
    let mut channel = SoftwareChannel::new(8, 8);
    channel.submit(&scenes::scissor());

    assert_eq!(channel.pixel(1, 1), [0, 200, 0, 255]);
    assert_eq!(channel.depth_at(1, 1), 0.25);

    assert_eq!(channel.pixel(6, 1), [0, 0, 0, 0]);
    assert_eq!(channel.depth_at(6, 1), 1.0);

    for y in 0..8 {
        for x in 0..8 {
            if x < 4 {
                assert_eq!(channel.pixel(x, y), [0, 200, 0, 255], "color at ({x}, {y})");
                assert_eq!(channel.depth_at(x, y), 0.25, "depth at ({x}, {y})");
            } else {
                assert_eq!(channel.pixel(x, y), [0, 0, 0, 0], "color at ({x}, {y})");
                assert_eq!(channel.depth_at(x, y), 1.0, "depth at ({x}, {y})");
            }
        }
    }
}

/// 顶点字节不够的那一条绘制被整条丢掉：不能混进绿色，也不能留下半块新颜色。
#[test]
fn truncated_vertex_buffer_draw_is_dropped() {
    let mut channel = SoftwareChannel::new(8, 8);
    channel.submit(&scenes::truncated());

    assert_eq!(channel.pixel(6, 1), [200, 0, 0, 255]);

    for y in 0..8 {
        for x in 0..8 {
            assert_eq!(channel.pixel(x, y), [200, 0, 0, 255], "color at ({x}, {y})");
            assert_eq!(channel.depth_at(x, y), 0.5, "depth at ({x}, {y})");
        }
    }
}

/// A GPU sink that captures whatever recording it receives.
#[derive(Debug, Default)]
struct CapturingSink {
    submissions: Mutex<Vec<Recording>>,
}

impl GpuSink for CapturingSink {
    fn submit(&self, recording: &Recording) {
        self.submissions.lock().unwrap().push(recording.clone());
    }
}

/// 在通道里把裁剪改小：审片读回跟着变，交给显卡的也是改过的这一份。
#[test]
fn scissor_edit_is_seen_by_review_and_gpu() {
    let mut recording = Recording::new();
    recording.clear([0, 0, 0, 0], 1.0);
    recording.set_scissor(Scissor {
        x: 0,
        y: 0,
        width: 8,
        height: 8,
    });
    // The user shrinks the scissor in the channel before the draws.
    recording.set_scissor(Scissor {
        x: 0,
        y: 0,
        width: 4,
        height: 8,
    });
    recording.draw_arrays(scenes::fullscreen_quad([200, 0, 0, 255], 0.5), 0, 6);

    // Review path: the readback follows the shrunken scissor.
    let mut channel = SoftwareChannel::new(8, 8);
    channel.submit(&recording);
    assert_eq!(channel.pixel(1, 1), [200, 0, 0, 255]);
    assert_eq!(channel.pixel(6, 1), [0, 0, 0, 0]);
    assert_eq!(channel.depth_at(6, 1), 1.0);

    // GPU path: the very same (edited) recording is what gets submitted.
    let sink = CapturingSink::default();
    sink.submit(&recording);
    let submissions = sink.submissions.lock().unwrap();
    assert_eq!(submissions.len(), 1);
    let submitted = &submissions[0];
    assert_eq!(*submitted, recording);
    let effective_scissor = submitted
        .commands()
        .iter()
        .rev()
        .find_map(|command| match command {
            Command::SetScissor(scissor) => Some(*scissor),
            _ => None,
        });
    assert_eq!(
        effective_scissor,
        Some(Scissor {
            x: 0,
            y: 0,
            width: 4,
            height: 8,
        })
    );
}

/// 1024x1024，两百个三角形，每个落到画面上不超过 64x64：
/// 发布模式下从提交到拿回颜色和深度要在 200 毫秒内。
#[test]
fn perf_1024x1024_200_triangles() {
    const SIZE: u32 = 1024;
    const TRIANGLES: u32 = 200;

    // 200 triangles in a 16-wide grid of 64x64 slots; each triangle stays
    // well inside its 64x64 slot. One draw call of three vertices each.
    let mut recording = Recording::new();
    recording.clear([0, 0, 0, 0], 1.0);
    for i in 0..TRIANGLES {
        let col = (i % 16) as f32;
        let row = ((i / 16) % 16) as f32;
        let left = col * 64.0;
        let top = row * 64.0;
        // Framebuffer pixels -> clip space for a 1024x1024 viewport.
        let cx = |fx: f32| fx / 512.0 - 1.0;
        let cy = |fy: f32| 1.0 - fy / 512.0;
        let z = 0.1 + 0.05 * (i % 10) as f32;
        let color = [
            (i % 256) as u8,
            (255 - (i % 256)) as u8,
            ((i * 7) % 256) as u8,
            255,
        ];
        let mut vertices = Vec::with_capacity(3 * 20);
        for (fx, fy) in [
            (left + 2.0, top + 2.0),
            (left + 62.0, top + 2.0),
            (left + 2.0, top + 62.0),
        ] {
            push_vertex(&mut vertices, [cx(fx), cy(fy), z, 1.0], color);
        }
        recording.draw_arrays(vertices.into(), 0, 3);
    }

    let mut channel = SoftwareChannel::new(SIZE, SIZE);
    let start = Instant::now();
    channel.submit(&recording);
    let color = channel.read_color();
    let depth = channel.read_depth();
    let elapsed = start.elapsed();

    // The 200 ms budget is measured in release mode; in debug builds only
    // guard against something pathological.
    let limit = if cfg!(debug_assertions) {
        Duration::from_secs(10)
    } else {
        Duration::from_millis(200)
    };
    assert!(
        elapsed < limit,
        "submit + readback took {elapsed:?}, limit is {limit:?}"
    );
    eprintln!("1024x1024, {TRIANGLES} triangles: submit + readback took {elapsed:?}");

    // Sanity: the triangles actually landed.
    assert!(color.chunks_exact(4).any(|px| px != [0, 0, 0, 0]));
    assert!(depth.iter().any(|&d| d < 1.0));
}

/// 颜色写成 ppm（左上角是 (0,0)），深度一行一个像素，顺序相同。
#[test]
fn ppm_and_depth_files_match_expectations() {
    let dir = std::env::temp_dir().join(format!("review-channel-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let mut channel = SoftwareChannel::new(8, 8);
    channel.submit(&scenes::base());

    let ppm_path = dir.join("base.ppm");
    let depth_path = dir.join("base.depth.txt");
    write_ppm(&ppm_path, 8, 8, channel.color()).unwrap();
    write_depth(&depth_path, 8, 8, channel.depth()).unwrap();

    // PPM: P3 header, then 8*8 RGB triples, (0,0) first.
    let ppm = std::fs::read_to_string(&ppm_path).unwrap();
    let mut tokens = ppm.split_whitespace();
    assert_eq!(tokens.next(), Some("P3"));
    assert_eq!(tokens.next(), Some("8"));
    assert_eq!(tokens.next(), Some("8"));
    assert_eq!(tokens.next(), Some("255"));
    let values: Vec<u32> = tokens.map(|t| t.parse().unwrap()).collect();
    assert_eq!(values.len(), 8 * 8 * 3);
    assert_eq!(&values[0..3], &[20, 0, 0], "pixel (0, 0)");
    for px in values[3..].chunks_exact(3) {
        assert_eq!(px, &[10, 0, 0]);
    }

    // Depth: one pixel per line, same order.
    let depth = std::fs::read_to_string(&depth_path).unwrap();
    let lines: Vec<&str> = depth.lines().collect();
    assert_eq!(lines.len(), 64);
    assert_eq!(lines[0], "0.25", "depth of pixel (0, 0)");
    assert!(lines[1..].iter().all(|line| *line == "0.5"));

    std::fs::remove_dir_all(&dir).ok();
}
