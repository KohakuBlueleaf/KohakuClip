//! Which frames to read. All return frame indices (display order) of a video with `n` frames.

use rand::{Rng, RngExt};

/// A contiguous clip of `frames` frames resampled to `target_fps` (None: every frame) at a random
/// start; frames repeat when the video is shorter than the clip.
pub fn clip(
    n: usize,
    fps: f64,
    frames: usize,
    target_fps: Option<f64>,
    rng: &mut impl Rng,
) -> Vec<u32> {
    let step = target_fps.map_or(1.0, |target| fps / target);
    let span = step * (frames as f64 - 1.0);
    let last = (n - 1) as f64;
    let start = rng.random::<f64>() * (last - span).max(0.0);
    (0..frames)
        .map(|k| ((start + k as f64 * step).round_ties_even() as u32).min(n as u32 - 1))
        .collect()
}

/// `frames` distinct frames drawn uniformly from the whole video, in display order.
pub fn random_frames(n: usize, frames: usize, rng: &mut impl Rng) -> Vec<u32> {
    let mut picked: Vec<u32> = rand::seq::index::sample(rng, n, frames.min(n))
        .into_iter()
        .map(|i| i as u32)
        .collect();
    picked.sort_unstable();
    picked
}
