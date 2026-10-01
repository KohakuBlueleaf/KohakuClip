//! Planning one clip: which bytes to read, which packets to decode, where to crop.

use std::fs::File;
use std::sync::Arc;

use rand::{Rng, RngExt};

use crate::storage::index::{Codec, Frame, VideoMeta};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Resize + crop + RGB conversion on the CPU: uint8 [3, S, S] per frame.
    Rgb,
    /// The stored-resolution window as YUV 4:2:0; resize and conversion happen on the GPU.
    Yuv,
    /// The planes resized to the S x S crop (YUV 4:2:0); conversion happens on the GPU.
    YuvResized,
}

#[derive(Clone, Copy, Debug)]
pub struct Augment {
    pub random_crop: bool,
    /// Probability of a horizontal / vertical flip.
    pub hflip: f64,
    pub vflip: f64,
}

/// One GOP to decode: a byte range of the file, then its packets.
pub struct Group {
    pub offset: u64,
    pub len: usize,
    pub packets: Vec<Packet>,
}

pub struct Packet {
    /// Offset within the group's bytes.
    pub offset: usize,
    pub len: usize,
    /// Display index (comes back as the decoded frame's pts).
    pub frame: u32,
}

/// Where the output pixels come from: the frame conceptually resized to (nh, nw), then the
/// (oh, ow) window at (top, left).
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub nh: u32,
    pub nw: u32,
    pub top: u32,
    pub left: u32,
    pub oh: u32,
    pub ow: u32,
    pub hflip: bool,
    pub vflip: bool,
}

pub struct ClipPlan {
    pub file: Arc<File>,
    pub codec: Codec,
    pub prefix: Vec<u8>,
    pub groups: Vec<Group>,
    /// Requested frame of each output slot (may repeat).
    pub frames: Vec<u32>,
    pub geometry: Geometry,
    pub mode: Mode,
    pub colorspace: String,
}

impl ClipPlan {
    /// Number of distinct frames to decode and emit.
    pub fn wanted(&self) -> usize {
        let mut want = self.frames.clone();
        want.sort_unstable();
        want.dedup();
        want.len()
    }
}

/// Plan the clip `frames` of a video. `side`: the yuv window side (shared by the batch).
#[allow(clippy::too_many_arguments)]
pub fn plan(
    meta: &VideoMeta,
    table: &[Frame],
    file: Arc<File>,
    frames: &[u32],
    size: u32,
    mode: Mode,
    side: u32,
    augment: &Augment,
    skip: bool,
    rng: &mut impl Rng,
) -> ClipPlan {
    let mut want = frames.to_vec();
    want.sort_unstable();
    want.dedup();

    ClipPlan {
        file,
        codec: meta.codec,
        prefix: meta.prefix.clone(),
        groups: groups(table, &want, skip),
        frames: frames.to_vec(),
        geometry: geometry(meta, size, mode, side, augment, rng),
        mode,
        colorspace: meta.colorspace.clone(),
    }
}

/// One group per GOP holding wanted frames: from its keyframe to its last wanted frame.
/// Unwanted samples feed only the bytes later frames depend on (`keep`), and none if that is 0.
fn groups(table: &[Frame], want: &[u32], skip: bool) -> Vec<Group> {
    let keyframe_of = |frame: u32| {
        let mut k = frame as usize;
        while k > 0 && !table[k].is_key() {
            k -= 1;
        }
        k
    };

    let mut groups = Vec::new();
    let mut i = 0;
    while i < want.len() {
        let start = keyframe_of(want[i]);
        // the GOP's last wanted frame: wanted frames up to the next keyframe
        let mut last = want[i] as usize;
        while i < want.len() && keyframe_of(want[i]) == start {
            last = want[i] as usize;
            i += 1;
        }

        let base = table[start].off();
        let end = table[last].off() + table[last].size() as u64;
        let packets = (start..=last)
            .filter_map(|k| {
                let wanted = want.binary_search(&(k as u32)).is_ok();
                let len = if wanted || !skip {
                    table[k].size()
                } else {
                    table[k].keep()
                };
                (len > 0).then(|| Packet {
                    offset: (table[k].off() - base) as usize,
                    len: len as usize,
                    frame: k as u32,
                })
            })
            .collect();
        groups.push(Group {
            offset: base,
            len: (end - base) as usize,
            packets,
        });
    }
    groups
}

fn geometry(
    meta: &VideoMeta,
    size: u32,
    mode: Mode,
    side: u32,
    augment: &Augment,
    rng: &mut impl Rng,
) -> Geometry {
    // yuv: the stored-resolution window; otherwise the short side resized to `size`
    let (nh, nw, out) = if mode == Mode::Yuv {
        (meta.h, meta.w, side)
    } else {
        let scale = size as f64 / meta.h.min(meta.w) as f64;
        let resized = |x: u32| ((x as f64 * scale).round_ties_even() as u32).max(size);
        (resized(meta.h), resized(meta.w), size)
    };

    let (mut top, mut left) = if augment.random_crop {
        (
            rng.random_range(0..=nh - out),
            rng.random_range(0..=nw - out),
        )
    } else {
        ((nh - out) / 2, (nw - out) / 2)
    };
    if mode == Mode::Yuv {
        // the window is copied: its chroma must start on a chroma sample
        top &= !1;
        left &= !1;
    }

    Geometry {
        nh,
        nw,
        top,
        left,
        oh: out,
        ow: out,
        hflip: rng.random::<f64>() < augment.hflip,
        vflip: rng.random::<f64>() < augment.vflip,
    }
}
