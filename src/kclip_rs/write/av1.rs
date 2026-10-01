//! How much of each AV1 temporal unit later frames depend on (computed when writing a shard).
//!
//! A frame with `refresh_frame_flags == 0` stores nothing for later frames. SVT-AV1's random-access
//! hierarchy puts half of all frames in that top layer, usually as the shown frame at the end of a
//! temporal unit (one mp4 sample) after hidden reference frames. When a sample is not wanted, only
//! its bytes up to the last frame something depends on need decoding (`keep`; 0: none). This parses
//! the sequence header and each frame header up to `refresh_frame_flags` (AV1 spec 5.5, 5.9.2).

const OBU_SEQUENCE_HEADER: u8 = 1;
const OBU_FRAME_HEADER: u8 = 3;
const OBU_FRAME: u8 = 6;

const KEY_FRAME: u32 = 0;
const INTRA_ONLY: u32 = 2;
const SWITCH_FRAME: u32 = 3;

/// Value of a "seq_choose_*" field: the frame header carries the actual value.
const SELECT: u32 = 2;

/// Big-endian bit reader.
struct Bits<'a> {
    data: &'a [u8],
    bit: usize,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8], byte: usize) -> Self {
        Self {
            data,
            bit: byte * 8,
        }
    }

    fn f(&mut self, n: u32) -> u32 {
        let mut v = 0;
        for _ in 0..n {
            let byte = self.data.get(self.bit >> 3).copied().unwrap_or(0);
            v = (v << 1) | ((byte >> (7 - (self.bit & 7))) & 1) as u32;
            self.bit += 1;
        }
        v
    }

    fn flag(&mut self) -> bool {
        self.f(1) == 1
    }

    fn uvlc(&mut self) -> u32 {
        let mut zeros = 0;
        while !self.flag() {
            zeros += 1;
            if zeros == 32 {
                return u32::MAX;
            }
        }
        (1 << zeros) - 1 + self.f(zeros)
    }
}

/// One OBU: type, temporal / spatial layer, payload range.
struct Obu {
    kind: u8,
    temporal: u32,
    spatial: u32,
    start: usize,
    end: usize,
}

fn obus(data: &[u8]) -> impl Iterator<Item = Obu> + '_ {
    let mut i = 0;
    std::iter::from_fn(move || {
        if i >= data.len() {
            return None;
        }
        let header = data[i];
        let kind = (header >> 3) & 15;
        let has_extension = header & 4 != 0;
        let has_size = header & 2 != 0;
        let mut j = i + 1;
        let (mut temporal, mut spatial) = (0, 0);
        if has_extension {
            temporal = (data[j] >> 5) as u32;
            spatial = ((data[j] >> 3) & 3) as u32;
            j += 1;
        }
        let size = if has_size {
            let mut size = 0usize;
            let mut shift = 0;
            loop {
                let b = data[j];
                size |= ((b & 0x7F) as usize) << shift;
                j += 1;
                shift += 7;
                if b & 0x80 == 0 {
                    break;
                }
            }
            size
        } else {
            data.len() - j
        };
        i = j + size;
        Some(Obu {
            kind,
            temporal,
            spatial,
            start: j,
            end: j + size,
        })
    })
}

/// The sequence header fields the frame headers need.
struct Sequence {
    reduced_still_picture: bool,
    decoder_model: bool,
    equal_picture_interval: bool,
    frame_ids: bool,
    frame_id_bits: u32,
    order_hint_bits: u32,
    screen_content: u32,
    integer_mv: u32,
    removal_delay_bits: u32,
    presentation_delay_bits: u32,
    /// Per operating point: (idc, decoder model info present).
    operating_points: Vec<(u32, bool)>,
}

fn sequence_header(data: &[u8]) -> Option<Sequence> {
    let obu = obus(data).find(|o| o.kind == OBU_SEQUENCE_HEADER)?;
    let mut b = Bits::new(data, obu.start);
    let mut q = Sequence {
        reduced_still_picture: false,
        decoder_model: false,
        equal_picture_interval: false,
        frame_ids: false,
        frame_id_bits: 0,
        order_hint_bits: 0,
        screen_content: SELECT,
        integer_mv: SELECT,
        removal_delay_bits: 0,
        presentation_delay_bits: 0,
        operating_points: Vec::new(),
    };
    b.f(3); // seq_profile
    b.f(1); // still_picture
    q.reduced_still_picture = b.flag();
    if q.reduced_still_picture {
        return Some(q);
    }

    let mut delay_bits = 0;
    if b.flag() {
        // timing_info
        b.f(32);
        b.f(32);
        q.equal_picture_interval = b.flag();
        if q.equal_picture_interval {
            b.uvlc();
        }
        if b.flag() {
            // decoder_model_info
            q.decoder_model = true;
            delay_bits = b.f(5) + 1;
            b.f(32);
            q.removal_delay_bits = b.f(5) + 1;
            q.presentation_delay_bits = b.f(5) + 1;
        }
    }
    let display_delay = b.flag();
    for _ in 0..b.f(5) + 1 {
        let idc = b.f(12);
        let level = b.f(5);
        if level > 7 {
            b.f(1); // tier
        }
        let mut present = false;
        if q.decoder_model {
            present = b.flag();
            if present {
                b.f(delay_bits);
                b.f(delay_bits);
                b.f(1);
            }
        }
        if display_delay && b.flag() {
            b.f(4);
        }
        q.operating_points.push((idc, present));
    }

    let width_bits = b.f(4) + 1;
    let height_bits = b.f(4) + 1;
    b.f(width_bits);
    b.f(height_bits);
    q.frame_ids = b.flag();
    if q.frame_ids {
        let delta = b.f(4) + 2;
        let extra = b.f(3) + 1;
        q.frame_id_bits = delta + extra;
    }
    b.f(3); // 128x128 superblock, filter intra, intra edge filter
    b.f(4); // interintra compound, masked compound, warped motion, dual filter
    let order_hint = b.flag();
    if order_hint {
        b.f(2); // jnt comp, ref frame mvs
    }
    q.screen_content = if b.flag() { SELECT } else { b.f(1) };
    if q.screen_content > 0 {
        q.integer_mv = if b.flag() { SELECT } else { b.f(1) };
    }
    if order_hint {
        q.order_hint_bits = b.f(3) + 1;
    }
    Some(q)
}

/// Whether the frame header at `data[obu.start..]` stores anything for later frames (or shows a
/// stored one).
fn refreshes(q: &Sequence, data: &[u8], obu: &Obu) -> bool {
    let mut b = Bits::new(data, obu.start);
    if q.reduced_still_picture || b.flag() {
        // a key frame, or show_existing_frame
        return true;
    }
    let frame_type = b.f(2);
    let show = b.flag();
    if show && q.decoder_model && !q.equal_picture_interval {
        b.f(q.presentation_delay_bits);
    }
    if !show {
        b.f(1); // showable_frame
    }
    if frame_type == SWITCH_FRAME || (frame_type == KEY_FRAME && show) {
        return true; // refresh_frame_flags = all
    }

    let error_resilient = b.flag();
    b.f(1); // disable_cdf_update
    let screen_content = if q.screen_content == SELECT {
        b.f(1)
    } else {
        q.screen_content
    };
    if screen_content > 0 && q.integer_mv == SELECT {
        b.f(1); // force_integer_mv (read even for intra frames, then overridden)
    }
    if q.frame_ids {
        b.f(q.frame_id_bits);
    }
    b.f(1); // frame_size_override_flag (not a switch frame, not reduced)
    b.f(q.order_hint_bits);
    if frame_type != KEY_FRAME && frame_type != INTRA_ONLY && !error_resilient {
        b.f(3); // primary_ref_frame
    }
    if q.decoder_model && b.flag() {
        // buffer_removal_time_present
        for &(idc, present) in &q.operating_points {
            let in_layer = (idc >> obu.temporal) & 1 == 1 && (idc >> (obu.spatial + 8)) & 1 == 1;
            if present && (idc == 0 || in_layer) {
                b.f(q.removal_delay_bits);
            }
        }
    }
    b.f(8) != 0 // refresh_frame_flags
}

/// Per temporal unit (mp4 sample): the bytes to decode when its shown frame is not wanted, i.e.
/// up to the end of the last frame something depends on (a frame spans its header OBU up to the
/// next frame's). `None` if `sequence` holds no sequence header.
pub fn keep_bytes<'a>(
    sequence: &[u8],
    samples: impl IntoIterator<Item = &'a [u8]>,
) -> Option<Vec<u32>> {
    let q = sequence_header(sequence)?;
    let keep = samples
        .into_iter()
        .map(|data| {
            let mut keep = 0;
            let mut needed = false;
            for obu in obus(data) {
                // a new frame; tile group OBUs belong to the last one
                if obu.kind == OBU_FRAME || obu.kind == OBU_FRAME_HEADER {
                    needed = refreshes(&q, data, &obu);
                }
                if needed {
                    keep = obu.end;
                }
            }
            keep as u32
        })
        .collect();
    Some(keep)
}
