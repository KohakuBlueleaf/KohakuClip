//! Decoder configuration records (av1C / avcC / hvcC).

use std::io;

use super::{bad, be16};
use crate::storage::index::Codec;

/// Decoder prefix from an mp4 codec configuration record (av1C / avcC / hvcC payload): the AV1
/// sequence header, or the H.264 / HEVC parameter sets in Annex B.
pub fn codec_prefix(codec: Codec, config: &[u8]) -> io::Result<Vec<u8>> {
    const START: [u8; 4] = [0, 0, 0, 1];
    let mut out = Vec::new();
    let mut push = |nal: &[u8]| {
        out.extend_from_slice(&START);
        out.extend_from_slice(nal);
    };
    match codec {
        // 4-byte av1C header, then the configOBUs
        Codec::Av1 => return Ok(config.get(4..).unwrap_or_default().to_vec()),

        // avcC: SPS count in [5] & 31, each u16 length + NAL; then a u8 PPS count, same layout
        Codec::H264 => {
            if config[4] & 3 != 3 {
                return Err(bad("only 4-byte NAL lengths are supported"));
            }
            let mut i = 6;
            let mut count = (config[5] & 31) as usize;
            for list in 0..2 {
                for _ in 0..count {
                    let len = be16(config, i) as usize;
                    push(&config[i + 2..i + 2 + len]);
                    i += 2 + len;
                }
                if list == 0 {
                    count = config.get(i).copied().unwrap_or(0) as usize;
                    i += 1;
                }
            }
        }

        // hvcC: 22-byte header (length size in [21] & 3), then arrays of NAL units
        Codec::Hevc => {
            if config[21] & 3 != 3 {
                return Err(bad("only 4-byte NAL lengths are supported"));
            }
            let mut i = 23;
            for _ in 0..config[22] {
                let count = be16(config, i + 1) as usize;
                i += 3;
                for _ in 0..count {
                    let len = be16(config, i) as usize;
                    push(&config[i + 2..i + 2 + len]);
                    i += 2 + len;
                }
            }
        }
    }
    Ok(out)
}
