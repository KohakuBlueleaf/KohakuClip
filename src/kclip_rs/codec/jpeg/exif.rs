//! The EXIF orientation of a JPEG (APP1 "Exif" segment, IFD0 tag 0x0112).

const SOI: [u8; 2] = [0xFF, 0xD8];
const APP1: u8 = 0xE1;
const SOS: u8 = 0xDA;
const ORIENTATION: u16 = 0x0112;

/// The EXIF orientation (1-8; 1 = as stored) of a JPEG; 1 when it has none.
pub fn orientation(jpeg: &[u8]) -> u8 {
    if !jpeg.starts_with(&SOI) {
        return 1;
    }
    let mut at = 2;
    // marker segments up to the first scan: FF <marker> <u16 length incl. itself> <payload>
    while at + 4 <= jpeg.len() && jpeg[at] == 0xFF {
        let marker = jpeg[at + 1];
        if marker == SOS {
            break;
        }
        let len = u16::from_be_bytes([jpeg[at + 2], jpeg[at + 3]]) as usize;
        let payload = jpeg.get(at + 4..at + 2 + len).unwrap_or_default();
        if marker == APP1 && payload.starts_with(b"Exif\0\0") {
            return tiff_orientation(&payload[6..]).unwrap_or(1);
        }
        at += 2 + len;
    }
    1
}

/// Tag 0x0112 of IFD0 in a TIFF structure (either byte order).
fn tiff_orientation(tiff: &[u8]) -> Option<u8> {
    let little = match tiff.get(..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |at: usize| {
        let b = [*tiff.get(at)?, *tiff.get(at + 1)?];
        Some(if little {
            u16::from_le_bytes(b)
        } else {
            u16::from_be_bytes(b)
        })
    };
    let u32_at = |at: usize| {
        let b: [u8; 4] = tiff.get(at..at + 4)?.try_into().ok()?;
        Some(if little {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    };

    let ifd = u32_at(4)? as usize;
    let entries = u16_at(ifd)? as usize;
    for k in 0..entries {
        let entry = ifd + 2 + 12 * k;
        if u16_at(entry)? == ORIENTATION {
            // a SHORT stored in the first two bytes of the value field
            let value = u16_at(entry + 8)?;
            return (1..=8).contains(&value).then_some(value as u8);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal JPEG prefix: SOI, then an APP1 Exif segment with one IFD0 entry.
    fn with_orientation(value: u16, little: bool) -> Vec<u8> {
        let mut tiff = Vec::new();
        let u16b = |v: u16| {
            if little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            }
        };
        let u32b = |v: u32| {
            if little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            }
        };
        tiff.extend_from_slice(if little { b"II" } else { b"MM" });
        tiff.extend_from_slice(&u16b(42));
        tiff.extend_from_slice(&u32b(8));
        tiff.extend_from_slice(&u16b(1));
        tiff.extend_from_slice(&u16b(ORIENTATION));
        tiff.extend_from_slice(&u16b(3)); // SHORT
        tiff.extend_from_slice(&u32b(1));
        tiff.extend_from_slice(&u16b(value));
        tiff.extend_from_slice(&[0, 0]);

        let mut jpeg = SOI.to_vec();
        jpeg.extend_from_slice(&[0xFF, APP1]);
        jpeg.extend_from_slice(&((2 + 6 + tiff.len()) as u16).to_be_bytes());
        jpeg.extend_from_slice(b"Exif\0\0");
        jpeg.extend_from_slice(&tiff);
        jpeg.extend_from_slice(&[0xFF, SOS, 0, 2]);
        jpeg
    }

    #[test]
    fn reads_both_byte_orders() {
        for value in 1..=8 {
            assert_eq!(orientation(&with_orientation(value, true)), value as u8);
            assert_eq!(orientation(&with_orientation(value, false)), value as u8);
        }
        assert_eq!(orientation(&SOI), 1);
        assert_eq!(orientation(b"not a jpeg"), 1);
    }
}
