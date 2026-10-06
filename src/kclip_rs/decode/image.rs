//! Decoding one planned image: one pread, then the codec's library (libjpeg-turbo, shrunk by the
//! IDCT when the output needs fewer pixels; libdav1d, libjxl, libwebp), then the output pixels.

use std::cell::RefCell;
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::emit::emit;
use super::profile::{self, DECODED, EMITTED, Stage, timed};
use crate::codec::{ImageCodec, decode_still, jpeg};
use crate::read::image_plan::{ImagePlan, ImageSettings, dct_shift, layout};
use crate::read::plan::Mode;

/// Per-thread buffers: the image's bytes, the resized planes of the rgb mode.
#[derive(Default)]
struct Buffers {
    bytes: Vec<u8>,
    scratch: Vec<u8>,
}

thread_local! {
    static BUFFERS: RefCell<Buffers> = RefCell::new(Buffers::default());
    static JPEG: RefCell<Option<jpeg::Decoder>> = const { RefCell::new(None) };
}

/// Decode `plan` into `out` (one output slot).
pub fn decode_image(
    plan: &ImagePlan,
    settings: &ImageSettings,
    out: &mut [u8],
) -> Result<(), String> {
    BUFFERS.with_borrow_mut(|b| {
        let entry = &plan.entry;
        timed(Stage::Read, || entry.read_into(&mut b.bytes)).map_err(|e| format!("read: {e}"))?;

        match entry.codec {
            ImageCodec::Jpeg => JPEG.with_borrow_mut(|slot| {
                let decoder = match slot {
                    Some(decoder) => decoder,
                    empty => empty.insert(jpeg::Decoder::new()?),
                };
                let header = decoder.header(&b.bytes)?;
                let output = layout(header.h, header.w, settings, plan);
                let shift = if settings.dct_scale && settings.mode != Mode::Yuv {
                    dct_shift(header.h, header.w, &output.grid)
                } else {
                    0
                };
                let decoded = timed(Stage::Decode, || decoder.decode(&b.bytes, shift))?;
                DECODED.fetch_add(1, Ordering::Relaxed);
                emit(&decoded, &output, out, &mut b.scratch)
            }),
            codec => {
                let start = Instant::now();
                decode_still(codec, &b.bytes, |decoded| {
                    profile::add(Stage::Decode, start.elapsed());
                    DECODED.fetch_add(1, Ordering::Relaxed);
                    let (h, w) = decoded.size();
                    let output = layout(h as u32, w as u32, settings, plan);
                    emit(decoded, &output, out, &mut b.scratch)
                })
            }
        }?;
        EMITTED.fetch_add(1, Ordering::Relaxed);
        Ok(())
    })
}
