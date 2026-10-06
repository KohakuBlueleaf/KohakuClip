//! Decoding a planned clip (one pread per GOP, a persistent decoder per thread, then the output
//! pixels of each wanted frame) or a planned image.

mod decoder;
mod emit;
mod frame;
mod image;
pub mod profile;

use std::os::unix::fs::FileExt;
use std::sync::atomic::Ordering;

use rsmpeg::avcodec::AVCodecContext;
use rsmpeg::avutil::AVFrame;
use rsmpeg::ffi;

use crate::image::Grid;
use crate::read::plan::ClipPlan;
use crate::storage::index::Codec;
use decoder::{DECODERS, open_decoder, to_annexb};
pub use emit::{Output, emit};
pub use frame::{colorspace_name, picture};
pub use image::decode_image;
use profile::{DECODED, EMITTED, Stage, timed};

/// Where a clip's output pixels come from: its planned geometry as an output grid.
fn clip_output(plan: &ClipPlan) -> Output {
    let g = plan.geometry;
    Output {
        grid: Grid {
            nh: g.nh as f64,
            nw: g.nw as f64,
            top: g.top as f64,
            left: g.left as f64,
            oh: g.oh as usize,
            ow: g.ow as usize,
        },
        hflip: g.hflip,
        vflip: g.vflip,
        mode: plan.mode,
    }
}

/// Decode `plan` into `out` (all of the clip's output slots).
pub fn decode_clip(plan: &ClipPlan, out: &mut [u8]) -> Result<(), String> {
    DECODERS.with_borrow_mut(|d| {
        let slot_len = out.len() / plan.frames.len();
        let context = match &mut d.contexts[plan.codec as usize] {
            Some(context) => context,
            empty => empty.insert(open_decoder(plan.codec)?),
        };
        let nprefix = plan.prefix.len();
        let mut emitted = 0;

        for group in &plan.groups {
            // [prefix][group bytes]: the prefix joins the keyframe (group offset 0)
            d.bytes.clear();
            d.bytes.extend_from_slice(&plan.prefix);
            d.bytes.resize(nprefix + group.len, 0);
            timed(Stage::Read, || {
                plan.file
                    .read_exact_at(&mut d.bytes[nprefix..], group.offset)
            })
            .map_err(|e| format!("read: {e}"))?;
            // SAFETY: the context is open
            unsafe { ffi::avcodec_flush_buffers(context.as_mut_ptr()) };

            for (i, packet) in group.packets.iter().enumerate() {
                let at = nprefix + packet.offset;
                let sample = &mut d.bytes[at..at + packet.len];
                if plan.codec != Codec::Av1 && !to_annexb(sample) {
                    return Err("bad NAL unit lengths".into());
                }
                let head = if i == 0 { nprefix } else { 0 };
                // SAFETY: the packet borrows `d.bytes` (not reference counted, so the decoder
                // copies what it keeps); the bytes outlive the send call
                unsafe {
                    let raw = d.packet.as_mut_ptr();
                    (*raw).data = d.bytes.as_mut_ptr().add(at - head);
                    (*raw).size = (packet.len + head) as i32;
                    (*raw).pts = packet.frame as i64;
                }
                timed(Stage::Decode, || context.send_packet(Some(&d.packet)))
                    .map_err(|e| format!("decode: {e}"))?;
                emitted += drain(context, &mut d.frame, plan, out, slot_len, &mut d.buffers)?;
            }
            // flush the group's last frames
            timed(Stage::Decode, || context.send_packet(None))
                .map_err(|e| format!("decode: {e}"))?;
            emitted += drain(context, &mut d.frame, plan, out, slot_len, &mut d.buffers)?;
        }

        let wanted = plan.wanted();
        if emitted != wanted {
            return Err(format!("decoded {emitted} of {wanted} wanted frames"));
        }
        Ok(())
    })
}

/// Receive every frame the decoder has ready (into the reused `frame`); emit the wanted ones.
fn drain(
    context: &mut AVCodecContext,
    frame: &mut AVFrame,
    plan: &ClipPlan,
    out: &mut [u8],
    slot_len: usize,
    buffers: &mut (Vec<u8>, Vec<u8>),
) -> Result<usize, String> {
    let output = clip_output(plan);
    let mut emitted = 0;
    loop {
        // SAFETY: both are valid; receive_frame unreferences `frame` before filling it
        let got = timed(Stage::Decode, || unsafe {
            ffi::avcodec_receive_frame(context.as_mut_ptr(), frame.as_mut_ptr())
        });
        if got < 0 {
            return Ok(emitted);
        }
        DECODED.fetch_add(1, Ordering::Relaxed);

        let index = frame.pts as u32;
        let mut first: Option<usize> = None;
        for (slot, &wanted) in plan.frames.iter().enumerate() {
            if wanted != index {
                continue;
            }
            match first {
                None => {
                    let (pixels, scratch) = buffers;
                    let decoded = picture(frame, pixels)?;
                    let dst = &mut out[slot * slot_len..(slot + 1) * slot_len];
                    emit(&decoded, &output, dst, scratch)?;
                    first = Some(slot);
                }
                Some(source) => {
                    // a repeated frame: copy the first copy
                    out.copy_within(source * slot_len..(source + 1) * slot_len, slot * slot_len);
                }
            }
        }
        if first.is_some() {
            emitted += 1;
            EMITTED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// The colorspace ("bt709" / "bt601") the reader applies to a video: decodes its first sample
/// (a keyframe) and reads the frame's matrix coefficients.
pub fn probe_colorspace(codec: Codec, prefix: &[u8], first: &[u8]) -> Result<&'static str, String> {
    let mut context = open_decoder(codec)?;
    let mut data = [prefix, first].concat();
    if codec != Codec::Av1 && !to_annexb(&mut data[prefix.len()..]) {
        return Err("bad NAL unit lengths".into());
    }

    let mut packet = rsmpeg::avcodec::AVPacket::new();
    // SAFETY: the packet borrows `data` for the send call only
    unsafe {
        let raw = packet.as_mut_ptr();
        (*raw).data = data.as_mut_ptr();
        (*raw).size = data.len() as i32;
    }
    let decode = |e: rsmpeg::error::RsmpegError| format!("decode: {e}");
    context.send_packet(Some(&packet)).map_err(decode)?;
    context.send_packet(None).map_err(decode)?;
    let frame = context.receive_frame().map_err(decode)?;
    Ok(colorspace_name(frame.colorspace))
}
