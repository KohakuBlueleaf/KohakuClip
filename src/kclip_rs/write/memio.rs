//! In-memory files for FFmpeg's custom IO (input bytes, muxer output).

use std::sync::{Arc, Mutex};

use rsmpeg::avformat::AVIOContextCustom;
use rsmpeg::avutil::AVMem;
use rsmpeg::ffi;

/// A growable in-memory file shared with FFmpeg's IO callbacks.
#[derive(Default)]
pub struct MemFile {
    pub data: Vec<u8>,
    pub pos: usize,
}

impl MemFile {
    fn read(&mut self, buf: &mut [u8]) -> i32 {
        let n = buf.len().min(self.data.len().saturating_sub(self.pos));
        if n == 0 {
            return ffi::AVERROR_EOF;
        }
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        n as i32
    }

    fn write(&mut self, buf: &[u8]) -> i32 {
        let end = self.pos + buf.len();
        if end > self.data.len() {
            self.data.resize(end, 0);
        }
        self.data[self.pos..end].copy_from_slice(buf);
        self.pos = end;
        buf.len() as i32
    }

    fn seek(&mut self, offset: i64, whence: i32) -> i64 {
        let whence = whence as u32 & !ffi::AVSEEK_FORCE;
        let base = match whence {
            ffi::AVSEEK_SIZE => return self.data.len() as i64,
            0 => 0,                      // SEEK_SET
            1 => self.pos as i64,        // SEEK_CUR
            2 => self.data.len() as i64, // SEEK_END
            _ => return -1,
        };
        let target = base + offset;
        if target < 0 {
            return -1;
        }
        self.pos = target as usize;
        target
    }
}

pub fn custom_io(file: Arc<Mutex<MemFile>>, write: bool) -> AVIOContextCustom {
    let (r, w, s) = (file.clone(), file.clone(), file);
    let read = Box::new(move |_: &mut Vec<u8>, buf: &mut [u8]| r.lock().unwrap().read(buf));
    let write_packet = Box::new(move |_: &mut Vec<u8>, buf: &[u8]| w.lock().unwrap().write(buf));
    let seek = Box::new(move |_: &mut Vec<u8>, offset: i64, whence: i32| {
        s.lock().unwrap().seek(offset, whence)
    });
    AVIOContextCustom::alloc_context(
        AVMem::new(1 << 16),
        write,
        Vec::new(),
        Some(read),
        write.then_some(write_packet as _),
        Some(seek),
    )
}
