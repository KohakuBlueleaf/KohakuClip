// KohakuClip native core: read + decode + resize/crop of training clips, batched and threaded.
//
// Python plans each request (which bytes, which frames, which crop) from the shard index and calls
// kc_decode once per batch. Everything per frame happens here, in parallel (OpenMP), with no Python
// and no GIL: pread of each GOP range, decode with a persistent per-thread decoder, and either
//   RGB mode: antialiased bilinear resize to (nh, nw) fused with the (oh, ow) crop + flips -> CHW, or
//   YUV mode: the stored-resolution (oh, ow) window of the Y/U/V planes (resize happens on the GPU).
// Codecs: H.264 / HEVC (libavcodec), AV1 (libdav1d through libavcodec).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <omp.h>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
}

enum Codec : int32_t { H264 = 0, HEVC = 1, AV1 = 2 };
enum Mode : int32_t { RGB = 0, YUV = 1 };

struct Request {
  int32_t fd, codec, mode;
  int32_t ngroups;
  const int64_t* group_off;   // [ngroups] byte offset of each group in the file
  const int32_t* group_npk;   // [ngroups] packets per group (contiguous bytes)
  const int32_t* pk_len;      // [sum npk] packet sizes
  const int32_t* pk_idx;      // [sum npk] display index of each packet
  const uint8_t* prefix;      // prepended to each group's first packet: AV1 sequence header, or
                              // H.264 / HEVC parameter sets (Annex B); may be null
  int32_t nprefix;
  int32_t nwant;
  const int32_t* want;        // [nwant] sorted unique display indices to output
  int32_t nh, nw;             // RGB: training resize of the stored frame (== stored size: no resize)
  int32_t top, left;          // RGB: crop origin in the resized frame; YUV: window origin (even)
  int32_t oh, ow;             // RGB: crop size; YUV: window size (even)
  int32_t hflip, vflip;       // RGB only
  uint8_t* out;               // RGB: [nwant, 3, oh, ow]; YUV: [nwant, oh*ow*3/2] (Y, U, V planes)
};

// ------------------------------------------------------------------ profiling (per stage, all threads)
enum Stage { READ, DECODE, CONVERT, RESIZE, NSTAGE };
static std::atomic<int64_t> g_ns[NSTAGE];
static std::atomic<int64_t> g_frames{0};
struct Timer {
  Stage s; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  explicit Timer(Stage st) : s(st) {}
  ~Timer() { g_ns[s] += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(); }
};

// ------------------------------------------------------------------ antialiased resize fused with crop
// Same filter as torch.nn.functional.interpolate(mode="bilinear", antialias=True) / PIL BILINEAR:
// the stored frame is conceptually resized to (nh, nw); only the (oh, ow) crop is produced, reading
// only the source window it needs. Q14 weights, uint8 intermediates, two separable passes.
struct Taps { std::vector<int> x0, n; std::vector<int16_t> w; int maxn; };

static void make_taps(int in, int out, int out0, int outn, Taps& t) {
  const double scale = (double)in / out, support = std::max(scale, 1.0), inv = 1.0 / support;
  t.maxn = (int)std::ceil(support) * 2 + 1;
  t.x0.assign(outn, 0); t.n.assign(outn, 0); t.w.assign((size_t)outn * t.maxn, 0);
  std::vector<double> ww(t.maxn);
  for (int i = 0; i < outn; ++i) {
    const double c = (out0 + i + 0.5) * scale;
    const int lo = std::max((int)(c - support + 0.5), 0), hi = std::min((int)(c + support + 0.5), in);
    const int n = std::min(hi - lo, t.maxn);
    double tot = 0;
    for (int j = 0; j < n; ++j) { ww[j] = std::max(0.0, 1.0 - std::fabs((j + lo - c + 0.5) * inv)); tot += ww[j]; }
    t.x0[i] = lo; t.n[i] = n;
    for (int j = 0; j < n; ++j) t.w[(size_t)i * t.maxn + j] = (int16_t)std::lround(ww[j] / tot * 16384);
  }
}

// dst[x] = sum_j wt[j] * rows[j][x] (Q14, rounded, clamped); contiguous -> vectorized by the compiler
static inline void weighted_rows(const uint8_t* const* rows, const int16_t* wt, int n, int len, uint8_t* dst) {
  thread_local std::vector<int32_t> acc;
  acc.assign(len, 1 << 13);
  int32_t* a = acc.data();
  for (int j = 0; j < n; ++j) {
    const uint8_t* r = rows[j]; const int32_t w = wt[j];
    for (int x = 0; x < len; ++x) a[x] += w * r[x];
  }
  for (int x = 0; x < len; ++x) dst[x] = (uint8_t)std::clamp(a[x] >> 14, 0, 255);
}

// source window [x0, x1) x [y0, y1) of an (h, w) stored frame that the crop needs
static void source_window(const Request& r, int h, int w, int& x0, int& x1, int& y0, int& y1) {
  if (r.nh == h && r.nw == w) { x0 = r.left; x1 = r.left + r.ow; y0 = r.top; y1 = r.top + r.oh; return; }
  const double sx = (double)w / r.nw, sy = (double)h / r.nh;
  x0 = std::max(0, (int)std::floor(r.left * sx - sx - 1)); x1 = std::min(w, (int)std::ceil((r.left + r.ow) * sx + sx + 1));
  y0 = std::max(0, (int)std::floor(r.top * sy - sy - 1));  y1 = std::min(h, (int)std::ceil((r.top + r.oh) * sy + sy + 1));
}

// rgb: HWC window [x0, ..) x [y0, ..) of an (h, w) frame -> CHW crop with flips
static void resize_crop(const uint8_t* rgb, int stride, int x0, int y0, int h, int w, const Request& r, uint8_t* out) {
  Timer t(RESIZE);
  const int oh = r.oh, ow = r.ow, plane = oh * ow;
  if (r.nh == h && r.nw == w) {  // no resize: plain crop
    for (int y = 0; y < oh; ++y) {
      const uint8_t* row = rgb + (size_t)(r.top + y - y0) * stride + (size_t)(r.left - x0) * 3;
      uint8_t* R = out + (size_t)(r.vflip ? oh - 1 - y : y) * ow; uint8_t* G = R + plane; uint8_t* B = G + plane;
      for (int x = 0; x < ow; ++x) { const int o = r.hflip ? ow - 1 - x : x; R[o] = row[3 * x]; G[o] = row[3 * x + 1]; B[o] = row[3 * x + 2]; }
    }
    return;
  }
  thread_local Taps tx, ty;
  thread_local std::vector<uint8_t> vert, planar, horiz;
  make_taps(w, r.nw, r.left, ow, tx); make_taps(h, r.nh, r.top, oh, ty);
  const int xs = tx.x0[0], win = tx.x0[ow - 1] + tx.n[ow - 1] - xs;
  const uint8_t* rows[64];
  // 1) vertical pass on interleaved rows: [oh][win * 3]
  vert.resize((size_t)oh * win * 3);
  for (int y = 0; y < oh; ++y) {
    for (int j = 0; j < ty.n[y]; ++j) rows[j] = rgb + (size_t)(ty.x0[y] + j - y0) * stride + (size_t)(xs - x0) * 3;
    weighted_rows(rows, ty.w.data() + (size_t)y * ty.maxn, ty.n[y], win * 3, vert.data() + (size_t)y * win * 3);
  }
  // 2) transpose to planar column-major [3][win][oh] (32 x 32 blocks)
  planar.resize((size_t)3 * win * oh);
  for (int yb = 0; yb < oh; yb += 32)
    for (int xb = 0; xb < win; xb += 32)
      for (int c = 0; c < 3; ++c)
        for (int x = xb; x < std::min(xb + 32, win); ++x) {
          uint8_t* d = planar.data() + ((size_t)c * win + x) * oh;
          for (int y = yb; y < std::min(yb + 32, oh); ++y) d[y] = vert[(size_t)y * win * 3 + 3 * x + c];
        }
  // 3) horizontal pass, done as a contiguous pass over columns: [3][ow][oh]
  horiz.resize((size_t)3 * ow * oh);
  for (int c = 0; c < 3; ++c)
    for (int x = 0; x < ow; ++x) {
      for (int j = 0; j < tx.n[x]; ++j) rows[j] = planar.data() + ((size_t)c * win + tx.x0[x] + j - xs) * oh;
      weighted_rows(rows, tx.w.data() + (size_t)x * tx.maxn, tx.n[x], oh, horiz.data() + ((size_t)c * ow + x) * oh);
    }
  // 4) transpose back to CHW with flips (32 x 32 blocks)
  for (int c = 0; c < 3; ++c) {
    const uint8_t* src = horiz.data() + (size_t)c * ow * oh; uint8_t* dst = out + (size_t)c * plane;
    for (int yb = 0; yb < oh; yb += 32)
      for (int xb = 0; xb < ow; xb += 32)
        for (int y = yb; y < std::min(yb + 32, oh); ++y) {
          uint8_t* d = dst + (size_t)(r.vflip ? oh - 1 - y : y) * ow;
          for (int x = xb; x < std::min(xb + 32, ow); ++x) d[r.hflip ? ow - 1 - x : x] = src[(size_t)x * oh + y];
        }
  }
}

// YUV window [x0, x1) x [y0, y1) -> interleaved RGB; 4:2:0 with centered bilinear chroma, or 4:4:4
static void yuv_to_rgb(const AVFrame* f, int x0, int x1, int y0, int y1, uint8_t* rgb) {
  Timer t(CONVERT);
  const bool full = f->color_range == AVCOL_RANGE_JPEG, bt709 = f->colorspace == AVCOL_SPC_BT709;
  const bool sub = f->format != AV_PIX_FMT_YUV444P && f->format != AV_PIX_FMT_YUVJ444P;
  const uint8_t *Y = f->data[0], *U = f->data[1], *V = f->data[2];
  const int ys = f->linesize[0], cs = f->linesize[1], w = f->width, h = f->height, rw = x1 - x0;
  const int ky = full ? 16384 : 19077, yoff = full ? 0 : 16;
  const double s = full ? 1.0 : 255.0 / 224.0;
  const double cr = bt709 ? 1.5748 : 1.402, cgu = bt709 ? -0.187324 : -0.344136, cgv = bt709 ? -0.468124 : -0.714136, cb = bt709 ? 1.8556 : 1.772;
  const int kr = (int)(cr * s * 16384 + .5), kgu = (int)(cgu * s * 16384 - .5), kgv = (int)(cgv * s * 16384 - .5), kb = (int)(cb * s * 16384 + .5);
  thread_local std::vector<int32_t> uu, vv, pu, pv;
  uu.resize(rw); vv.resize(rw);
  const int cw = (w + 1) / 2, chh = (h + 1) / 2;
  const int c_lo = std::max(0, (x0 >> 1) - 1), c_hi = std::min(cw - 1, (x1 >> 1) + 1), nc = c_hi - c_lo + 1;
  pu.resize(nc + 2); pv.resize(nc + 2);
  for (int yy = y0; yy < y1; ++yy) {
    if (!sub) {
      for (int x = 0; x < rw; ++x) { uu[x] = (U[(size_t)yy * cs + x0 + x] - 128) * 16; vv[x] = (V[(size_t)yy * cs + x0 + x] - 128) * 16; }
    } else {  // vertical 3:1 / 1:3 taps (centered chroma siting), then horizontal
      const int cy2 = 2 * yy - 1, c0 = cy2 < 0 ? 0 : cy2 >> 2, fy = cy2 < 0 ? 0 : cy2 & 3, c1 = std::min(c0 + 1, chh - 1);
      const uint8_t *U0 = U + (size_t)c0 * cs + c_lo, *U1 = U + (size_t)c1 * cs + c_lo, *V0 = V + (size_t)c0 * cs + c_lo, *V1 = V + (size_t)c1 * cs + c_lo;
      int32_t *qu = pu.data() + 1, *qv = pv.data() + 1;
      for (int k = 0; k < nc; ++k) { qu[k] = U0[k] * (4 - fy) + U1[k] * fy; qv[k] = V0[k] * (4 - fy) + V1[k] * fy; }
      qu[-1] = qu[0]; qv[-1] = qv[0]; qu[nc] = qu[nc - 1]; qv[nc] = qv[nc - 1];
      for (int x = 0; x < rw; ++x) {
        const int xx = x0 + x, k = (xx >> 1) - c_lo, odd = xx & 1;
        const int a = odd ? k : k - 1, b = odd ? k + 1 : k, wa = odd ? 3 : 1, wb = odd ? 1 : 3;
        uu[x] = qu[a] * wa + qu[b] * wb - 128 * 16; vv[x] = qv[a] * wa + qv[b] * wb - 128 * 16;
      }
      if (x0 == 0) { uu[0] = qu[0] * 4 - 128 * 16; vv[0] = qv[0] * 4 - 128 * 16; }
    }
    const uint8_t* Yr = Y + (size_t)yy * ys + x0; uint8_t* o = rgb + (size_t)(yy - y0) * rw * 3;
    for (int x = 0; x < rw; ++x) {
      const int yv = (Yr[x] - yoff) * ky * 16;
      o[3 * x] = (uint8_t)std::clamp((yv + kr * vv[x] + 131072) >> 18, 0, 255);
      o[3 * x + 1] = (uint8_t)std::clamp((yv + kgu * uu[x] + kgv * vv[x] + 131072) >> 18, 0, 255);
      o[3 * x + 2] = (uint8_t)std::clamp((yv + kb * uu[x] + 131072) >> 18, 0, 255);
    }
  }
}

// YUV mode: copy the (oh, ow) window at (top, left) of the Y, U, V planes (4:2:0) into out
static void copy_window(const AVFrame* f, const Request& r, uint8_t* out) {
  Timer t(CONVERT);
  for (int y = 0; y < r.oh; ++y) memcpy(out + (size_t)y * r.ow, f->data[0] + (size_t)(r.top + y) * f->linesize[0] + r.left, r.ow);
  uint8_t* o = out + (size_t)r.oh * r.ow;
  const int ch = r.oh / 2, cw = r.ow / 2;
  for (int p = 1; p <= 2; ++p, o += (size_t)ch * cw)
    for (int y = 0; y < ch; ++y) memcpy(o + (size_t)y * cw, f->data[p] + (size_t)(r.top / 2 + y) * f->linesize[p] + r.left / 2, cw);
}

// ------------------------------------------------------------------ per-thread decoders
struct Decoders {
  AVCodecContext* video[3] = {};
  AVPacket* pkt = av_packet_alloc();
  AVFrame* frame = av_frame_alloc();
  std::vector<uint8_t> bytes, rgb;
  ~Decoders() {
    for (auto& c : video) if (c) avcodec_free_context(&c);
    av_packet_free(&pkt); av_frame_free(&frame);
  }
  AVCodecContext* get(int codec) {
    if (video[codec]) return video[codec];
    const AVCodec* dec = codec == AV1 ? avcodec_find_decoder_by_name("libdav1d")
                                      : avcodec_find_decoder(codec == H264 ? AV_CODEC_ID_H264 : AV_CODEC_ID_HEVC);
    AVCodecContext* ctx = avcodec_alloc_context3(dec);
    ctx->thread_count = 1;  // one decode thread per worker: parallelism comes from the batch
    if (codec == AV1) { av_opt_set_int(ctx->priv_data, "max_frame_delay", 1, 0); av_opt_set_int(ctx->priv_data, "tilethreads", 1, 0); }
    if (avcodec_open2(ctx, dec, nullptr) < 0) { avcodec_free_context(&ctx); return nullptr; }
    return video[codec] = ctx;
  }
};
static thread_local Decoders D;

static bool read_range(int fd, int64_t off, size_t n, std::vector<uint8_t>& buf, size_t at) {
  Timer t(READ);
  buf.resize(at + n);
  for (size_t done = 0; done < n;) {
    const ssize_t k = pread(fd, buf.data() + at + done, n - done, off + (int64_t)done);
    if (k <= 0) return false;
    done += (size_t)k;
  }
  return true;
}

// write the wanted frame `f` (display index idx) into its output slot
static void emit(const Request& r, const AVFrame* f, int slot) {
  ++g_frames;
  if (r.mode == YUV) { copy_window(f, r, r.out + (size_t)slot * r.oh * r.ow * 3 / 2); return; }
  int x0, x1, y0, y1;
  source_window(r, f->height, f->width, x0, x1, y0, y1);
  x0 &= ~1;
  D.rgb.resize((size_t)(x1 - x0) * (y1 - y0) * 3);
  yuv_to_rgb(f, x0, x1, y0, y1, D.rgb.data());
  resize_crop(D.rgb.data(), (x1 - x0) * 3, x0, y0, f->height, f->width, r, r.out + (size_t)slot * 3 * r.oh * r.ow);
}

// mp4 H.264 / HEVC samples are length-prefixed NAL units (4-byte lengths); the decoder takes Annex B:
// swap each length for a start code, in place
static bool to_annexb(uint8_t* p, size_t n) {
  for (size_t i = 0; i + 4 <= n;) {
    const size_t len = (size_t)p[i] << 24 | (size_t)p[i + 1] << 16 | (size_t)p[i + 2] << 8 | p[i + 3];
    p[i] = p[i + 1] = p[i + 2] = 0; p[i + 3] = 1;
    i += 4 + len;
    if (i > n) return false;
  }
  return true;
}

// decode each group from its keyframe to its last wanted frame; emit wanted frames
static int decode_video(const Request& r) {
  AVCodecContext* ctx = D.get(r.codec);
  if (!ctx) return -1;
  int pk = 0, emitted = 0;
  for (int g = 0; g < r.ngroups; ++g) {
    const int npk = r.group_npk[g];
    size_t total = r.nprefix;
    for (int i = 0; i < npk; ++i) total += r.pk_len[pk + i];
    D.bytes.resize(total);
    if (r.nprefix) memcpy(D.bytes.data(), r.prefix, r.nprefix);
    if (!read_range(r.fd, r.group_off[g], total - r.nprefix, D.bytes, r.nprefix)) return -2;
    avcodec_flush_buffers(ctx);
    size_t at = 0;
    for (int i = 0; i <= npk; ++i) {
      int sent;
      {
        Timer t(DECODE);
        if (i < npk) {
          const size_t len = r.pk_len[pk + i] + (i == 0 ? r.nprefix : 0);
          if (r.codec != AV1 && !to_annexb(D.bytes.data() + at + (i == 0 ? r.nprefix : 0), r.pk_len[pk + i])) return -3;
          av_packet_unref(D.pkt);
          D.pkt->data = D.bytes.data() + at; D.pkt->size = (int)len; D.pkt->pts = r.pk_idx[pk + i];
          at += len;
          sent = avcodec_send_packet(ctx, D.pkt);
        } else {
          sent = avcodec_send_packet(ctx, nullptr);  // drain the group
        }
      }
      if (sent < 0) return -3;
      for (;;) {
        int got;
        { Timer t(DECODE); got = avcodec_receive_frame(ctx, D.frame); }
        if (got < 0) break;
        const int idx = (int)D.frame->pts;
        const int* w = std::lower_bound(r.want, r.want + r.nwant, idx);
        if (w != r.want + r.nwant && *w == idx) { emit(r, D.frame, (int)(w - r.want)); ++emitted; }
        av_frame_unref(D.frame);
      }
    }
    pk += npk;
  }
  return emitted == r.nwant ? 0 : -4;
}

extern "C" {
// Decode a batch; per-request status in `status` (0 = ok). Returns the number of failed requests.
int kc_decode(const Request* reqs, int n, int threads, int32_t* status) {
  int bad = 0;
#pragma omp parallel for num_threads(std::max(threads, 1)) schedule(dynamic, 1) reduction(+ : bad)
  for (int i = 0; i < n; ++i) {
    status[i] = decode_video(reqs[i]);
    bad += status[i] != 0;
  }
  return bad;
}
// Cumulative per-stage time (ns: read, decode, convert, resize) and frames emitted; reset if asked.
void kc_profile(int64_t* ns, int64_t* frames, int reset) {
  for (int s = 0; s < NSTAGE; ++s) ns[s] = reset ? g_ns[s].exchange(0) : g_ns[s].load();
  *frames = reset ? g_frames.exchange(0) : g_frames.load();
}
int kc_request_size() { return (int)sizeof(Request); }
}
