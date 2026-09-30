// KohakuClip native core: read + decode + resize/crop of training clips, batched and threaded.
//
// Python plans each request (which bytes, which frames, which crop) from the shard index and calls
// kc_decode once per batch. Everything per frame happens here, in parallel (OpenMP), with no Python
// and no GIL: pread of each GOP range, decode with a persistent per-thread decoder, and either
//   RGB mode: antialiased bilinear resize of each YUV plane to (nh, nw) fused with the (oh, ow)
//             crop, then conversion of the output pixels + flips -> CHW, or
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
static std::atomic<int64_t> g_frames{0}, g_decoded{0};  // frames emitted / decoded
struct Timer {
  Stage s; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  explicit Timer(Stage st) : s(st) {}
  ~Timer() { g_ns[s] += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(); }
};

// ------------------------------------------------------------------ antialiased resize fused with crop
// Same filter as torch.nn.functional.interpolate(mode="bilinear", antialias=True) / PIL BILINEAR,
// applied per YUV plane before color conversion (so only output pixels are converted). Q14 weights,
// uint8 intermediates, two separable passes.
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

// dst[x][y] = src[y][x] for an (h, w) uint8 matrix, 32 x 32 blocks
static void transpose(const uint8_t* src, int h, int w, uint8_t* dst) {
  for (int yb = 0; yb < h; yb += 32)
    for (int xb = 0; xb < w; xb += 32)
      for (int x = xb; x < std::min(xb + 32, w); ++x)
        for (int y = yb; y < std::min(yb + 32, h); ++y) dst[(size_t)x * h + y] = src[(size_t)y * w + x];
}

// One plane, conceptually resized to (nh, nw) (antialiased), cropped to (oh, ow) at (top, left):
// vertical pass over the needed rows, transpose, horizontal pass as a vertical pass, transpose.
// Only the source window the crop needs is read. Chroma planes use the same call: their taps map
// the smaller plane straight onto the output grid (centered siting falls out of the geometry).
static void resize_plane(const uint8_t* src, int stride, int ph, int pw, const Request& r, uint8_t* dst) {
  thread_local Taps tx, ty;
  thread_local std::vector<uint8_t> vert, cols, horiz;
  const int oh = r.oh, ow = r.ow;
  make_taps(pw, r.nw, r.left, ow, tx); make_taps(ph, r.nh, r.top, oh, ty);
  const int xs = tx.x0[0], win = tx.x0[ow - 1] + tx.n[ow - 1] - xs;
  const uint8_t* rows[64];
  vert.resize((size_t)oh * win);
  for (int y = 0; y < oh; ++y) {
    for (int j = 0; j < ty.n[y]; ++j) rows[j] = src + (size_t)(ty.x0[y] + j) * stride + xs;
    weighted_rows(rows, ty.w.data() + (size_t)y * ty.maxn, ty.n[y], win, vert.data() + (size_t)y * win);
  }
  cols.resize((size_t)win * oh);
  transpose(vert.data(), oh, win, cols.data());
  horiz.resize((size_t)ow * oh);
  for (int x = 0; x < ow; ++x) {
    for (int j = 0; j < tx.n[x]; ++j) rows[j] = cols.data() + (size_t)(tx.x0[x] + j - xs) * oh;
    weighted_rows(rows, tx.w.data() + (size_t)x * tx.maxn, tx.n[x], oh, horiz.data() + (size_t)x * oh);
  }
  transpose(horiz.data(), ow, oh, dst);
}

// Y, U, V at output resolution -> RGB CHW with flips (BT.601 / BT.709, limited or full range)
static void to_rgb(const uint8_t* Y, const uint8_t* U, const uint8_t* V, const AVFrame* f, const Request& r, uint8_t* out) {
  const bool full = f->color_range == AVCOL_RANGE_JPEG, bt709 = f->colorspace == AVCOL_SPC_BT709;
  const int ky = full ? 16384 : 19077, yoff = full ? 0 : 16;
  const double s = full ? 1.0 : 255.0 / 224.0;
  const double cr = bt709 ? 1.5748 : 1.402, cgu = bt709 ? -0.187324 : -0.344136, cgv = bt709 ? -0.468124 : -0.714136, cb = bt709 ? 1.8556 : 1.772;
  const int kr = (int)(cr * s * 16384 + .5), kgu = (int)(cgu * s * 16384 - .5), kgv = (int)(cgv * s * 16384 - .5), kb = (int)(cb * s * 16384 + .5);
  const int oh = r.oh, ow = r.ow;
  const size_t plane = (size_t)oh * ow;
  for (int y = 0; y < oh; ++y) {
    const size_t i = (size_t)y * ow;
    uint8_t* R = out + (size_t)(r.vflip ? oh - 1 - y : y) * ow;
    uint8_t *G = R + plane, *B = G + plane;
    for (int x = 0; x < ow; ++x) {
      const int yv = (Y[i + x] - yoff) * ky * 16, u = (U[i + x] - 128) * 16, v = (V[i + x] - 128) * 16;
      const int o = r.hflip ? ow - 1 - x : x;
      R[o] = (uint8_t)std::clamp((yv + kr * v + 131072) >> 18, 0, 255);
      G[o] = (uint8_t)std::clamp((yv + kgu * u + kgv * v + 131072) >> 18, 0, 255);
      B[o] = (uint8_t)std::clamp((yv + kb * u + 131072) >> 18, 0, 255);
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
  std::vector<uint8_t> bytes, planes;
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

// write the wanted frame `f` into its output slot
static void emit(const Request& r, const AVFrame* f, int slot) {
  ++g_frames;
  if (r.mode == YUV) { copy_window(f, r, r.out + (size_t)slot * r.oh * r.ow * 3 / 2); return; }
  const size_t plane = (size_t)r.oh * r.ow;
  D.planes.resize(3 * plane);
  const bool sub = f->format != AV_PIX_FMT_YUV444P && f->format != AV_PIX_FMT_YUVJ444P;
  const int ch = sub ? (f->height + 1) / 2 : f->height, cw = sub ? (f->width + 1) / 2 : f->width;
  {
    Timer t(RESIZE);
    resize_plane(f->data[0], f->linesize[0], f->height, f->width, r, D.planes.data());
    resize_plane(f->data[1], f->linesize[1], ch, cw, r, D.planes.data() + plane);
    resize_plane(f->data[2], f->linesize[2], ch, cw, r, D.planes.data() + 2 * plane);
  }
  Timer t(CONVERT);
  to_rgb(D.planes.data(), D.planes.data() + plane, D.planes.data() + 2 * plane, f, r, r.out + (size_t)slot * 3 * plane);
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
        ++g_decoded;
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
// Cumulative per-stage time (ns: read, decode, convert, resize) and frames [emitted, decoded];
// reset if asked.
void kc_profile(int64_t* ns, int64_t* frames, int reset) {
  for (int s = 0; s < NSTAGE; ++s) ns[s] = reset ? g_ns[s].exchange(0) : g_ns[s].load();
  frames[0] = reset ? g_frames.exchange(0) : g_frames.load();
  frames[1] = reset ? g_decoded.exchange(0) : g_decoded.load();
}
int kc_request_size() { return (int)sizeof(Request); }
}
