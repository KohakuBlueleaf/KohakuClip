"""The GPU half of the yuv modes as one Triton kernel.

Per output pixel: antialiased bilinear taps over the stored window (none when the window already
has the output size, ``mode="yuv_resized"``), for each tap the luma sample, the chroma bilinearly
upsampled from 4:2:0 and the YUV -> RGB conversion (limited range, or full range for "-full"
colorspaces such as JPEG's; clamped), then the flips and the [-1, 1] scaling. Same math, in the
same order, as ``kohakuclip.torch.yuv_to_rgb_torch``; reads the uint8 planes once and writes the
output once.
"""

import math

import torch
import triton
import triton.language as tl

# YUV -> RGB: (R from V, G from U, G from V, B from U)
COEFFICIENTS = {
    "bt709": (1.5748, -0.187324, -0.468124, 1.8556),
    "bt601": (1.402, -0.344136, -0.714136, 1.772),
}


def color_coefficients(colorspace: str) -> list[float]:
    """The kernel's 7 conversion constants of a colorspace name ("bt709", "bt601", with
    "-full" for full range): the 4 matrix coefficients, the luma offset, the luma scale and the
    chroma scale (to [0, 1] / [-0.5, 0.5])."""
    matrix, _, value_range = colorspace.partition("-")
    coefficients = list(COEFFICIENTS[matrix])
    if value_range == "full":
        return coefficients + [0.0, 1 / 255, 1 / 255]
    return coefficients + [16.0, 1 / 219, 1 / 224]


@triton.jit
def _chroma(plane, cy, cx, ch, cw):
    """Bilinear sample of a chroma plane at (cy, cx) (align_corners=False, edges clamped)."""
    cy = tl.maximum(cy, 0.0)
    cx = tl.maximum(cx, 0.0)
    y0 = tl.minimum(cy.to(tl.int32), ch - 1)
    x0 = tl.minimum(cx.to(tl.int32), cw - 1)
    y1 = tl.minimum(y0 + 1, ch - 1)
    x1 = tl.minimum(x0 + 1, cw - 1)
    wy = cy - y0.to(tl.float32)
    wx = cx - x0.to(tl.float32)
    top = tl.load(plane + y0 * cw + x0).to(tl.float32) * (1 - wx)
    top += tl.load(plane + y0 * cw + x1).to(tl.float32) * wx
    bottom = tl.load(plane + y1 * cw + x0).to(tl.float32) * (1 - wx)
    bottom += tl.load(plane + y1 * cw + x1).to(tl.float32) * wx
    return top * (1 - wy) + bottom * wy


@triton.jit
def _yuv_to_rgb_kernel(
    src,  # uint8 [N, H * W * 3 / 2]
    dst,  # [N, 3, S, S]
    coef,  # float32 [N, 7]: matrix, luma offset, luma scale, chroma scale
    flips,  # int8 [N, 2]
    H,
    W,
    S: tl.constexpr,
    TAPS: tl.constexpr,  # taps per axis (1: no resize)
    BLOCK: tl.constexpr,
):
    frame = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = offs < S * S
    oy = offs // S
    ox = offs % S

    plane_y = src + frame.to(tl.int64) * (H * W * 3 // 2)
    plane_u = plane_y + H * W
    plane_v = plane_u + (H // 2) * (W // 2)
    r_v = tl.load(coef + frame * 7 + 0)
    g_u = tl.load(coef + frame * 7 + 1)
    g_v = tl.load(coef + frame * 7 + 2)
    b_u = tl.load(coef + frame * 7 + 3)
    y_offset = tl.load(coef + frame * 7 + 4)
    y_scale = tl.load(coef + frame * 7 + 5)
    c_scale = tl.load(coef + frame * 7 + 6)

    # antialiased bilinear taps (as torch antialias=True): support = max(scale, 1)
    scale_y = H / S
    scale_x = W / S
    support_y = tl.maximum(scale_y, 1.0)
    support_x = tl.maximum(scale_x, 1.0)
    center_y = (oy.to(tl.float32) + 0.5) * scale_y
    center_x = (ox.to(tl.float32) + 0.5) * scale_x
    first_y = tl.maximum((center_y - support_y + 0.5).to(tl.int32), 0)
    first_x = tl.maximum((center_x - support_x + 0.5).to(tl.int32), 0)
    last_y = tl.minimum((center_y + support_y + 0.5).to(tl.int32), H)
    last_x = tl.minimum((center_x + support_x + 0.5).to(tl.int32), W)

    r = tl.zeros([BLOCK], tl.float32)
    g = tl.zeros([BLOCK], tl.float32)
    b = tl.zeros([BLOCK], tl.float32)
    total_y = tl.zeros([BLOCK], tl.float32)
    total_x = tl.zeros([BLOCK], tl.float32)
    for j in tl.static_range(TAPS):
        sx = first_x + j
        wx = 1.0 - tl.abs((sx.to(tl.float32) - center_x + 0.5) / support_x)
        wx = tl.where(sx < last_x, tl.maximum(wx, 0.0), 0.0)
        total_x += wx
    for i in tl.static_range(TAPS):
        sy = first_y + i
        wy = 1.0 - tl.abs((sy.to(tl.float32) - center_y + 0.5) / support_y)
        wy = tl.where(sy < last_y, tl.maximum(wy, 0.0), 0.0)
        total_y += wy
        sy_c = tl.minimum(sy, H - 1)
        for j in tl.static_range(TAPS):
            sx = first_x + j
            wx = 1.0 - tl.abs((sx.to(tl.float32) - center_x + 0.5) / support_x)
            wx = tl.where(sx < last_x, tl.maximum(wx, 0.0), 0.0)
            sx_c = tl.minimum(sx, W - 1)

            luma = tl.load(plane_y + sy_c * W + sx_c, mask=inside, other=16).to(
                tl.float32
            )
            cy = (sy_c.to(tl.float32) + 0.5) * 0.5 - 0.5
            cx = (sx_c.to(tl.float32) + 0.5) * 0.5 - 0.5
            u = _chroma(plane_u, cy, cx, H // 2, W // 2)
            v = _chroma(plane_v, cy, cx, H // 2, W // 2)

            yy = (luma - y_offset) * y_scale
            u = (u - 128.0) * c_scale
            v = (v - 128.0) * c_scale
            w = wy * wx
            r += w * tl.minimum(tl.maximum(yy + r_v * v, 0.0), 1.0)
            g += w * tl.minimum(tl.maximum(yy + g_u * u + g_v * v, 0.0), 1.0)
            b += w * tl.minimum(tl.maximum(yy + b_u * u, 0.0), 1.0)

    norm = 1.0 / (total_y * total_x)
    hflip = tl.load(flips + frame * 2 + 0)
    vflip = tl.load(flips + frame * 2 + 1)
    ty = tl.where(vflip != 0, S - 1 - oy, oy)
    tx = tl.where(hflip != 0, S - 1 - ox, ox)
    out = dst + frame.to(tl.int64) * (3 * S * S) + ty * S + tx
    tl.store(out, (r * norm * 2 - 1).to(dst.dtype.element_ty), mask=inside)
    tl.store(out + S * S, (g * norm * 2 - 1).to(dst.dtype.element_ty), mask=inside)
    tl.store(out + 2 * S * S, (b * norm * 2 - 1).to(dst.dtype.element_ty), mask=inside)


def yuv_to_rgb_triton(
    video: torch.Tensor,
    window: tuple[int, int],
    colorspace: list[str],
    flips: torch.Tensor,
    size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """uint8 [B, T, H * W * 3 / 2] (on the GPU) -> [B, T, 3, size, size] in [-1, 1]."""
    b, t, _ = video.shape
    h, w = window
    coef = torch.tensor(
        [color_coefficients(c) for c in colorspace],
        dtype=torch.float32,
        device=video.device,
    )
    coef = coef.repeat_interleave(t, 0)
    flips = flips.to(video.device, torch.int8).repeat_interleave(t, 0)
    out = torch.empty((b, t, 3, size, size), dtype=dtype, device=video.device)

    taps = 1 if (h, w) == (size, size) else 2 * math.ceil(max(h, w) / size) + 1
    block = 256
    grid = (b * t, triton.cdiv(size * size, block))
    _yuv_to_rgb_kernel[grid](
        video, out, coef, flips, h, w, S=size, TAPS=taps, BLOCK=block
    )
    return out
