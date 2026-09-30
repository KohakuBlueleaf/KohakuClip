"""ctypes binding of the native core (native/kc.cpp). One call decodes a whole batch without the GIL."""

import ctypes as C
import glob
import os

import numpy as np

CODECS = {"h264": 0, "hevc": 1, "av1": 2}
MODES = {"rgb": 0, "yuv": 1}
STAGES = ("read", "decode", "convert", "resize")


class Request(C.Structure):
    _fields_ = [
        ("fd", C.c_int32), ("codec", C.c_int32), ("mode", C.c_int32), ("ngroups", C.c_int32),
        ("group_off", C.c_void_p), ("group_npk", C.c_void_p), ("pk_len", C.c_void_p), ("pk_idx", C.c_void_p),
        ("prefix", C.c_void_p), ("nprefix", C.c_int32), ("nwant", C.c_int32), ("want", C.c_void_p),
        ("nh", C.c_int32), ("nw", C.c_int32), ("top", C.c_int32), ("left", C.c_int32),
        ("oh", C.c_int32), ("ow", C.c_int32), ("hflip", C.c_int32), ("vflip", C.c_int32), ("out", C.c_void_p),
    ]


def _load() -> C.CDLL:
    here = os.path.dirname(os.path.abspath(__file__))
    found = glob.glob(os.path.join(here, "_kc*.so")) or glob.glob(os.path.join(here, "native", "_kc*.so"))
    if not found:
        raise ImportError("kohakuclip native core not built (pip install -e . or python -m kohakuclip.build)")
    lib = C.CDLL(found[0])
    lib.kc_decode.argtypes = [C.POINTER(Request), C.c_int, C.c_int, C.c_void_p]
    lib.kc_decode.restype = C.c_int
    lib.kc_profile.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
    if lib.kc_request_size() != C.sizeof(Request):
        raise ImportError("kohakuclip native core ABI mismatch; rebuild it")
    return lib


_lib = _load()


def decode(requests: list[Request], threads: int) -> np.ndarray:
    """Decode a batch of requests in parallel; returns the per-request status (0 = ok)."""
    arr = (Request * len(requests))(*requests)
    status = np.zeros(len(requests), np.int32)
    _lib.kc_decode(arr, len(requests), threads, status.ctypes.data)
    return status


def profile(reset: bool = False) -> dict:
    """Cumulative time per stage (seconds, summed over threads) and frames emitted / decoded since
    the last reset."""
    ns = np.zeros(len(STAGES), np.int64)
    frames = np.zeros(2, np.int64)
    _lib.kc_profile(ns.ctypes.data, frames.ctypes.data, int(reset))
    return {**{s: float(v) / 1e9 for s, v in zip(STAGES, ns)}, "frames": int(frames[0]), "decoded": int(frames[1])}
