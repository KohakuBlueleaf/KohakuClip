"""Builds the native core (src/kohakuclip/native/kc.cpp) as kohakuclip/_kc*.so (loaded with ctypes).

FFmpeg (libavcodec with libdav1d) is found with pkg-config, or under $KOHAKUCLIP_FFMPEG (a prefix
with include/ and lib/). $KOHAKUCLIP_MARCH sets -march (default: native; wheels: x86-64-v3).
"""

import os
import subprocess

from setuptools import Extension, setup


def ffmpeg():
    prefix = os.environ.get("KOHAKUCLIP_FFMPEG")
    if prefix:
        return [os.path.join(prefix, "include")], [os.path.join(prefix, "lib")]
    flags = subprocess.check_output(["pkg-config", "--cflags", "--libs", "libavcodec", "libavutil"], text=True).split()
    return [f[2:] for f in flags if f.startswith("-I")], [f[2:] for f in flags if f.startswith("-L")]


include, lib = ffmpeg()
setup(
    ext_modules=[
        Extension(
            "kohakuclip._kc",
            sources=["src/kohakuclip/native/kc.cpp"],
            include_dirs=include,
            library_dirs=lib,
            libraries=["avcodec", "avutil"],
            extra_compile_args=["-O3", "-std=c++17", "-fopenmp", f"-march={os.environ.get('KOHAKUCLIP_MARCH', 'native')}"],
            extra_link_args=["-fopenmp"] + [f"-Wl,-rpath,{d}" for d in lib],
            language="c++",
        )
    ]
)
