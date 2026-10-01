#!/bin/bash
# The wheels' build environment, run inside the manylinux image (maturin-action
# before-script): an LGPL FFmpeg 8.1 from conda-forge (libdav1d, SVT-AV1, aom, openh264;
# no x264 / x265) plus libclang 18 and clang 18's headers for the bindings (newer clang
# generates broken FFmpeg bindings with this bindgen), all under /opt/ffmpeg.
# release.yml points the build at it with fixed paths (FFMPEG_*, LIBCLANG_PATH, the clang
# resource headers in BINDGEN_EXTRA_CLANG_ARGS,
# LD_LIBRARY_PATH for auditwheel, which then bundles the libraries into the wheel).
set -euo pipefail
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local bin/micromamba
micromamba create -y -q -p /opt/ffmpeg -c conda-forge "ffmpeg=8.1.*=lgpl*" "libclang=18.*" "clang=18.*"
