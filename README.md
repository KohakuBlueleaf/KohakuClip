# KohakuClip

Stateless Rust+pyo3 video loader tailored for fast, memory-efficient PyTorch pipelines.

## Current Capabilities

- Decode RGB frames via ffmpeg-next with optional native resizing.
- Frame-range filtering (start / end / step) to limit memory pressure.
- Load from filesystem paths or raw in-memory bytes.
- Stateless Python builder (`KClip`) that supports lazy range/resize/crop chaining
  and materializes data via `.to_array()` / `.to_tensor()`.

## Development

### Prerequisites

- Rust toolchain + Python 3.8+
- FFmpeg 8.x development libraries (libavformat/libavcodec/libswscale/etc.). Install via your
  package manager (e.g., `apt install ffmpeg libavcodec-dev libavformat-dev libswscale-dev`).
  On macOS, use Homebrew (`brew install ffmpeg`). Windows builds have not been validated, but if
  you can build `ffmpeg-sys-next` there, this crate should compile as well.

Build the extension with maturin:

```bash
maturin develop --release
uv pip install -e .[dev]
```

Then import from Python:

```python
from kohakuclip import KClip

clip = (
    KClip("/path/to/video.mp4")
    .range(start=0, end=120, step=2)   # lazy frame selection
    .resize(width=224, height=224)     # native ffmpeg downscale
)

frames = clip.to_array()               # (num_frames, height, width, 3) uint8
tensor = clip.to_tensor(dtype=None)    # Optional PyTorch tensor materialization
```

For a CLI walkthrough, try the new `examples/range_and_resize.py` helper:

```bash
python examples/range_and_resize.py data/demo.mp4 --start 30 --end 150 --step 3 --size 256 144 --tensor
```

