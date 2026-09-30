"""How much of each AV1 temporal unit later frames depend on (write time only).

A frame with ``refresh_frame_flags == 0`` stores nothing for later frames; SVT-AV1's random-access
hierarchy puts half of all frames in that top layer, usually as the shown frame at the end of a
temporal unit (one mp4 sample) after hidden reference frames. When the sample is not wanted, only
the bytes up to the last frame something depends on need decoding (``keep``; 0: none). Parses the
sequence header and each frame header up to ``refresh_frame_flags`` (AV1 spec 5.5, 5.9.2).
"""

OBU_SEQUENCE_HEADER, OBU_FRAME_HEADER, OBU_FRAME = 1, 3, 6
KEY_FRAME, INTRA_ONLY, SWITCH_FRAME = 0, 2, 3
SELECT = 2


class Bits:
    def __init__(self, data: bytes, pos: int = 0):
        self.data, self.bit = data, pos * 8

    def f(self, n: int) -> int:
        v = 0
        for _ in range(n):
            v = (v << 1) | ((self.data[self.bit >> 3] >> (7 - (self.bit & 7))) & 1)
            self.bit += 1
        return v

    def uvlc(self) -> int:
        zeros = 0
        while not self.f(1):
            zeros += 1
        return (1 << zeros) - 1 + self.f(zeros) if zeros < 32 else (1 << 32) - 1


def obus(data: bytes):
    """(type, temporal_id, spatial_id, payload start, payload end) for each OBU."""
    i = 0
    while i < len(data):
        h = data[i]
        typ, ext, has_size = (h >> 3) & 15, (h >> 2) & 1, (h >> 1) & 1
        tid = sid = 0
        j = i + 1
        if ext:
            tid, sid = data[j] >> 5, (data[j] >> 3) & 3
            j += 1
        if has_size:
            size = shift = 0
            while True:
                b = data[j]
                size |= (b & 0x7F) << shift
                j += 1
                shift += 7
                if not b & 0x80:
                    break
        else:
            size = len(data) - j
        yield typ, tid, sid, j, j + size
        i = j + size


def sequence_header(data: bytes) -> dict:
    """The fields of the sequence header (in ``data``, e.g. av1C configOBUs) the frame headers need."""
    for typ, _, _, s, _ in obus(data):
        if typ == OBU_SEQUENCE_HEADER:
            break
    else:
        raise ValueError("no sequence header")
    b = Bits(data, s)
    q = dict(decoder_model=False, equal_interval=False, frame_ids=False, id_len=0, order_bits=0,
             screen=SELECT, integer_mv=SELECT, ops=[], removal_len=0, presentation_len=0)
    b.f(3)
    b.f(1)
    q["reduced"] = b.f(1)
    if q["reduced"]:
        b.f(5)
        q["screen"], q["integer_mv"] = SELECT, SELECT
        return q
    if b.f(1):  # timing_info
        b.f(32), b.f(32)
        q["equal_interval"] = bool(b.f(1))
        if q["equal_interval"]:
            b.uvlc()
        if b.f(1):  # decoder_model_info
            q["decoder_model"] = True
            delay_len = b.f(5) + 1
            b.f(32)
            q["removal_len"], q["presentation_len"] = b.f(5) + 1, b.f(5) + 1
    display_delay = b.f(1)
    for _ in range(b.f(5) + 1):
        idc, level = b.f(12), b.f(5)
        if level > 7:
            b.f(1)
        present = False
        if q["decoder_model"]:
            present = bool(b.f(1))
            if present:
                b.f(delay_len), b.f(delay_len), b.f(1)
        if display_delay and b.f(1):
            b.f(4)
        q["ops"].append((idc, present))
    wbits, hbits = b.f(4) + 1, b.f(4) + 1
    b.f(wbits), b.f(hbits)
    q["frame_ids"] = bool(b.f(1))
    if q["frame_ids"]:
        delta, extra = b.f(4) + 2, b.f(3) + 1
        q["id_len"] = delta + extra
    b.f(1), b.f(1), b.f(1)  # 128x128 superblock, filter intra, intra edge filter
    b.f(1), b.f(1), b.f(1), b.f(1)  # interintra, masked compound, warped motion, dual filter
    order_hint = b.f(1)
    if order_hint:
        b.f(1), b.f(1)  # jnt comp, ref frame mvs
    q["screen"] = SELECT if b.f(1) else b.f(1)
    if q["screen"] > 0:
        q["integer_mv"] = SELECT if b.f(1) else b.f(1)
    if order_hint:
        q["order_bits"] = b.f(3) + 1
    return q


def refreshes(q: dict, data: bytes, s: int, tid: int, sid: int) -> bool:
    """Whether the frame header at ``data[s:]`` stores anything for later frames (or shows one)."""
    b = Bits(data, s)
    if q["reduced"] or b.f(1):  # reduced still picture (a key frame), or show_existing_frame
        return True
    frame_type, show = b.f(2), b.f(1)
    if show and q["decoder_model"] and not q["equal_interval"]:
        b.f(q["presentation_len"])
    if not show:
        b.f(1)  # showable_frame
    if frame_type == SWITCH_FRAME or (frame_type == KEY_FRAME and show):
        return True  # refresh_frame_flags = all
    error_resilient = b.f(1)
    b.f(1)  # disable_cdf_update
    screen = b.f(1) if q["screen"] == SELECT else q["screen"]
    if screen and q["integer_mv"] == SELECT:
        b.f(1)  # force_integer_mv (read even for intra frames, then overridden)
    if q["frame_ids"]:
        b.f(q["id_len"])
    b.f(1)  # frame_size_override_flag (not a switch frame, not reduced)
    b.f(q["order_bits"])
    if frame_type not in (KEY_FRAME, INTRA_ONLY) and not error_resilient:
        b.f(3)  # primary_ref_frame
    if q["decoder_model"] and b.f(1):  # buffer_removal_time_present
        for idc, present in q["ops"]:
            if present and (idc == 0 or ((idc >> tid) & 1 and (idc >> (sid + 8)) & 1)):
                b.f(q["removal_len"])
    return b.f(8) != 0  # refresh_frame_flags


def keep_bytes(seq: bytes, samples: list[bytes]) -> list[int]:
    """Per temporal unit: bytes to decode when the unit's shown frame is not wanted, i.e. up to the
    end of the last frame something depends on (a frame spans its header OBU to the next one)."""
    q = sequence_header(seq)
    out = []
    for data in samples:
        keep, needed = 0, False
        for typ, tid, sid, s, e in obus(data):
            if typ in (OBU_FRAME, OBU_FRAME_HEADER):  # a new frame; tile groups belong to the last one
                needed = refreshes(q, data, s, tid, sid)
            if needed:
                keep = e
        out.append(keep)
    return out
