"""Pure helper functions for speech segments (no heavy libraries needed)."""


def merge_segments(segments, max_gap):
    """Join segments whose gap is <= max_gap seconds."""
    merged = []
    for s in sorted(segments, key=lambda x: x["start"]):
        if merged and s["start"] - merged[-1]["end"] <= max_gap:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append({"start": float(s["start"]), "end": float(s["end"])})
    return merged


def split_into_chunks(segments, chunk_sec, min_len):
    """Cut long segments into chunks. A leftover shorter than min_len is
    attached to the previous chunk instead of being left on its own."""
    chunks = []
    for s in segments:
        t, end = s["start"], s["end"]
        while end - t > 1e-9:
            e = min(t + chunk_sec, end)
            if end - e < min_len:
                e = end
            chunks.append({"start": t, "end": e})
            t = e
    return chunks
