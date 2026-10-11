"""The event table: one row per threshold crossing that survived alignment.

Events are flagged, never dropped, so a downstream user can audit or re-decide.
`accepted` is the conjunction of the rejection flags. The schema (column names
and dtypes) is the same for every input, including empty and all-noise ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCHEMA = {
    "channel": "int64",
    "sample": "int64",            # index of the trough in the filtered signal
    "time_s": "float64",          # time of that sample
    "subsample_offset": "float64",  # parabolic refinement of the trough, in samples (-0.5..0.5)
    "amplitude_uv": "float64",    # filtered trough value
    "threshold_uv": "float64",    # threshold in force at that sample
    "width_ms": "float64",        # time continuously below threshold around the trough
    "artifact": "bool",           # polarity / width / ceiling test failed
    "edge": "bool",               # waveform truncated by the start or end of the recording (kept)
    "dropped_sample": "bool",     # trough on a NaN (interpolated) sample
    "clipped": "bool",            # raw signal at the amplifier rail near the trough (kept)
    "refractory_duplicate": "bool",  # deeper trough within the refractory period, or a 10x deeper one within 3 ms
    "common_artifact": "bool",    # coincident on most channels (multichannel only)
    "accepted": "bool",
}
# flags that reject an event; `edge` and `clipped` are informative only: a truncated
# or saturated spike is still a spike, but its waveform should not be trusted
REJECTION_FLAGS = ["artifact", "dropped_sample", "refractory_duplicate", "common_artifact"]


def empty_events() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in SCHEMA.items()})


def make_events(**columns) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    df = pd.DataFrame({c: columns.get(c, np.zeros(n, dtype=t)) for c, t in SCHEMA.items()})
    df = df.astype(SCHEMA)
    df["accepted"] = ~df[REJECTION_FLAGS].any(axis=1)
    return df


def accepted(events: pd.DataFrame) -> pd.DataFrame:
    return events.loc[events["accepted"]].reset_index(drop=True)
