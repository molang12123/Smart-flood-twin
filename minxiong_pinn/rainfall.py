from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def read_hourly_archive(path: str | Path):
    values = []
    with ZipFile(path) as archive:
        names = sorted(n for n in archive.namelist() if n.lower().endswith(".csv"))
        if not names:
            raise ValueError("rainfall archive contains no CSV files")
        for name in names:
            text = archive.read(name).decode("utf-8-sig").lstrip("\ufeff\r\n")
            rows = list(csv.reader(io.StringIO(text)))
            if not rows or rows[0][:2] != ["日/時", "1"]:
                raise ValueError(f"unexpected rainfall table layout: {name}")
            year_month = Path(name).stem.split("-")[1:3]
            year, month = map(int, year_month)
            for row in rows[1:]:
                if not row or not row[0].isdigit():
                    continue
                day = int(row[0])
                for hour, raw in enumerate(row[1:25], start=1):
                    stamp = datetime(year, month, day) + timedelta(hours=hour)
                    try: value = float(raw)
                    except ValueError: value = np.nan
                    values.append((stamp, value))
    values.sort(key=lambda item: item[0])
    return values


def extract_events(path: str | Path, *, minimum_total_mm: float = .5):
    """Convert contiguous positive hourly observations to scalar rain events.

    The conditional model accepts mean intensity and duration. Peak hourly
    intensity and total depth remain in the catalog for traceability.
    Missing observations terminate an event and are never treated as rain.
    """
    hourly = read_hourly_archive(path)
    events, current = [], []
    for stamp, value in hourly + [(None, 0.)]:
        if np.isfinite(value) and value > 0:
            current.append((stamp, value))
        elif current:
            total = float(sum(v for _, v in current))
            if total >= minimum_total_mm:
                duration = len(current)
                events.append({"event_id": f"event_{len(events):03d}",
                               "start": current[0][0].isoformat(),
                               "duration_s": float(duration * 3600),
                               "mean_intensity_mm_hr": total / duration,
                               "peak_hourly_mm_hr": float(max(v for _, v in current)),
                               "total_mm": total, "hours": duration})
            current = []
    if len(events) < 10:
        raise ValueError("too few rainfall events for train/validation/test splits")
    # Chronological split prevents future events leaking into training.
    n = len(events); train_end = max(1, int(n * .70)); validation_end = max(train_end + 1, int(n * .85))
    for i, event in enumerate(events):
        event["split"] = "train" if i < train_end else "validation" if i < validation_end else "test"
    return events


def add_design_events(events, intensities_mm_hr, durations_hr):
    """Add a deterministic parameter grid without pretending it was observed."""
    result = list(events)
    for i, intensity in enumerate(intensities_mm_hr):
        for j, hours in enumerate(durations_hr):
            if intensity <= 0 or hours <= 0:
                raise ValueError("design rainfall intensity and duration must be positive")
            # Domain edges remain training cases so the checkpoint's supported
            # rectangle is actually constrained. Interior combinations provide
            # complete held-out scenarios.
            edge = i in {0, len(intensities_mm_hr) - 1} or j in {0, len(durations_hr) - 1}
            marker = (i * 17 + j * 31) % 10
            split = "train" if edge else "validation" if marker == 3 else "test" if marker == 7 else "train"
            result.append({"event_id": f"design_r{i:02d}_d{j:02d}", "start": "",
                           "duration_s": float(hours * 3600),
                           "mean_intensity_mm_hr": float(intensity),
                           "peak_hourly_mm_hr": float(intensity),
                           "total_mm": float(intensity * hours), "hours": float(hours),
                           "split": split, "source": "synthetic_parameter_design"})
    for event in result:
        event.setdefault("source", "C0M760_2024_observation")
    return result
