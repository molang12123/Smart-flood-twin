"""Validation for user-entered rainfall predictions, independent of any GUI."""
from __future__ import annotations

from dataclasses import dataclass
import math


UNIT_SECONDS = {"分钟": 60.0, "小时": 3600.0, "秒": 1.0}


@dataclass(frozen=True)
class PredictionRequest:
    rainfall_mm_hr: float
    rain_duration_s: float
    simulation_s: float
    interval_s: float
    output_times_s: tuple[float, ...]
    estimated_uncompressed_bytes: int

    @property
    def frame_count(self):
        return len(self.output_times_s)


def _number(value, label, *, positive=True):
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{label}必须填写数字") from None
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"{label}必须是{'正' if positive else '非负'}有限数值")
    return result


def _seconds(value, unit, label):
    if unit not in UNIT_SECONDS:
        raise ValueError(f"{label}单位必须是秒、分钟或小时")
    return _number(value, label) * UNIT_SECONDS[unit]


def _deduplicated_times(simulation_s, interval_s, rain_duration_s):
    count = int(math.floor(simulation_s / interval_s + 1e-12))
    values = [index * interval_s for index in range(count + 1)]
    if rain_duration_s <= simulation_s:
        values.append(rain_duration_s)
    values.append(simulation_s)
    values.sort()
    result = []
    tolerance = max(1e-9, simulation_s * 1e-12)
    for value in values:
        value = min(max(float(value), 0.), simulation_s)
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
        else:
            result[-1] = max(result[-1], value)
    result[0], result[-1] = 0., float(simulation_s)
    return tuple(result)


def build_prediction_request(*, rainfall_mm_hr, rain_duration_value, rain_duration_unit,
                             simulation_value, simulation_unit, interval_value, interval_unit,
                             checkpoint, model_time_s, active_cells, max_frames=500):
    rainfall = _number(rainfall_mm_hr, "平均雨势")
    rain_s = _seconds(rain_duration_value, rain_duration_unit, "降雨时长")
    simulation_s = _seconds(simulation_value, simulation_unit, "模拟总时长")
    interval_s = _seconds(interval_value, interval_unit, "输出间隔")
    domain = checkpoint.get("training_rain_domain", {})
    rain_domain = domain.get("rainfall_mm_hr")
    duration_domain = domain.get("duration_s")
    if not isinstance(rain_domain, (list, tuple)) or len(rain_domain) != 2:
        raise ValueError("模型检查点缺少雨势支持范围")
    if not isinstance(duration_domain, (list, tuple)) or len(duration_domain) != 2:
        raise ValueError("模型检查点缺少降雨时长支持范围")
    if not rain_domain[0] <= rainfall <= rain_domain[1]:
        raise ValueError(f"平均雨势超出模型范围：{rain_domain[0]:g}–{rain_domain[1]:g} mm/hr")
    if not duration_domain[0] <= rain_s <= duration_domain[1]:
        raise ValueError(f"降雨时长超出模型范围：{duration_domain[0] / 3600:g}–{duration_domain[1] / 3600:g} 小时")
    if simulation_s > model_time_s:
        raise ValueError(f"模拟总时长超过模型时域：最多 {model_time_s / 3600:g} 小时")
    times = _deduplicated_times(simulation_s, interval_s, rain_s)
    if len(times) > max_frames:
        raise ValueError(f"将产生 {len(times)} 帧，超过上限 {max_frames}；请增大输出间隔")
    if isinstance(active_cells, bool) or not isinstance(active_cells, int) or active_cells <= 0:
        raise ValueError("有效网格数量无效")
    # NPZ raw columns plus one float32 GeoTIFF band; actual compressed size varies.
    estimate = int(len(times) * active_cells * 52)
    return PredictionRequest(rainfall, rain_s, simulation_s, interval_s, times, estimate)
