"""Portable, human-readable training reports, without changing source evidence.

The HTML embeds its plots and CSS and can be opened without a web server. It
summarizes existing evidence; generating a report never certifies a model and
never starts training, inference, a browser, or a GIS application.
"""
from __future__ import annotations

import base64
import csv
from html import escape
import json
import math
from pathlib import Path
from urllib.parse import quote
import uuid


_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_MAX_TOTAL_IMAGE_BYTES = 16 * 1024 * 1024
_MAX_IMAGES = 24
_MAX_ARTIFACTS = 200
_MAX_CSV_ROWS = 2_000_000
_MAX_JSON_BYTES = 8 * 1024 * 1024
_REPORT_NAMES = {"training_results_report.html", "training_results_report.md"}
_LOSSES = (
    ("total_loss", "Total Loss"), ("pde_loss", "PDE Loss"),
    ("ic_loss", "IC Loss"), ("bc_loss", "BC Loss"),
    ("data_loss", "Data Loss"), ("conservation_loss", "积分连续性 Loss"),
)


def _finite(value):
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value):
    if value is None or value == "":
        return "未提供 / 未评估"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return "异常：非有限数值 NaN/Inf"
        return f"{value:.7g}" if isinstance(value, float) else str(value)
    if isinstance(value, (list, tuple)):
        return " / ".join(_fmt(item) for item in value)
    if isinstance(value, dict):
        return _json_text(value)
    text = str(value)
    if text.strip().lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "-infinity"}:
        return "异常：非有限数值 NaN/Inf"
    return text


def _json_text(value):
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str)
    except (ValueError, TypeError):
        return "包含无法规范序列化的数据（可能为非有限值），仅供诊断：\n" + str(value)


def _safe_file(root, path):
    """Do not follow evidence/image symlinks outside this particular run."""
    try:
        resolved = path.resolve()
        return resolved.is_relative_to(root) and resolved.is_file()
    except (OSError, RuntimeError):
        return False


def _read_json(root, relative, warnings):
    path = root / relative
    if not path.exists():
        return {}
    if not _safe_file(root, path):
        warnings.append(f"忽略运行目录外的证据文件：{relative}")
        return {}
    try:
        if path.stat().st_size > _MAX_JSON_BYTES:
            raise ValueError("超过报告的JSON大小限制")
        def reject_constant(token):
            raise ValueError(f"不允许非有限JSON数值：{token}")
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
        if not isinstance(data, dict):
            raise ValueError("报告证据必须是JSON对象")
        return data
    except (OSError, UnicodeError, ValueError) as exc:
        warnings.append(f"无法读取 {relative}：{exc}")
        return {}


def _csv_rows(root, relative, warnings):
    path = root / relative
    if not path.exists():
        return
    if not _safe_file(root, path):
        warnings.append(f"忽略运行目录外的CSV：{relative}")
        return
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            for index, row in enumerate(reader):
                if index >= _MAX_CSV_ROWS:
                    warnings.append(f"{relative}超过报告扫描上限，仅汇总前{_MAX_CSV_ROWS}行")
                    break
                yield row
    except (OSError, UnicodeError, csv.Error) as exc:
        warnings.append(f"无法完整读取 {relative}：{exc}")


def _history(root, warnings):
    phases, count, nonfinite = {}, 0, 0
    numeric = [key for key, _ in _LOSSES] + ["learning_rate", "elapsed_s", "gradient_norm_before_clip", "gradient_norm_after_clip"]
    for row in _csv_rows(root, "training/training_history.csv", warnings):
        count += 1
        phase = str(row.get("phase") or "unknown")
        entry = phases.setdefault(phase, {"first": row, "last": row, "evaluations": 0})
        entry["last"] = row
        entry["evaluations"] += 1
        flagged = str(row.get("nonfinite", "")).lower() in {"true", "1", "yes"}
        for name in numeric:
            value = row.get(name)
            if value not in (None, "") and _finite(value) is None:
                flagged = True
        nonfinite += int(flagged)
    return {"phases": phases, "row_count": count, "nonfinite_rows": nonfinite}


def _csv_audit(root, relative, warnings):
    count, nonfinite = 0, 0
    for row in _csv_rows(root, relative, warnings):
        count += 1
        if any(str(value).strip().lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}
               for value in row.values()):
            nonfinite += 1
    return count, nonfinite


def _status(value, kind):
    if value is None or value == "":
        return "未评估 / 未提供证据"
    if kind == "completed":
        return "已完成配置中的训练步骤" if value is True else "未正常完成全部训练步骤"
    if kind == "converged":
        return "报告记录为收敛（仍需查看各项验收）" if value is True else "未确认收敛"
    if kind == "physics":
        return "报告记录物理检查通过" if value is True else "物理检查未通过"
    translations = {
        "passed": "通过（以对应数据、范围和门槛为准）",
        "failed": "未通过",
        "not_completed": "尚未完成外部验证",
        "not_evaluated": "未评估",
        "comparison_only_not_accuracy_approval": "已做参考对比，未获准确度认证",
        "comparison_completed": "已完成数值对比，不能单独视为准确度认证",
        "physics_audited_not_accuracy_validated": "已审计留出雨情物理残差，尚未验证预测准确度",
        "research_unvalidated": "研究模式，未验证",
        "complete": "流程执行完成；不代表模型验收通过",
        "completed": "流程执行完成；不代表模型验收通过",
        "configured_steps_completed": "完成配置中的迭代步骤",
        "time_budget_exhausted": "达到时间预算，训练停止",
        "diagnostics_failed": "训练后诊断失败",
        "pending": "尚未完成",
        "incomplete": "执行未完成",
        "measured_not_certified": "已记录水量余差，尚未认证",
        "requires_time_series_export": "需要时序导出才能完成水量审计",
    }
    return translations.get(str(value), _fmt(value))


def _table(headers, rows):
    if not rows:
        return '<p class="muted">没有可用记录，未评估。</p>'
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(_fmt(v))}</td>" for v in row) + "</tr>" for row in rows)
    return '<div class="table-scroll"><table><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table></div>'


def _definition(rows):
    return _table(["项目", "记录"], rows)


def _artifacts(root, warnings):
    result = []
    try:
        for path in root.rglob("*"):
            if len(result) >= _MAX_ARTIFACTS:
                warnings.append(f"原始文件列表仅展示前{_MAX_ARTIFACTS}项")
                break
            if path.name in _REPORT_NAMES or not _safe_file(root, path):
                continue
            relative = path.relative_to(root).as_posix()
            result.append((relative, path.stat().st_size))
    except OSError as exc:
        warnings.append(f"部分文件无法列出：{exc}")
    return sorted(result)


def _images(root, warnings):
    images, total = [], 0
    for folder in (root / "training", root / "prediction"):
        if not folder.is_dir() or not folder.resolve().is_relative_to(root):
            continue
        for path in sorted(folder.glob("*.png")):
            relative = path.relative_to(root).as_posix()
            if not _safe_file(root, path):
                warnings.append(f"忽略运行目录外图像：{relative}")
                continue
            size = path.stat().st_size
            if size > _MAX_IMAGE_BYTES or total + size > _MAX_TOTAL_IMAGE_BYTES or len(images) >= _MAX_IMAGES:
                warnings.append(f"图像超过报告内嵌大小/数量限制，保留原文件链接：{relative}")
                continue
            try:
                data = path.read_bytes()
                if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                    warnings.append(f"忽略非PNG内容的图像：{relative}")
                    continue
                images.append((relative, base64.b64encode(data).decode("ascii")))
                total += size
            except OSError as exc:
                warnings.append(f"无法读取图像 {relative}：{exc}")
    return images


def _atomic_text(path, content):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def generate_training_report(run_dir, *, context=None):
    """Write ``training_results_report.html`` and ``.md`` inside this run.

    ``context`` is a JSON-like dictionary from the caller (manual rainfall,
    fixed-terrain identity, run status/error). Missing and failed training or
    inference evidence remains explicitly unassessed. Source artifacts are read
    only; the function does not regenerate their reports, plots or checkpoints.
    """
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if context is not None and not isinstance(context, dict):
        raise ValueError("report context must be a dictionary")
    context = dict(context or {})
    warnings = []
    supplied_warnings = context.get("warnings", [])
    if isinstance(supplied_warnings, (list, tuple)):
        warnings.extend(str(w) for w in supplied_warnings)
    elif supplied_warnings:
        warnings.append(str(supplied_warnings))
    training = _read_json(root, "training/training_report.json", warnings)
    historical_training = False
    historical_report = context.get("loaded_model_training_report")
    if not training and not (root / "training/training_report.json").exists() and isinstance(historical_report, dict):
        training = dict(historical_report)
        historical_training = True
    metrics = _read_json(root, "training/validation_metrics.json", warnings)
    simulation = _read_json(root, "prediction/simulation_summary.json", warnings)
    assessment = _read_json(root, "assessment.json", warnings)
    history = _history(root, warnings)
    audit_count, audit_nonfinite = _csv_audit(root, "training/physics_diagnostics.csv", warnings)
    if history["nonfinite_rows"] or audit_nonfinite:
        warnings.append("源证据中检测到非有限值/无效数值；不能把该运行当作健康模型。")
    config = training.get("config") if isinstance(training.get("config"), dict) else {}
    fixed = context.get("fixed_terrain") if isinstance(context.get("fixed_terrain"), dict) else {}
    terrain = context.get("terrain") if isinstance(context.get("terrain"), dict) else {}
    synthetic = context.get("synthetic", fixed.get("synthetic", terrain.get("synthetic")))
    fingerprint = context.get("terrain_fingerprint", fixed.get("fingerprint", terrain.get("fingerprint", training.get("terrain_fingerprint"))))
    grid_id = context.get("grid_id", fixed.get("grid_id", terrain.get("grid_id", simulation.get("grid_id"))))
    status = context.get("status")
    error = context.get("error", None if historical_training else training.get("error"))
    external = training.get("reference_validation_status", metrics.get("reference_validation_status"))
    heldout = training.get("held_out_scenario_status", metrics.get("held_out_scenario_status"))
    image_data = _images(root, warnings)
    artifacts = _artifacts(root, warnings)
    failed = status in {"failed", "incomplete"} or bool(error) or (
        not historical_training and training.get("termination_reason") in {"failed", "diagnostics_failed"})
    banner = ("本次运行失败或未完整结束；请先查看异常与停止原因。" if failed else
              "报告已汇总现有证据；训练步骤结束或Loss下降不代表模型已经通过验收。")
    if not training:
        banner += " 未找到本次训练报告；可能是仅推理运行、尚未训练或训练前即失败。"
    if historical_training:
        banner += " 本次仅推理；所加载模型的历史训练记录展示在下方，本次没有重新训练。"
    if external != "passed":
        banner += " 尚未完成可信外部准确度验证，不可宣称已达到民雄真实淹水预测精度。"
    if synthetic is True:
        banner += " 本次使用合成固定地形，不是民雄实测DEM。"

    sections = []
    sections.append("<h2>运行结论与适用范围</h2>" + _definition([
        ("运行状态", _status(status, "workflow")),
        ("操作", context.get("action")),
        ("历史训练步骤正常完成" if historical_training else "训练步骤正常完成", _status(training.get("training_completed"), "completed")),
        ("优化收敛", _status(training.get("converged"), "converged")),
        ("物理检查", _status(training.get("physics_checks_passed"), "physics")),
        ("外部参考验证", _status(external, "reference")),
        ("完整留出雨情验证", _status(heldout, "reference")),
        ("预定门槛评估", _status(assessment.get("status"), "assessment")),
        ("本次预测标签", simulation.get("validation_status")),
        ("停止原因", _status(training.get("termination_reason"), "termination")),
        ("训练证据来源", "所加载模型的历史训练记录；本次仅推理" if historical_training else "本次运行目录中的训练记录"),
        ("错误", error),
    ]))
    manual = context.get("requested_rainfall") if isinstance(context.get("requested_rainfall"), dict) else {}
    time_cfg = config.get("time") if isinstance(config.get("time"), dict) else {}
    inputs = context.get("input_files", fixed.get("input_files", config.get("terrain")))
    sections.append("<h2>固定地形与手动降雨</h2>" + _definition([
        ("地形使用规则", "使用指定的同一份地形文件；随机训练点不等于随机地形。"),
        ("网格标识", grid_id), ("固定地形指纹 SHA-256", fingerprint),
        ("合成数据", synthetic), ("输入文件", inputs),
        ("人工雨强 [mm/hr]", context.get("rainfall_mm_hr", manual.get("rainfall_mm_hr", simulation.get("rainfall_mm_hr")))),
        ("降雨持续时间 [s]", context.get("duration_s", manual.get("duration_s", simulation.get("rain_duration_s")))),
        ("总模拟时间 [s]", context.get("end_s", time_cfg.get("end_s", simulation.get("total_simulation_time_s")))),
        ("输出覆盖", simulation.get("coverage")),
    ]) + '<p class="muted">手动预测雨情与多雨情训练集合是两件事：只修改雨强不能使单情境权重自动适用于新雨情；参数化模型也必须在经过验证的适用范围内使用。</p>')

    phase_rows = []
    for phase, entry in history["phases"].items():
        last = entry["last"]
        phase_rows.append([phase, entry["evaluations"], last.get("optimizer_step"), last.get("closure_eval")]
                         + [last.get(key) for key, _ in _LOSSES]
                         + [last.get("learning_rate"), last.get("elapsed_s"),
                            last.get("gradient_norm_before_clip"), last.get("gradient_norm_after_clip")])
    sections.append("<h2>训练过程与各项Loss</h2>" + _definition([
        ("设备", training.get("device")), ("模型版本", training.get("model_version")),
        ("网络配置", config.get("model")),
        ("物理版本", training.get("physics_version")),
        ("优化耗时 [s]", training.get("elapsed_training_s")),
        ("训练及诊断总耗时 [s]", training.get("elapsed_total_s")),
        ("整个运行耗时 [s]", context.get("elapsed_s")),
        ("Adam完成步数", training.get("adam_steps_completed")),
        ("L-BFGS内部迭代数", training.get("lbfgs_iterations_completed")),
        ("L-BFGS目标求值次数", training.get("closure_evaluations")),
        ("历史记录行数", history["row_count"]),
        ("训练报告记录数值异常", training.get("nonfinite_detected")),
        ("含非有限/无效数值的日志行数", history["nonfinite_rows"]),
        ("独立PDE审计点数", audit_count), ("含非有限值的审计行数", audit_nonfinite),
    ]) + _table(["阶段", "记录/求值次数", "末optimizer_step", "末closure_eval"]
                + [label for _, label in _LOSSES] + ["末学习率", "末累计秒", "末梯度范数（裁剪前）", "末梯度范数（裁剪后）"], phase_rows)
       + '<p class="muted">表中为各阶段最后一次目标求值，L-BFGS closure次数不是epoch次数。缺失Data Loss为未启用/未评估，不是零误差；不同阶段最后一次求值也不能独立证明模型收敛。</p>')

    scenario_rows = []
    cases = metrics.get("scenarios") if isinstance(metrics.get("scenarios"), list) else []
    for case in cases:
        if not isinstance(case, dict):
            warnings.append("忽略无效的雨情指标条目")
            continue
        relative = _finite(case.get("relative_l2"))
        scenario_rows.append([
            case.get("scenario_id"), case.get("split"), case.get("reference_sample_count"),
            case.get("depth_mae_m"), case.get("depth_rmse_m"), relative*100 if relative is not None else None,
            case.get("velocity_rmse_m_s"), case.get("flood_iou"), case.get("flood_f1"),
            case.get("ic_h_rmse_m"), case.get("bc_discharge_rmse_m2_s"), case.get("pde_rms_normalized"),
        ])
    sections.append("<h2>物理残差与各雨情验证</h2>" + _definition([
        ("参考数据类型", metrics.get("reference_type")),
        ("参考数据SHA-256", training.get("reference_sha256")),
        ("分组数量", metrics.get("split_counts")),
        ("实际训练情境ID", training.get("training_scenario_ids")),
        ("训练参考点数", training.get("training_reference_count")),
    ]) + _table(["情境", "split", "参考点数", "深度MAE [m]", "深度RMSE [m]", "相对L2 [%]",
                  "速度RMSE [m/s]", "范围IoU", "范围F1", "IC深度RMSE [m]", "BC法向流量RMSE [m²/s]",
                  "PDE无量纲RMS（连续/x动量/y动量）"], scenario_rows)
       + '<p class="muted">无参考、零范数或空集等不可计算指标保持未评估。更细网格的同刻比较不等于完整留出雨情验证；有参考数值对比也不自动等于真实观测认证。</p>')

    balance = simulation.get("water_balance") if isinstance(simulation.get("water_balance"), dict) else {}
    sections.append("<h2>预测摘要与水量平衡</h2>" + _definition([
        ("预测状态", _status(simulation.get("status"), "workflow")),
        ("最大水深 [m]", simulation.get("max_depth_m")),
        ("最大水深采样时刻 [s]", simulation.get("max_depth_time_s")),
        ("最大流速 [m/s]", simulation.get("max_speed_m_s")),
        ("最大淹水面积 [m²]", simulation.get("max_flooded_area_m2")),
        ("总降雨体积 [m³]", simulation.get("total_rain_volume_m3")),
        ("账本初始储水 [m³]", balance.get("initial_storage_m3")),
        ("账本最终储水 [m³]", balance.get("final_storage_m3")),
        ("区间降雨 [m³]", balance.get("rain_volume_m3")),
        ("边界累计流入 [m³]", balance.get("boundary_in_m3")),
        ("边界累计流出 [m³]", balance.get("boundary_out_m3")),
        ("最终全域水量余差 [m³]", balance.get("error_m3")),
        ("相对余差（比值）", balance.get("relative_error")),
        ("最大时刻全域绝对余差 [m³]", simulation.get("max_global_balance_error_m3")),
        ("最大格级绝对余差 [m³]", simulation.get("max_cell_balance_error_m3")),
        ("水量状态", _status(balance.get("status", training.get("water_balance_status")), "water")),
    ]) + '<p class="muted">共享面在邻格中符号相反，只保证交换账本一致，不证明PINN严格离散守恒。干格显示掩膜不能掩盖原始速度或删除微小储水；仅请求部分帧时，峰值和水量只代表对应采样/区间。</p>')

    gates = assessment.get("gates") if isinstance(assessment.get("gates"), list) else []
    sections.append("<h2>预定门槛与异常</h2>" + _table(["检查项", "实测", "门槛", "状态"], [
        [gate.get("name"), gate.get("value"), gate.get("limit"), _status(gate.get("status"), "assessment")]
        for gate in gates if isinstance(gate, dict)
    ]))
    if warnings:
        sections.append("<ul class=\"warnings\">" + "".join(f"<li>{escape(w)}</li>" for w in warnings) + "</ul>")
    else:
        sections.append('<p class="muted">报告读取过程未发现额外异常；这不等于训练或预测已通过物理/精度验收。</p>')

    gallery = []
    for relative, encoded in image_data:
        gallery.append(f'<figure><figcaption>{escape(relative)}</figcaption><img alt="{escape(relative)}" '
                       f'src="data:image/png;base64,{encoded}" loading="lazy"></figure>')
    sections.append("<h2>训练与预测图表</h2>" + ("".join(gallery) if gallery else
                    '<p class="muted">本次没有可内嵌的图表，可能尚未生成、运行提前失败或图片超出大小限制。</p>'))
    sections.append("<h2>原始文件与复核入口</h2><ul>" + "".join(
        f'<li><a href="{escape(quote(relative, safe="/"), quote=True)}">{escape(relative)}</a> '
        f'<span class="muted">({size:,} bytes)</span></li>' for relative, size in artifacts) + "</ul>"
        + '<p class="muted">HTML中的图表已内嵌，可离线查看；原始文件链接需要保留相同相对目录。报告不会更改原始CSV/JSON或把失败结果改为通过。</p>')
    sections.append("<details><summary>完整训练配置、固定地形元数据与运行上下文</summary><h3>训练配置</h3><pre>"
                    + escape(_json_text(config)) + "</pre><h3>运行上下文</h3><pre>"
                    + escape(_json_text(context)) + "</pre><h3>固定地形元数据</h3><pre>"
                    + escape(_json_text(context.get("terrain_metadata", fixed.get("metadata", {})))) + "</pre></details>")
    html = """<!doctype html>
<html lang="zh-Hans"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PINN 模型训练成果报告</title><style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#172a3a;background:#f2f5f8;margin:0;line-height:1.65}
main{max-width:1200px;margin:24px auto;padding:28px;background:white;border-radius:12px}
h1{margin-top:0;color:#123d58}h2{margin-top:32px;border-bottom:2px solid #dce8f0;padding-bottom:6px}
.notice{padding:16px;background:#fff2d8;border-left:5px solid #b97700}.muted{color:#596a76;font-size:.92em}
.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9em;margin:12px 0}
th,td{text-align:left;border:1px solid #dce3e9;padding:8px;vertical-align:top;overflow-wrap:anywhere}th{background:#eaf1f6}
tr:nth-child(even){background:#f8fafc}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f7;padding:14px}
figure{margin:18px 0}img{display:block;max-width:100%;height:auto}figcaption{font-size:.9em;color:#465d6d}
a{color:#075e8c}.warnings{background:#fff5e6;padding:14px 14px 14px 34px}@media print{body{background:white}main{margin:0;padding:0}details{display:block}a{color:inherit}}
</style></head><body><main><h1>PINN 模型训练成果报告</h1><p class="notice">""" + escape(banner) + "</p>" + "".join(sections) + "</main></body></html>\n"
    destination = root / "training_results_report.html"
    _atomic_text(destination, html)

    markdown = ["# PINN 模型训练成果报告", "", banner, "", "完整图表及表格见 [HTML报告](training_results_report.html)。", "",
                "| 项目 | 记录 |", "|---|---|",
                f"| 运行状态 | {_md(_status(status, 'workflow'))} |",
                f"| 训练完成 | {_md(_status(training.get('training_completed'), 'completed'))} |",
                f"| 收敛 | {_md(_status(training.get('converged'), 'converged'))} |",
                f"| 物理检查 | {_md(_status(training.get('physics_checks_passed'), 'physics'))} |",
                f"| 外部验证 | {_md(_status(external, 'reference'))} |",
                f"| 门槛评估 | {_md(_status(assessment.get('status'), 'assessment'))} |",
                f"| 地形ID | {_md(grid_id)} |", f"| 地形指纹 | {_md(fingerprint)} |",
                f"| 优化秒数 | {_md(training.get('elapsed_training_s'))} |",
                f"| 总Loss末值（分阶段） | {_md({phase: entry['last'].get('total_loss') for phase, entry in history['phases'].items()})} |",
                f"| 最大全域水量余差 m³ | {_md(simulation.get('max_global_balance_error_m3'))} |",
                f"| 错误 | {_md(error)} |", "", "## 异常与限制", ""]
    markdown.extend("- " + _md(w) for w in warnings)
    markdown.extend(["", "- 没有可信参考时，尚未完成外部验证；Loss下降不证明真实预测准确。",
                     "- 地形来自指定固定文件；训练采样点可以随机，但不改变地形。", "", "## 文件", ""])
    markdown.extend(f"- [{_md(relative)}]({quote(relative, safe='/')})（{size:,} bytes）" for relative, size in artifacts)
    _atomic_text(root / "training_results_report.md", "\n".join(markdown) + "\n")
    return destination


def _md(value):
    # Markdown engines may allow raw HTML; escape it just as in the HTML report.
    return escape(_fmt(value), quote=False).replace("|", "\\|").replace("\n", "<br>")
