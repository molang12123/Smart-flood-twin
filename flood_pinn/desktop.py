"""Fixed-file workflow for editor execution, with manual rainfall and a portable report."""
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from datetime import datetime, timezone
import uuid
import webbrowser

from .contracts import write_json

ROOT = Path(__file__).resolve().parents[1]


def _resolve(parent, value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}必须填写文件或目录路径")
    return (parent / value).resolve()


def _positive_number(value, label, *, zero=True):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{label}必须是{'非负' if zero else '正'}有限数值")
    return float(value)


def load_settings(path):
    path = Path(path).resolve()
    settings = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(settings, dict):
        raise ValueError("运行设置必须为JSON对象")
    allowed = {"action","base_config","rainfall_mm_hr","duration","output_root",
               "model_dir","allow_unvalidated","batch_size","open_report","standards"}
    if set(settings)-allowed:
        raise ValueError(f"不支持的设置字段：{sorted(set(settings)-allowed)}")
    if settings.get("action") not in {"train_and_predict","train","predict"}:
        raise ValueError("action必须为train_and_predict、train或predict")
    for key in ("allow_unvalidated","open_report"):
        if not isinstance(settings.get(key, False), bool):
            raise ValueError(f"{key}必须为true或false")
    batch = settings.get("batch_size",4096)
    if isinstance(batch,bool) or not isinstance(batch,int) or batch <= 0:
        raise ValueError("batch_size必须为正整数")
    settings["batch_size"] = batch
    settings["rainfall_mm_hr"] = _positive_number(settings.get("rainfall_mm_hr"),"rainfall_mm_hr")
    duration = settings.get("duration")
    if not isinstance(duration,dict) or set(duration) != {"value","unit"} or duration["unit"] not in {"s","min","h"}:
        raise ValueError("duration必须含value和unit（s/min/h）")
    value = _positive_number(duration["value"],"duration.value")
    settings["duration_s"] = value*{"s":1.,"min":60.,"h":3600.}[duration["unit"]]
    if not math.isfinite(settings["duration_s"]):
        raise ValueError("转换后的降雨时长超出有限数范围")
    settings["base_config"] = _resolve(path.parent,settings.get("base_config"),"base_config")
    settings["output_root"] = _resolve(path.parent,settings.get("output_root"),"output_root")
    if settings["action"] == "predict":
        settings["model_dir"] = _resolve(path.parent,settings.get("model_dir"),"model_dir")
    if settings.get("standards") is not None:
        settings["standards"] = _resolve(path.parent,settings["standards"],"standards")
    return settings


def _manual_scenario_check(config, settings):
    """Fail before expensive training if the requested demonstration cannot infer."""
    end = config["time"]["end_s"]
    duration = settings["duration_s"]
    if duration > end:
        raise ValueError(f"降雨时长{duration:g}s超过配置的模拟总时间{end:g}s；请先定义相应的训练时间范围")
    if settings["action"] == "train":
        return
    training = [s for s in config["scenarios"] if s["split"] == "train"]
    rain = settings["rainfall_mm_hr"]
    if not config.get("model",{}).get("parameterized",True):
        if len(training) != 1 or rain != training[0]["rainfall_mm_hr"] or duration != training[0]["duration_s"]:
            raise ValueError("单情境模型只支持其训练雨情；修改雨势需训练该情境或使用参数化模型")
    elif not (min(s["rainfall_mm_hr"] for s in training) <= rain <= max(s["rainfall_mm_hr"] for s in training)
              and min(s["duration_s"] for s in training) <= duration <= max(s["duration_s"] for s in training)):
        raise ValueError("手动雨情超出已定义的训练范围；请先修改训练情境/时间尺度并重新训练，不能直接外推")


def run_workflow(settings_path):
    """Always produce a human report, including configuration/numerical failures.

    All file paths in settings are relative to that settings file. Base terrain
    paths are relative to the base model config. No terrain generator is called.
    Repeated runs receive distinct directories and preserve all previous results.
    """
    started = perf_counter()
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
    folder = None
    context = {"status":"running","settings_file":str(Path(settings_path).resolve()),
               "warnings":[], "terrain_policy":"Read fixed CSV files; never generate/randomize terrain"}
    settings = {}
    try:
        settings = load_settings(settings_path)
        candidate = settings["output_root"]/name
        candidate.mkdir(parents=True,exist_ok=False)
        folder = candidate
        context.update(action=settings["action"],rainfall_mm_hr=settings["rainfall_mm_hr"],
                       duration_s=settings["duration_s"],run_dir=str(folder))
        from .platform_info import prepare_runtime_cache
        prepare_runtime_cache()
        from .__main__ import load_config
        from .training import validate_config
        grid, config = load_config(settings["base_config"])
        config = validate_config(config)
        context.update(terrain_fingerprint=grid.fingerprint,grid_id=grid.metadata["grid_id"],
                       synthetic=grid.metadata["synthetic"],end_s=config["time"]["end_s"],
                       input_files={key:str((settings["base_config"].parent/value).resolve())
                                    for key,value in config["terrain"].items()
                                    if key in {"csv","metadata","roughness"} and value is not None},
                       terrain_metadata=grid.to_metadata())
        if context["synthetic"]:
            context["warnings"].append("当前是固定的合成示例地形，并非民雄真实DEM。")
        if settings.get("allow_unvalidated",False):
            context["warnings"].append("已显式开启研究模式；预测不代表已通过物理/外部精度验收。")
        _manual_scenario_check(config,settings)
        write_json(folder/"effective_config.json",config)
        print(f"固定地形：{context['input_files']['csv']}")
        print(f"地形指纹：{grid.fingerprint}")
        print(f"手动雨情：{settings['rainfall_mm_hr']:g} mm/hr，持续 {settings['duration_s']:g} s")
        model_dir = settings.get("model_dir")
        if settings["action"] in {"train","train_and_predict"}:
            print("开始有时间预算的训练；训练情境来自配置，地形不变。")
            from .training import train
            from .checkpoint import save_checkpoint
            model, report = train(grid,config,folder/"training")
            if not report["training_completed"]:
                raise RuntimeError(f"训练未正常完成：{report['termination_reason']}")
            model_dir = folder/"training"/"model"
            save_checkpoint(model,grid,config,model_dir,report)
            context["model_dir"] = str(model_dir)
        if settings["action"] in {"predict","train_and_predict"}:
            from .predict import Predictor
            predictor = Predictor.from_checkpoint(model_dir,grid,conditions=config["conditions"])
            context["model_dir"] = str(model_dir)
            if settings["action"] == "predict":
                context["loaded_model_training_report"] = predictor.metadata["training_report"]
            result = predictor.predict(settings["rainfall_mm_hr"],settings["duration_s"]/60,
                                       [0.,config["time"]["end_s"]],
                                       batch_size=settings["batch_size"],output_dir=folder/"prediction",
                                       allow_unvalidated=settings.get("allow_unvalidated",False))
            print("开始模型推理和结果导出；此过程不重新训练。")
            for _ in result.frames_iterator:
                pass
            result.finalize()
            context["validation_status"] = result.validation_status
        if settings.get("standards") is not None and (folder/"training/validation_metrics.json").exists() and (folder/"prediction/simulation_summary.json").exists():
            from .assessment import assess
            assessment = assess(json.loads((folder/"training/validation_metrics.json").read_text(encoding="utf-8-sig")),
                                json.loads((folder/"prediction/simulation_summary.json").read_text(encoding="utf-8-sig")),
                                json.loads(settings["standards"].read_text(encoding="utf-8-sig")))
            write_json(folder/"assessment.json",assessment)
            context["diagnostic_assessment_status"] = assessment["status"]
        context["status"] = "complete"
    except Exception as exc:
        context.update(status="failed",error=f"{type(exc).__name__}: {exc}")
    finally:
        if folder is None:
            folder = ROOT/"outputs"/"vscode"/name
            folder.mkdir(parents=True,exist_ok=False)
        context["elapsed_s"] = perf_counter()-started
        context["run_dir"] = str(folder)
        write_json(folder/"workflow_context.json",context)
        from .report import generate_training_report
        report_path = generate_training_report(folder,context=context)
    if settings.get("open_report",False):
        if os.name == "nt" or sys.platform == "darwin" or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            try:
                webbrowser.open(report_path.resolve().as_uri())
            except Exception:
                print("报告已生成，请手动打开HTML文件。")
    return {**context,"run_dir":folder,"report_path":report_path}
