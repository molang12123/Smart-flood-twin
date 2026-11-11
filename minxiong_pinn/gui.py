"""Tkinter desktop interface for fixed-terrain Minxiong PINN prediction."""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

from .predict import load_model, predict
from .terrain import MaskedRasterTerrain
from .user_input import build_prediction_request


ROOT = Path(__file__).resolve().parents[1]
TERRAIN_PATH = ROOT / "data" / "minxiong" / "裁剪后民雄.tif"
MODEL_PATH = ROOT / "models" / "minxiong" / "model.pt"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "minxiong" / "user_predictions"


class PredictionApp:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk

        self.root, self.tk, self.ttk = root, tk, ttk
        self.terrain = self.model = self.checkpoint = None
        root.title("民雄固定地形 PINN 淹水预测")
        root.geometry("760x650")
        root.minsize(700, 600)
        outer = ttk.Frame(root, padding=18); outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="民雄固定地形 PINN 淹水预测", font=("Microsoft JhengHei UI", 18, "bold")).pack(anchor="w")
        ttk.Label(outer, text="输入平均雨势、降雨时长、模拟总时长与输出间隔；地形和模型固定。",
                  foreground="#444").pack(anchor="w", pady=(4, 14))
        form = ttk.LabelFrame(outer, text="预测参数", padding=14); form.pack(fill="x")
        self.rainfall = self._row(form, 0, "平均雨势", "10", "mm/hr", units=None)
        self.rain_duration, self.rain_unit = self._row(form, 1, "降雨时长", "2", None, units=("小时", "分钟"))
        self.simulation, self.simulation_unit = self._row(form, 2, "模拟总时长", "3", None, units=("小时", "分钟"))
        self.interval, self.interval_unit = self._row(form, 3, "输出间隔", "60", None, units=("分钟", "小时"))
        output = ttk.Frame(form); output.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(9, 0))
        output.columnconfigure(1, weight=1)
        ttk.Label(output, text="输出目录").grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.output_root = tk.StringVar(value=str(DEFAULT_OUTPUT_ROOT))
        ttk.Entry(output, textvariable=self.output_root).grid(row=0, column=1, sticky="ew")
        ttk.Button(output, text="选择…", command=self._choose_output).grid(row=0, column=2, padx=(8, 0))
        self.domain_label = ttk.Label(outer, text="正在读取固定地形与模型…", foreground="#174a6e")
        self.domain_label.pack(anchor="w", pady=(14, 6))
        self.estimate_label = ttk.Label(outer, text=""); self.estimate_label.pack(anchor="w")
        actions = ttk.Frame(outer); actions.pack(fill="x", pady=12)
        self.run_button = ttk.Button(actions, text="开始预测", command=self._start_prediction, state="disabled")
        self.run_button.pack(side="left")
        self.open_button = ttk.Button(actions, text="打开结果目录", command=self._open_result, state="disabled")
        self.open_button.pack(side="left", padx=8)
        self.progress = ttk.Progressbar(outer, mode="determinate"); self.progress.pack(fill="x")
        self.status = tk.Text(outer, height=13, wrap="word", state="disabled")
        self.status.pack(fill="both", expand=True, pady=(10, 0))
        self.last_output = None
        for variable in (self.rainfall, self.rain_duration, self.simulation, self.interval):
            variable.trace_add("write", lambda *_: self._refresh_estimate())
        for combo in (self.rain_unit, self.simulation_unit, self.interval_unit):
            combo.bind("<<ComboboxSelected>>", lambda *_: self._refresh_estimate())
        threading.Thread(target=self._load_backend, daemon=True).start()

    def _row(self, parent, row, label, default, suffix, units):
        parent.columnconfigure(1, weight=1)
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 12))
        value = self.tk.StringVar(value=default)
        self.ttk.Entry(parent, textvariable=value, width=20).grid(row=row, column=1, sticky="ew", pady=5)
        if units is None:
            self.ttk.Label(parent, text=suffix).grid(row=row, column=2, sticky="w", padx=8)
            return value
        combo = self.ttk.Combobox(parent, values=units, state="readonly", width=10)
        combo.set(units[0]); combo.grid(row=row, column=2, sticky="w", padx=8)
        return value, combo

    def _log(self, message):
        self.status.configure(state="normal"); self.status.insert("end", message + "\n")
        self.status.see("end"); self.status.configure(state="disabled")

    def _load_backend(self):
        try:
            terrain = MaskedRasterTerrain(TERRAIN_PATH, manning_n=.03)
            model, checkpoint = load_model(MODEL_PATH, terrain)
        except Exception as exc:
            self.root.after(0, lambda exc=exc: self._backend_failed(exc))
            return
        self.terrain, self.model, self.checkpoint = terrain, model, checkpoint
        self.root.after(0, self._backend_ready)

    def _backend_failed(self, exc):
        from tkinter import messagebox
        self.domain_label.configure(text="模型载入失败", foreground="#a00000")
        self._log(f"载入失败：{type(exc).__name__}: {exc}")
        messagebox.showerror("模型载入失败", str(exc))

    def _backend_ready(self):
        domain = self.checkpoint["training_rain_domain"]
        self.domain_label.configure(
            text=(f"模型支持：雨势 {domain['rainfall_mm_hr'][0]:g}–{domain['rainfall_mm_hr'][1]:g} mm/hr；"
                  f"降雨 {domain['duration_s'][0] / 3600:g}–{domain['duration_s'][1] / 3600:g} 小时；"
                  f"模拟最长 {self.model.scales['time_s'] / 3600:g} 小时。"))
        self.run_button.configure(state="normal")
        self._log(f"固定地形已载入：{self.terrain.active_count:,} 个有效20米网格。")
        self._log("结果属于当前物理假设下的研究级理论预测。")
        self._refresh_estimate()

    def _values(self):
        return build_prediction_request(
            rainfall_mm_hr=self.rainfall.get(),
            rain_duration_value=self.rain_duration.get(), rain_duration_unit=self.rain_unit.get(),
            simulation_value=self.simulation.get(), simulation_unit=self.simulation_unit.get(),
            interval_value=self.interval.get(), interval_unit=self.interval_unit.get(),
            checkpoint=self.checkpoint, model_time_s=self.model.scales["time_s"],
            active_cells=self.terrain.active_count)

    def _refresh_estimate(self):
        if self.checkpoint is None: return
        try:
            request = self._values()
            size_gib = request.estimated_uncompressed_bytes / 1024 ** 3
            self.estimate_label.configure(text=f"将输出 {request.frame_count} 帧；估计未压缩数据约 {size_gib:.2f} GiB。", foreground="#555")
        except Exception as exc:
            self.estimate_label.configure(text=str(exc), foreground="#a00000")

    def _choose_output(self):
        from tkinter import filedialog
        selected = filedialog.askdirectory(initialdir=self.output_root.get() or str(ROOT))
        if selected: self.output_root.set(selected)

    def _start_prediction(self):
        from tkinter import messagebox
        try:
            request = self._values()
            root = Path(self.output_root.get()).expanduser().resolve()
            if root.exists() and not root.is_dir(): raise ValueError("输出路径不是目录")
        except Exception as exc:
            messagebox.showerror("输入无效", str(exc)); return
        if request.frame_count > 120:
            size_gib = request.estimated_uncompressed_bytes / 1024 ** 3
            if not messagebox.askyesno("输出较大", f"将产生 {request.frame_count} 帧，估计未压缩数据约 {size_gib:.2f} GiB。\n\n是否继续？"):
                return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        target = root / f"prediction_{stamp}"
        self.run_button.configure(state="disabled"); self.open_button.configure(state="disabled")
        self.progress.configure(maximum=request.frame_count, value=0)
        self._log(f"开始预测：{request.rainfall_mm_hr:g} mm/hr，降雨 {request.rain_duration_s / 3600:g} 小时，{request.frame_count} 帧。")
        if request.simulation_s < request.rain_duration_s:
            self._log("提示：模拟结束时降雨仍在持续。")
        threading.Thread(target=self._predict_worker, args=(request, target), daemon=True).start()

    def _predict_worker(self, request, target):
        try:
            summary = predict(self.model, self.checkpoint, self.terrain, request.rainfall_mm_hr,
                              request.rain_duration_s, request.output_times_s, target,
                              progress_callback=lambda done, total, _t, _frame:
                                  self.root.after(0, self._progress_update, done, total))
        except Exception as exc:
            self.root.after(0, self._prediction_failed, exc); return
        self.root.after(0, self._prediction_complete, target, summary)

    def _progress_update(self, done, total):
        self.progress.configure(maximum=total, value=done)
        self._log(f"已完成 {done}/{total} 帧")

    def _prediction_failed(self, exc):
        from tkinter import messagebox
        self.run_button.configure(state="normal")
        self._log(f"预测失败：{type(exc).__name__}: {exc}")
        messagebox.showerror("预测失败", str(exc))

    def _prediction_complete(self, target, summary):
        from tkinter import messagebox
        self.last_output = target; self.run_button.configure(state="normal"); self.open_button.configure(state="normal")
        peak_depth = max(frame["max_depth_m"] for frame in summary["frames"])
        peak_speed = max(frame["max_speed_m_s"] for frame in summary["frames"])
        self._log(f"预测完成：最大水深 {peak_depth:.4g} m，最大流速 {peak_speed:.4g} m/s")
        self._log(f"结果目录：{target}")
        messagebox.showinfo("预测完成", f"已输出 {len(summary['frames'])} 帧。\n\n结果目录：\n{target}")

    def _open_result(self):
        if self.last_output is None: return
        if os.name == "nt": os.startfile(self.last_output)  # type: ignore[attr-defined]
        elif sys.platform == "darwin": subprocess.Popen(["open", str(self.last_output)])
        else: subprocess.Popen(["xdg-open", str(self.last_output)])


def launch():
    import tkinter as tk
    root = tk.Tk()
    PredictionApp(root)
    root.mainloop()
