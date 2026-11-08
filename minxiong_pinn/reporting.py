from __future__ import annotations

import base64
import csv
import io
import json
from pathlib import Path


def generate_report(output_dir: str | Path):
    output = Path(output_dir)
    report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
    history = list(csv.DictReader((output / "training_history.csv").open(encoding="utf-8")))
    events = list(csv.DictReader((output / "event_metrics.csv").open(encoding="utf-8")))
    from flood_pinn.platform_info import prepare_runtime_cache
    prepare_runtime_cache()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def figure_data(kind):
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
        if kind == "history":
            ax.semilogy([int(r["step"]) for r in history], [float(r["total_loss"]) for r in history], label="train batch")
            points = [(int(r["step"]), float(r["validation_loss"])) for r in history if r.get("validation_loss")]
            ax.semilogy([x for x, _ in points], [y for _, y in points], "o-", label="validation")
            ax.set(xlabel="Adam step", ylabel="dimensionless loss", title="Training and validation")
        else:
            colors = {"train": "#4477aa", "validation": "#ee9944", "test": "#228833"}
            for split in colors:
                rows = [r for r in events if r["split"] == split]
                ax.scatter([float(r["rainfall_mm_hr"]) * float(r["duration_s"]) / 3600 for r in rows],
                           [float(r["mass_loss"]) for r in rows], s=18, alpha=.75, label=split, color=colors[split])
            ax.set_yscale("log"); ax.set(xlabel="event rainfall depth (mm)", ylabel="mass loss", title="Event water-balance constraint")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        buffer = io.BytesIO(); fig.savefig(buffer, format="png"); plt.close(fig)
        return base64.b64encode(buffer.getvalue()).decode()

    status = "通过研究级物理门槛" if report["physics_training_passed"] else "未通过物理门槛"
    rows = "".join(f"<tr><th>{name}</th><td>{value['events']}</td><td>{value['mean_pde_loss']:.6g}</td><td>{value['mean_mass_loss']:.6g}</td><td>{value['worst_loss']:.6g}</td></tr>"
                   for name, value in report["splits"].items())
    assumptions = "".join(f"<li>{item}</li>" for item in report["assumptions"])
    html = f"""<!doctype html><html lang='zh-Hant'><meta charset='utf-8'><title>民雄 PINN 訓練成果</title>
<style>body{{font:16px/1.6 system-ui;max-width:1050px;margin:35px auto;padding:0 20px;color:#17202a}}h1,h2{{color:#174a6e}}.status{{padding:14px;background:#eef7f1;border-left:5px solid #29804b}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border:1px solid #ccd6dd;text-align:right}}th:first-child{{text-align:left}}img{{max-width:100%}}code{{background:#eef2f5;padding:2px 5px}}</style>
<h1>民雄固定地形參數化 PINN 訓練成果</h1><p class='status'><b>{status}</b><br>完成 {report['completed_steps']} 步，耗時 {report['elapsed_s']:.1f} 秒；最佳驗證 loss {report['best_validation_loss']:.6g}。</p>
<h2>獨立事件評估</h2><table><tr><th>資料組</th><th>事件數</th><th>平均 PDE loss</th><th>平均水量 loss</th><th>最差總 loss</th></tr>{rows}</table>
<p>全部 151 個事件中的最差水量 loss：<b>{report['worst_all_event_mass_loss']:.6g}</b>。驗證與測試事件沒有加入參數更新。</p>
<img src='data:image/png;base64,{figure_data('history')}'><img src='data:image/png;base64,{figure_data('events')}'>
<h2>固定地形</h2><pre>{json.dumps(report['terrain'], ensure_ascii=False, indent=2)}</pre>
<h2>目前物理假設</h2><ul>{assumptions}</ul>
<h2>結果邊界</h2><p>目前沒有實測淹水深度、流速或可信水動力模型結果，因此 <b>external flood observation validation = not available</b>。本報告的「通過」只代表設定的 PINN 方程殘差、封閉域邊界與全域水量門檻通過，不代表真實民雄淹水精度已驗證。</p>
</html>"""
    path = output / "training_results_report.html"
    path.write_text(html, encoding="utf-8")
    return path
