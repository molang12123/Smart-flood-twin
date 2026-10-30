import base64
import csv
import json
from pathlib import Path
import tempfile
import unittest

from flood_pinn.report import generate_training_report


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jkL8AAAAASUVORK5CYII="
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class ReportTests(unittest.TestCase):
    def fixture(self, root):
        write_json(root / "training/training_report.json", {
            "training_completed": True, "converged": False, "physics_checks_passed": None,
            "reference_validation_status": "comparison_only_not_accuracy_approval",
            "held_out_scenario_status": "physics_audited_not_accuracy_validated",
            "termination_reason": "configured_steps_completed", "elapsed_training_s": 2.5,
            "adam_steps_completed": 3, "lbfgs_iterations_completed": 2, "closure_evaluations": 4,
            "terrain_fingerprint": "abc123", "training_scenario_ids": ["train"],
            "config": {"time": {"end_s": 6}, "scenarios": [
                {"scenario_id": "train", "rainfall_mm_hr": 40, "duration_s": 3, "split": "train"}
            ]},
        })
        write_json(root / "training/validation_metrics.json", {
            "reference_type": "analytic_flat_rain", "split_counts": {"train": 1, "test": 1},
            "scenarios": [{"scenario_id": "test", "split": "test", "reference_sample_count": 5,
                           "depth_mae_m": .01, "depth_rmse_m": .02, "relative_l2": .25,
                           "pde_rms_normalized": [.1, .2, .3]}],
        })
        with (root / "training/training_history.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["phase", "optimizer_step", "closure_eval", "total_loss", "pde_loss",
                      "ic_loss", "bc_loss", "data_loss", "learning_rate", "elapsed_s", "nonfinite"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow(dict(phase="adam", optimizer_step=1, total_loss=10, pde_loss=4,
                                 ic_loss=1, bc_loss=1, data_loss="", learning_rate=.001,
                                 elapsed_s=.1, nonfinite=False))
            writer.writerow(dict(phase="adam", optimizer_step=3, total_loss=5, pde_loss=2,
                                 ic_loss=.5, bc_loss=.1, data_loss="", learning_rate=.001,
                                 elapsed_s=1.5, nonfinite=False))
            writer.writerow(dict(phase="lbfgs", optimizer_step=4, closure_eval=4, total_loss=3,
                                 pde_loss=1, ic_loss=.1, bc_loss=.1, data_loss="", learning_rate=.1,
                                 elapsed_s=2.5, nonfinite=False))
        write_json(root / "prediction/simulation_summary.json", {
            "status": "complete", "validation_status": "research_unvalidated", "grid_id": "fixed-grid",
            "rainfall_mm_hr": 60, "rain_duration_s": 3, "coverage": "complete",
            "max_depth_m": .02, "max_global_balance_error_m3": .1,
            "water_balance": {"initial_storage_m3": .01, "final_storage_m3": 2,
                              "rain_volume_m3": 2.1, "error_m3": -.1, "status": "measured_not_certified"},
        })
        write_json(root / "assessment.json", {
            "status": "failed", "gates": [{"name": "water", "value": .1, "limit": .01, "status": "failed"}]
        })
        (root / "training/loss.png").write_bytes(PNG)
        return {"status": "complete", "action": "train_and_predict", "synthetic": True,
                "grid_id": "fixed-grid", "rainfall_mm_hr": 60, "duration_s": 3,
                "input_files": {"terrain": "data/fixed.csv"}}

    def test_complete_run_embeds_plots_and_distinguishes_outcomes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            context = self.fixture(root)
            original = (root / "training/training_report.json").read_bytes()
            report = generate_training_report(root, context=context)
            text = report.read_text(encoding="utf-8")
            self.assertEqual(report.name, "training_results_report.html")
            self.assertIn("data:image/png;base64,", text)
            self.assertIn("abc123", text)
            self.assertIn("人工雨强", text)
            self.assertIn("固定地形", text)
            self.assertIn("已完成配置中的训练步骤", text)
            self.assertIn("未确认收敛", text)
            self.assertIn("已做参考对比，未获准确度认证", text)
            self.assertIn("尚未完成可信外部准确度验证", text)
            self.assertIn("<td>adam</td><td>2</td>", text)
            self.assertIn("<td>lbfgs</td><td>1</td>", text)
            self.assertIn('href="training/training_report.json"', text)
            self.assertNotIn(str(root), text)
            self.assertEqual((root / "training/training_report.json").read_bytes(), original)
            markdown = (root / "training_results_report.md").read_text(encoding="utf-8")
            self.assertIn("[HTML报告](training_results_report.html)", markdown)
            self.assertIn("未获准确度认证", markdown)

    def test_failed_or_missing_run_produces_honest_report_and_escapes_html(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = generate_training_report(root, context={
                "status": "failed", "error": '<script>alert("unsafe")</script>',
                "warnings": ["<img src=x onerror=bad>"],
            })
            text = path.read_text(encoding="utf-8")
            self.assertIn("运行失败或未完整结束", text)
            self.assertIn("未找到本次训练报告", text)
            self.assertIn("未评估 / 未提供证据", text)
            self.assertIn("&lt;script&gt;", text)
            self.assertNotIn("<script>", text)
            self.assertNotIn("<img src=x", text)
            self.assertNotIn("报告记录物理检查通过", text)

    def test_nonfinite_evidence_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "training").mkdir()
            (root / "training/training_report.json").write_text('{"training_completed":true,"loss":NaN}')
            (root / "training/training_history.csv").write_text(
                "phase,total_loss,pde_loss,nonfinite\nadam,NaN,Inf,True\n", encoding="utf-8")
            text = generate_training_report(root).read_text(encoding="utf-8")
            self.assertIn("不允许非有限JSON数值", text)
            self.assertIn("异常：非有限数值", text)
            self.assertIn("源证据中检测到非有限值", text)

    def test_external_symlink_images_and_json_are_not_read(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d); root = folder / "run"; (root / "training").mkdir(parents=True)
            external = folder / "outside.png"; external.write_bytes(PNG)
            (root / "training/loss.png").symlink_to(external)
            outside_json = folder / "secret.json"
            outside_json.write_text('{"private":"DO_NOT_READ_ME"}')
            (root / "training/training_report.json").symlink_to(outside_json)
            text = generate_training_report(root).read_text(encoding="utf-8")
            self.assertNotIn("data:image/png;base64,", text)
            self.assertNotIn("DO_NOT_READ_ME", text)
            self.assertIn("运行目录外", text)

    def test_budget_stop_and_invalid_shapes_do_not_claim_success(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_json(root / "training/training_report.json", {
                "training_completed": False, "converged": False,
                "termination_reason": "time_budget_exhausted", "physics_checks_passed": False,
            })
            write_json(root / "training/validation_metrics.json", {"scenarios": [None, "bad"]})
            text = generate_training_report(root).read_text(encoding="utf-8")
            self.assertIn("达到时间预算，训练停止", text)
            self.assertIn("未正常完成全部训练步骤", text)
            self.assertIn("物理检查未通过", text)
            self.assertIn("忽略无效的雨情指标条目", text)
            self.assertNotIn("报告记录物理检查通过", text)

    def test_predict_only_marks_loaded_training_as_historical(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            report = generate_training_report(root, context={
                "status": "complete", "action": "predict",
                "loaded_model_training_report": {
                    "training_completed": True, "converged": False,
                    "termination_reason": "configured_steps_completed",
                    "reference_validation_status": "not_completed", "terrain_fingerprint": "loaded-fixed-terrain",
                    "config": {"time": {"end_s": 10}},
                },
                "terrain_metadata": {"grid_id": "fixed", "synthetic": True},
            })
            text = report.read_text(encoding="utf-8")
            self.assertIn("所加载模型的历史训练记录；本次仅推理", text)
            self.assertIn("本次没有重新训练", text)
            self.assertIn("loaded-fixed-terrain", text)
            self.assertNotIn("未找到本次训练报告", text)
            self.assertFalse((root / "training").exists())


if __name__ == "__main__":
    unittest.main()
