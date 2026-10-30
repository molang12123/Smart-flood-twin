"""VS Code F5 entry: edit configs/vscode_run.json, then Run and Debug.

No interactive terminal input is required. Terrain is read from fixed CSV files;
random collocation points never generate or modify terrain.
"""
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path,
                        default=Path(__file__).resolve().parent / "configs" / "vscode_run.json")
    args = parser.parse_args()
    from flood_pinn.desktop import run_workflow
    result = run_workflow(args.settings)
    print(f"\n运行状态：{result['status']}")
    print(f"训练成果报告：{result['report_path']}")
    if result.get("error"):
        print(f"原因：{result['error']}")
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
