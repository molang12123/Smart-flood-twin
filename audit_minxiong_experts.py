from __future__ import annotations

import json
import sys

from minxiong_pinn.overlap_audit import audit_and_publish


def main(config_path="configs/minxiong_experts_0_800.json"):
    report = audit_and_publish(config_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["activation_approved"] else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else "configs/minxiong_experts_0_800.json"))
