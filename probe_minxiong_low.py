"""Run short, reproducible low-rain ablations before a full expert campaign."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from train_minxiong_experts import run_expert_config


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "configs" / "minxiong_experts_0_800.json"

VARIANTS = {
    "uniform": {
        "hydrostatic_depth_baseline": False,
        "hydrostatic_learned_blend": 1.0,
        "hydrostatic_uniform_floor": 0.0,
        "hydrostatic_blend_start_step": 0,
        "hydrostatic_blend_ramp_steps": 0,
    },
    "hydro90": {
        "hydrostatic_depth_baseline": True,
        "hydrostatic_learned_blend": 1.0,
        "hydrostatic_uniform_floor": 0.9,
        "hydrostatic_blend_start_step": 0,
        "hydrostatic_blend_ramp_steps": 200,
    },
    "uniform_strict": {
        "hydrostatic_depth_baseline": False,
        "hydrostatic_learned_blend": 1.0,
        "hydrostatic_uniform_floor": 0.0,
        "hydrostatic_blend_start_step": 0,
        "hydrostatic_blend_ramp_steps": 0,
        "robust_pde_warmup_steps": 0,
        "learning_rate": 0.00003,
    },
}


def probe_config(variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of: {', '.join(VARIANTS)}")
    master = json.loads(BASE_CONFIG.read_text(encoding="utf-8-sig"))
    master["output_root"] = f"../outputs/minxiong/v10_probes/{variant}"
    master["experts"] = [copy.deepcopy(master["experts"][0])]
    training = master["training"]
    training.update({
        "adam_steps": 800,
        "max_seconds": 1800,
        "depth_warmup_steps": 0,
        "depth_gradient_scale": 1.0,
        "robust_pde_warmup_steps": 400,
        "separate_output_gradient_clipping": True,
        "depth_gradient_clip": 100.0,
        "velocity_gradient_clip": 100.0,
        "validation_interval": 100,
        "validation_repeats": 3,
        "validation_interior_points": 256,
        "final_evaluation_repeats": 1,
        "final_evaluation_interior_points": 256,
        "checkpoint_interval": 50,
    })
    training.update(VARIANTS[variant])
    training["assumptions"] = list(training["assumptions"])
    training["assumptions"][4] = (
        "Short v10 low-rain ablation; this output is diagnostic and cannot be deployed."
    )
    return master


def main(variant="uniform"):
    return run_expert_config(probe_config(variant), BASE_CONFIG.parent)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "uniform")
