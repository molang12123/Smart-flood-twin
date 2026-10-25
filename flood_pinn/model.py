"""The original Tanh/Softplus MLP with explicit, checkpointed physical scales."""
from __future__ import annotations

import math
from numbers import Real
import torch
from torch import nn


SCALE_KEYS = (
    "x_origin", "y_origin", "x_scale", "y_scale", "t_scale", "r_scale",
    "duration_scale", "h_scale", "u_scale",
)


class ParametricPINN(nn.Module):
    """Map physical (x,y,t,R_m/s,duration_s) points to physical (h,u,v).

    ``parameterized=False`` preserves the original three-input architecture.
    The public point matrix still has five columns, keeping the physics API
    identical. A single-scenario training configuration is required in that mode.
    """

    def __init__(self, scales: dict, width: int = 128, layers: int = 6,
                 parameterized: bool = True):
        super().__init__()
        if isinstance(width, bool) or isinstance(layers, bool) or not isinstance(width, int) or not isinstance(layers, int) or width < 1 or layers < 1:
            raise ValueError("width and layers must be positive integers")
        if not isinstance(parameterized, bool):
            raise ValueError("parameterized must be boolean")
        missing = set(SCALE_KEYS) - set(scales)
        if missing:
            raise ValueError(f"missing physical scales: {sorted(missing)}")
        if any(isinstance(scales[name], bool) or not isinstance(scales[name], Real) for name in SCALE_KEYS):
            raise ValueError("physical scales/origins must be real numbers, not booleans/strings")
        self.scales = {name: float(scales[name]) for name in SCALE_KEYS}
        if any(not math.isfinite(value) for value in self.scales.values()):
            raise ValueError("all physical scales/origins must be finite")
        if any(self.scales[k] <= 0 for k in SCALE_KEYS if not k.endswith("origin")):
            raise ValueError("physical normalization scales must be strictly positive")
        self.width, self.layers, self.parameterized = width, layers, parameterized
        modules = [nn.Linear(5 if parameterized else 3, width), nn.Tanh()]
        for _ in range(layers - 1):
            modules.extend((nn.Linear(width, width), nn.Tanh()))
        modules.append(nn.Linear(width, 3))
        self.net = nn.Sequential(*modules)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def normalize(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 2 or points.shape[1] != 5 or not points.is_floating_point():
            raise ValueError("points must be a floating Nx5 tensor: x,y,t,R_m/s,duration_s")
        s = self.scales
        # Callers must retain original GIS coordinates as float64; converting
        # already-rounded float32 coordinates cannot recover lost centimetres.
        precise = points.to(dtype=torch.float64)
        offset = precise.new_tensor([s["x_origin"], s["y_origin"], 0., 0., 0.])
        scale = precise.new_tensor([s["x_scale"], s["y_scale"], s["t_scale"], s["r_scale"], s["duration_scale"]])
        normalized = (precise - offset) / scale
        return normalized if self.parameterized else normalized[:, :3]

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize(points).to(dtype=self.net[0].weight.dtype)
        raw = self.net(normalized)
        h = nn.functional.softplus(raw[:, :1]) * self.scales["h_scale"]
        velocities = raw[:, 1:] * self.scales["u_scale"]
        return torch.cat((h, velocities), dim=1)

    def architecture(self) -> dict:
        return {"width": self.width, "layers": self.layers, "parameterized": self.parameterized}

    def initialize_from_single(self, single: "ParametricPINN") -> None:
        """Copy a matching three-input model; newly conditional columns start at zero.

        This preserves the initial forward result, not multi-scenario validity.
        """
        if not self.parameterized or single.parameterized:
            raise ValueError("requires a parameterized target and single-scenario source")
        if self.width != single.width or self.layers != single.layers or self.scales != single.scales:
            raise ValueError("source architecture and scales must match")
        source = single.state_dict()
        target = self.state_dict()
        for key, value in source.items():
            if key == "net.0.weight":
                target[key].zero_()
                target[key][:, :3].copy_(value)
            else:
                target[key].copy_(value)
        self.load_state_dict(target)
