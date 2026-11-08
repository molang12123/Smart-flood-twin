from __future__ import annotations

import math
import torch
from torch import nn


class ConditionalFloodPINN(nn.Module):
    """(x,y,t,mean rain intensity,duration) -> (water depth,u,v)."""

    def __init__(self, scales: dict, width: int = 96, layers: int = 5):
        super().__init__()
        self.scales = {k: float(v) for k, v in scales.items()}
        blocks = [nn.Linear(5, width), nn.Tanh()]
        for _ in range(layers - 1): blocks += [nn.Linear(width, width), nn.Tanh()]
        blocks.append(nn.Linear(width, 3)); self.net = nn.Sequential(*blocks)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, points):
        s = self.scales
        p = points.to(torch.float64)
        offset = p.new_tensor([s["x0"], s["y0"], 0., 0., 0.])
        scale = p.new_tensor([s["lx"], s["ly"], s["time_s"], s["rain_m_s"], s["duration_s"]])
        q = ((p - offset) / scale).to(next(self.parameters()).dtype)
        raw = self.net(q)
        # Analytic rainfall accumulation is the depth baseline. The network
        # learns terrain-driven redistribution as a positive multiplicative
        # factor; raw=0 starts exactly at the closed-basin water balance.
        ramp = 1. - torch.exp(-points[:, 2:3].to(raw.dtype) / s["ramp_s"])
        precise = points.to(raw.dtype)
        accumulated_rain = precise[:, 3:4] * torch.minimum(precise[:, 2:3], precise[:, 4:5])
        depth_factor = torch.nn.functional.softplus(raw[:, :1]) / math.log(2.)
        h = accumulated_rain * depth_factor
        uv = ramp * torch.tanh(raw[:, 1:]) * s["velocity_m_s"]
        return torch.cat((h, uv), dim=1)
