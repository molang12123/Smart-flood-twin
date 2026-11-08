from __future__ import annotations

import math
import torch
from torch import nn


class ConditionalFloodPINN(nn.Module):
    """(x,y,t,mean rain intensity,duration) -> (water depth,u,v)."""

    def __init__(self, scales: dict, width: int = 96, layers: int = 5,
                 log_conditions: bool = False):
        super().__init__()
        self.scales = {k: float(v) for k, v in scales.items()}
        self.log_conditions = bool(log_conditions)
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
        q = (p - offset) / scale
        if self.log_conditions:
            rain_min = s["rain_min_m_s"]
            duration_min = s["duration_min_s"]
            if torch.any(p[:, 3] <= 0) or torch.any(p[:, 4] <= 0):
                raise ValueError("log-conditioned rain intensity and duration must be positive")
            q[:, 3] = ((torch.log(p[:, 3]) - math.log(rain_min)) /
                       (math.log(s["rain_m_s"]) - math.log(rain_min)))
            q[:, 4] = ((torch.log(p[:, 4]) - math.log(duration_min)) /
                       (math.log(s["duration_s"]) - math.log(duration_min)))
        q = q.to(next(self.parameters()).dtype)
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
