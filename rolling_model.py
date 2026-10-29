"""Optional temporal CNN runtime. Import only for an approved ML execution."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from rolling_features import FEATURE_NAMES, HISTORY_SECONDS, schema

ARCHITECTURE = "rolling-tcn-v1"


class CausalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(8, 8, 3, dilation=dilation)
        self.dropout = nn.Dropout(.1)

    def forward(self, values):
        return values + self.dropout(torch.relu(self.conv(F.pad(values, (self.padding, 0)))))


class RollingTemporalCNN(nn.Module):
    """8 hidden channels; dilations 1/2/4/8; last-state head, 31-bin receptive field."""
    def __init__(self):
        super().__init__()
        self.project = nn.Conv1d(len(FEATURE_NAMES), 8, 1)
        self.temporal = nn.Sequential(*(CausalBlock(dilation) for dilation in (1, 2, 4, 8)))
        self.head = nn.Linear(8, 2)

    def forward(self, values):
        if values.ndim != 3 or values.shape[1:] != (HISTORY_SECONDS, len(FEATURE_NAMES)):
            raise ValueError("Expected (batch, 20 seconds, rolling feature count)")
        hidden = torch.relu(self.project(values.transpose(1, 2)))
        return self.head(self.temporal(hidden)[:, :, -1])

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps({"architecture": ARCHITECTURE, "schema": schema()}))
        save_file({name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()},
                  str(directory / "model.safetensors"))

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        config_path, weights = directory / "config.json", directory / "model.safetensors"
        if config_path.stat().st_size > 100_000 or weights.stat().st_size > 1_000_000:
            raise ValueError("Rolling checkpoint exceeds size limits")
        config = json.loads(config_path.read_text())
        if config != {"architecture": ARCHITECTURE, "schema": schema()}:
            raise ValueError("Incompatible rolling checkpoint")
        model = cls()
        model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        return model.eval()
