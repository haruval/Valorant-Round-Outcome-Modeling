"""Local 1D CNN with JSON metadata and safetensors weights only."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from data_processing import FEATURE_NAMES, SCHEMA_VERSION
from experiment import CNN_ARCHITECTURE


class ValorantRoundCNN(nn.Module):
    def __init__(self, observation_seconds: int = 30, dropout: float = 0.1):
        super().__init__()
        if type(observation_seconds) is not int or not 8 <= observation_seconds <= 120:
            raise ValueError("CNN observation window must be an integer between 8 and 120 seconds")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        self.observation_seconds = observation_seconds
        self.dropout = float(dropout)
        layers = []
        in_channels = len(FEATURE_NAMES)
        for out_channels in (8, 16):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(16, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (self.observation_seconds, len(FEATURE_NAMES)):
            raise ValueError("Expected (batch, observation_seconds, n_features)")
        return self.classifier(self.pool(self.backbone(values.transpose(1, 2))).squeeze(-1))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        config = {"schema_version": SCHEMA_VERSION, "architecture": CNN_ARCHITECTURE,
                  "feature_names": FEATURE_NAMES, "observation_seconds": self.observation_seconds,
                  "dropout": self.dropout, "labels": ["defender_win", "attacker_win"]}
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
                  str(directory / "model.safetensors"))

    @classmethod
    def load(cls, directory: Path) -> ValorantRoundCNN:
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError("Provide an existing local model directory")
        config_path, weights_path = directory / "config.json", directory / "model.safetensors"
        if config_path.stat().st_size > 16_384 or weights_path.stat().st_size > 10_000_000:
            raise ValueError("Checkpoint exceeds the size limit for this architecture")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (config.get("schema_version") != SCHEMA_VERSION
                or config.get("architecture") != CNN_ARCHITECTURE
                or config.get("feature_names") != FEATURE_NAMES
                or config.get("labels") != ["defender_win", "attacker_win"]):
            raise ValueError("Incompatible checkpoint feature schema; retrain with this pipeline")
        model = cls(config["observation_seconds"], config["dropout"])
        model.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
        return model.eval()
