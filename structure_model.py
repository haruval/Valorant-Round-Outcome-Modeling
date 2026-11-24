"""Optional extended runtime. Import only for an approved ML execution.

The frozen models in rolling_model.py are subclassed, not edited.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from rolling_features import FEATURE_NAMES, HISTORY_SECONDS, schema
from rolling_model import PooledHistoryMLP, RollingTemporalCNN
from structure_inputs import EXTENSION_ID, EXTENSION_NAMES, extension_contract, validate_extension_policy

CONFIG_KEYS = {"architecture", "schema", "input_policy", "extension", "extension_policy"}


class ExtensionMixin:
    """Feed extension channels into the first projection without disturbing the frozen model.

    The frozen constructor runs first and draws from the global random stream exactly as
    it does under train_rolling.py. Extension weights come from a separate generator, so
    nothing after construction sees a different stream. Adding their output to the frozen
    projection equals one wider 1x1 convolution with a shared bias.

    With every extension channel masked the added term is zero and its weights receive
    zero gradient, so the model computes what the frozen model computes.
    """

    def __init__(self, input_policy, extension_policy, init_seed=0):
        super().__init__(input_policy)
        validate_extension_policy(extension_policy)
        self.extension_policy = json.loads(json.dumps(extension_policy))
        self.register_buffer("extension_mask", torch.tensor(extension_policy["keep"], dtype=torch.float32),
                             persistent=False)
        generator = torch.Generator().manual_seed(init_seed)
        bound = 1 / math.sqrt(len(FEATURE_NAMES) + len(EXTENSION_NAMES))
        weight = (torch.rand(8, len(EXTENSION_NAMES), 1, generator=generator) * 2 - 1) * bound
        self.extension = nn.Parameter(weight)

    def projected(self, values, extension):
        if (values.ndim != 3 or values.shape[1:] != (HISTORY_SECONDS, len(FEATURE_NAMES))
                or extension.shape != (values.shape[0], HISTORY_SECONDS, len(EXTENSION_NAMES))):
            raise ValueError("Expected (batch, 20 seconds, v2 channels) and (batch, 20 seconds, extension channels)")
        added = F.conv1d((extension * self.extension_mask).transpose(1, 2), self.extension)
        return torch.relu(self.project((values * self.input_mask).transpose(1, 2)) + added)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps({
            "architecture": self.architecture, "schema": schema(), "input_policy": self.input_policy,
            "extension": extension_contract(), "extension_policy": self.extension_policy}))
        save_file({name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()},
                  str(directory / "model.safetensors"))

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        config_path, weights = directory / "config.json", directory / "model.safetensors"
        if config_path.stat().st_size > 200_000 or weights.stat().st_size > 1_000_000:
            raise ValueError("Extended checkpoint exceeds size limits")
        config = json.loads(config_path.read_text())
        if (set(config) != CONFIG_KEYS or config["architecture"] != cls.architecture
                or config["schema"] != schema() or config["extension"] != extension_contract()):
            raise ValueError("Incompatible extended checkpoint")
        model = cls(config["input_policy"], config["extension_policy"])
        model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        return model.eval()


class StructureTemporalCNN(ExtensionMixin, RollingTemporalCNN):
    architecture = "rolling-tcn-v2+" + EXTENSION_ID

    def forward(self, values, extension):
        return self.head(self.temporal(self.projected(values, extension))[:, :, -1])


class StructurePooledMLP(ExtensionMixin, PooledHistoryMLP):
    architecture = "rolling-pooled-mlp-v2+" + EXTENSION_ID

    def forward(self, values, extension):
        hidden = self.temporal(self.projected(values, extension))
        return self.head(torch.cat((hidden.mean(dim=-1), hidden[:, :, -1]), dim=-1))


MODEL_CLASSES = {"temporal_cnn": StructureTemporalCNN, "pooled_mlp": StructurePooledMLP}
