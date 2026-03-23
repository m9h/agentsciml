"""Parameter Golf project adapter.

Bridges the OpenAI parameter-golf competition with AgenticSciML.
Evolves train_gpt.py variants to minimize bits-per-byte on FineWeb.
"""

from __future__ import annotations

import re
from pathlib import Path

from .base import ProjectAdapter


class ParameterGolfAdapter(ProjectAdapter):
    """Adapter for the OpenAI parameter-golf competition."""

    ARTIFACT_LIMIT_BYTES = 16_000_000  # 16 MB decimal
    WALL_CLOCK_LIMIT_SEC = 600  # 10 minutes on 8xH100

    def __init__(self, project_root: Path | None = None) -> None:
        root = project_root or Path.home() / "dev" / "parameter-golf"
        super().__init__(root)

    def get_context(self) -> str:
        readme = self.project_root / "README.md"
        if readme.exists():
            return readme.read_text()[:4000]
        return (
            "OpenAI Parameter Golf: train the best possible language model "
            "within a 16 MB artifact size (INT8 + zlib) and 10-minute wall-clock "
            "training time on 8x H100 GPUs. Metric: bits-per-byte (BPB) on "
            "FineWeb validation set (first 50,000 documents). Lower is better."
        )

    def get_results_history(self) -> str:
        if self.results_path.exists():
            return self.results_path.read_text()
        return ""

    def get_current_experiment(self) -> str:
        train_script = self.project_root / "train_gpt.py"
        if train_script.exists():
            return train_script.read_text()
        if self.experiment_path.exists():
            return self.experiment_path.read_text()
        return ""

    def get_available_api(self) -> str:
        return """\
This is a self-contained training script (train_gpt.py). Key components:

Architecture (modify freely):
    - Transformer layers, dimensions, attention heads, KV heads
    - Grouped Query Attention (GQA), depth recurrence, parameter tying
    - Activation functions, normalization, positional encoding

Tokenizer (modify freely):
    - SentencePiece variants: sp1024, sp4096, byte260
    - Custom tokenizers (e.g., BigramHash)
    - Vocabulary size affects both model size and BPB calculation

Optimizer (modify freely):
    - Muon (weight matrices) + Adam (embeddings/scalars) is the baseline
    - Learning rate schedules, weight decay, gradient clipping

Quantization & compression:
    - Final artifact = INT8 quantized weights + zlib level-9 compression
    - Custom quantization (INT4/INT5/INT6) allowed if roundtrip-validated
    - Quantization-aware training (QAT) encouraged

Evaluation:
    - val_bpb = cross_entropy_loss / ln(2) / bytes_per_token
    - Computed on FineWeb validation set (first 50,000 documents)
    - Must match regardless of tokenizer choice (tokenizer-agnostic metric)

Output format:
    Script must print validation BPB during training.
    Final line must contain: val_bpb=<score>

Hard constraints:
    - Compressed artifact <= 16,000,000 bytes
    - Training wall-clock <= 600 seconds on 8x H100
    - No network access during evaluation
    - Fully reproducible (deterministic seeds)
"""

    def get_metric_name(self) -> str:
        return "val_bpb"

    def get_result_metric_key(self) -> str:
        return "val_bpb"

    def get_score_direction(self) -> str:
        return "minimize"

    def get_constraints(self) -> str:
        return (
            "HARD CONSTRAINTS — proposals violating these are invalid:\n"
            "1. Compressed artifact (INT8 + zlib level 9) must be <= 16,000,000 bytes.\n"
            "   Current baseline is ~15.86 MB. Adding layers/dimensions/vocab "
            "   increases artifact size.\n"
            "2. Training must complete in <= 600 seconds on 8x NVIDIA H100.\n"
            "3. No external data or network access during training or evaluation.\n"
            "4. Results must be reproducible across runs.\n"
            "5. No 'paid prefix' — do not compress validation data into the artifact.\n"
            "6. val_bpb calculation must be correct if tokenizer is changed "
            "   (bytes_per_token must reflect actual UTF-8 bytes).\n"
        )

    def parse_score(self, result_lines: list[str]) -> float:
        """Extract best val_bpb from output lines (lower is better)."""
        best = float("inf")
        for line in result_lines:
            # Match val_bpb=X.XXXX in RESULT| lines or plain output
            for match in re.finditer(r"val_bpb[=:\s]+([0-9]+\.?[0-9]*)", line):
                try:
                    val = float(match.group(1))
                    best = min(best, val)
                except ValueError:
                    continue
        return best if best < float("inf") else 0.0

    @property
    def experiment_path(self) -> Path:
        """Parameter golf uses train_gpt.py, not autoresearch/experiment.py."""
        train = self.project_root / "train_gpt.py"
        if train.exists():
            return train
        return self.project_root / "autoresearch" / "experiment.py"
