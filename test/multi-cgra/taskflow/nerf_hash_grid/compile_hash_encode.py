"""Compiles hash_encode to Torch Dialect MLIR with neura.gather preserved.

Exports the hash_encode kernel through torch-mlir's FX importer. The custom
op neura::gather (registered in python/neura_ops.py via torch.library.custom_op)
stays opaque through torch.export, so it appears in the output as
``torch.operator "torch.neura.gather"`` instead of being decomposed back into
aten.index.

The RAW output type is used so that torch-mlir skips its backend lowering
pipeline (which does not accept the custom op). Because torch.export already
applies decomposition and functionalization, the RAW output is a single clean
SSA module suitable for downstream inspection and lowering.

Usage:
    python3 compile_hash_encode.py [output.mlir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python"))
import neura_ops  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch_mlir.fx import export_and_import  # noqa: E402
from torch_mlir.compiler_utils import OutputType  # noqa: E402

from nerf_kernels import hash_encode  # noqa: E402


class HashEncodeModule(nn.Module):
    """Wraps hash_encode with embeddings passed as a function argument.

    Keeping embeddings as an argument (rather than an nn.Parameter) avoids
    global-slot handling during export. Offsets are compile-time constants
    stored as a Python list, which keeps the per-level slice bounds static.
    """

    def __init__(self, num_levels=2, level_dim=2, base_resolution=16,
                 per_level_scale=2.0, log2_hashmap_size=8, bound=1.0):
        super().__init__()
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale
        self.log2_hashmap_size = log2_hashmap_size
        self.bound = bound

        # Precomputes per-level offsets as a Python list of ints.
        import numpy as np
        scale_log2 = np.log2(per_level_scale)
        max_params = 2 ** log2_hashmap_size
        offsets = [0]
        for level in range(num_levels):
            scale = np.exp2(level * scale_log2) * base_resolution - 1.0
            resolution = int(np.ceil(scale)) + 1
            n_dense = (resolution + 1) ** 3
            n_params = min(n_dense, max_params)
            offsets.append(offsets[-1] + n_params)
        self.offsets_list = offsets
        self.total_params = offsets[-1]

    def forward(self, inputs, embeddings):
        """Delegates to nerf_kernels.hash_encode.

        Args:
            inputs: Coordinates of shape [N, 3], float32, in [-bound, bound].
            embeddings: Embedding table of shape [total_params, C], float32.

        Returns:
            Encoded features of shape [N, num_levels * level_dim], float32.
        """
        return hash_encode(
            inputs, embeddings, self.offsets_list, self.bound,
            num_levels=self.num_levels,
            level_dim=self.level_dim,
            base_resolution=self.base_resolution,
            per_level_scale=self.per_level_scale,
            log2_hashmap_size=self.log2_hashmap_size,
        )


def compile_hash_encode(output_file):
    """Compiles hash_encode and writes Torch Dialect MLIR to disk.

    Args:
        output_file: Path to the output MLIR file.

    Returns:
        The MLIR module string.
    """
    model = HashEncodeModule().eval()

    num_points = 4
    inputs = torch.randn(num_points, 3).clamp(-1.0, 1.0)
    embeddings = torch.randn(model.total_params, model.level_dim) * 0.01

    module = export_and_import(
        model, inputs, embeddings,
        output_type=OutputType.RAW,
        func_name="forward",
    )
    mlir_str = str(module)

    with open(output_file, "w") as f:
        f.write(mlir_str)

    return mlir_str


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "hash_encode_torch.mlir"
    compile_hash_encode(out)
