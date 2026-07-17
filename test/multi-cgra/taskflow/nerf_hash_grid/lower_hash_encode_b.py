"""Lowers the NeRF hash_encode kernel through the B route (local only).

Reuses the reusable B bridge from the frontend unit-test driver
(test/frontend/gather/compile_gather.py): it runs torch-mlir's partial
conversion pipeline so the surrounding ops become linalg while the gather stays
opaque, then neutralizes the gather islands into builtin-typed generic ops.

This script is a local integration probe for the real NeRF kernel. It is not
part of the repository test suite.

Usage:
    python3 lower_hash_encode_b.py [output.mlir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "..", "..", "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "..", "..", "frontend", "gather"))
import neura_ops  # noqa: E402
import compile_gather as bridge  # noqa: E402

import torch  # noqa: E402
from torch_mlir.fx import export_and_import  # noqa: E402
from torch_mlir.compiler_utils import OutputType  # noqa: E402

from compile_hash_encode import HashEncodeModule  # noqa: E402


def export_nerf_module():
    """Exports the NeRF hash_encode kernel to a torch-typed MLIR module.

    Returns:
        The torch-mlir Module produced by the FX importer (RAW output).
    """
    model = HashEncodeModule().eval()
    inputs = torch.randn(4, 3).clamp(-1.0, 1.0)
    embeddings = torch.randn(model.total_params, model.level_dim) * 0.01
    return export_and_import(
        model, inputs, embeddings,
        output_type=OutputType.RAW,
        func_name="forward",
    )


def report_residuals(mlir_str):
    """Prints the residual torch constructs that still block a clean handoff.

    Args:
        mlir_str: The lowered MLIR module string.
    """
    markers = [
        "torch.operator", "torch.neura.gather", "torch.aten",
        "torch_c.", "neura.gather", "linalg.",
    ]
    for marker in markers:
        print(f"  {marker:24s}: {mlir_str.count(marker)}")


def main(output_file):
    """Runs the B route on NeRF and writes the lowered module.

    Args:
        output_file: Path to the output MLIR file.
    """
    module = export_nerf_module()
    mlir_str = bridge.build_neutral_module(module)
    with open(output_file, "w") as f:
        f.write(mlir_str)

    print("=== residual construct counts after B lowering ===")
    report_residuals(mlir_str)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "hash_encode_b.mlir"
    main(out)
