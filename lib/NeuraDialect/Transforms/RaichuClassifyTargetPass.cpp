#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

#define GEN_PASS_DEF_RAICHUCLASSIFYTARGET
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

// Name-based classifier that routes RAICHU rendering kernels onto either the
// Memory CGRA (M-CGRA) or the Compute CGRA (C-CGRA).
//
// This is intentionally a simple substring match on the kernel/function name
// rather than an analysis of the kernel body. The RAICHU paper describes an
// operator classification stage (Section 4.2) that inspects algorithmic
// properties; for the initial NeRF bring-up we hardcode the mapping:
//
//   ray marching  -> M-CGRA (irregular sampling / memory bound)
//   hash encoding -> M-CGRA (hash-table gather, memory bound)
//   MLP           -> C-CGRA (dense arithmetic, compute bound)
//
// Pixel composition (blending) is deliberately left unclassified for now; it
// will be handled when the 3DGS pipeline is added.

// Case-insensitive substring test.
static bool nameContains(StringRef name, StringRef needle) {
  return name.contains_insensitive(needle);
}

static StringRef classifyByName(StringRef name) {
  // Memory-bound rendering stages (irregular gather/scatter / data
  // reorganization) -> M-CGRA. Covers all five RAICHU pipelines:
  //   Hash NeRF   : hash-grid gather
  //   L-NeRF      : factor gather (coherent embedding generation)
  //   3DGS        : sort partition / compare-swap, tile binning (job lists)
  //   Ray Tracing : hit-node stream compaction (BVH traversal reorg)
  // Keywords are matched case-insensitively as substrings, so they must be
  // underscore-free to also match camelCase kernel names (e.g. "compact"
  // matches rtCompact, "partition" matches sortPartition).
  static constexpr StringRef kMCgraKeywords[] = {
      "raymarch", "marchrays", "raysampl", "sampleposition",
      "hashencode", "hashgrid", "gather", "scatter",
      "compact", "partition", "binning", "coverage", "tilerange"};
  for (StringRef keyword : kMCgraKeywords) {
    if (nameContains(name, keyword)) {
      return mlir::accel::kMCgraTarget;
    }
  }

  // Compute-bound rendering stages (dense fused arithmetic) -> C-CGRA. Covers:
  //   NeRF MLP / positional encoding / accumulation
  //   L-NeRF   : VM product/reduce, appearance projection, MLP decode
  //   3DGS     : covariance, conic, radius, SH color, blending, intersection
  //   Ray Tracing : ray gen, AABB slab test, Moller-Trumbore, shading
  static constexpr StringRef kCCgraKeywords[] = {
      "mlp",       "posenc",    "sample",    "accum",     "vmproduct",
      "vmreduce",  "appear",    "reduce",    "sigma",     "conic",
      "radius",    "shcolor",   "preprocess","blend",     "alpha",
      "composite", "frustum",   "raygen",    "aabb",      "triangle",
      "shade",     "intersect", "subtile",   "rotm",      "cov"};
  for (StringRef keyword : kCCgraKeywords) {
    if (nameContains(name, keyword)) {
      return mlir::accel::kCCgraTarget;
    }
  }

  // Unclassified: left untouched.
  return StringRef();
}

struct RaichuClassifyTargetPass
    : public PassWrapper<RaichuClassifyTargetPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RaichuClassifyTargetPass)

  StringRef getArgument() const override { return "raichu-classify-target"; }
  StringRef getDescription() const override {
    return "Classifies rendering kernels onto the RAICHU M-CGRA or C-CGRA by "
           "name.";
  }

  void assignIfUnset(Operation *op, StringRef name, Builder &builder) {
    if (op->hasAttr(mlir::accel::kRaichuTargetAttr)) {
      return;
    }
    StringRef target = classifyByName(name);
    if (target.empty()) {
      return;
    }
    op->setAttr(mlir::accel::kRaichuTargetAttr, builder.getStringAttr(target));
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(&getContext());

    // Classifies neura.kernel ops by their symbol/kernel name if present,
    // otherwise by the name of the enclosing function.
    module.walk([&](neura::KernelOp kernel_op) {
      StringRef name;
      if (auto sym = kernel_op->getAttrOfType<StringAttr>("sym_name")) {
        name = sym.getValue();
      } else if (auto parent =
                     kernel_op->getParentOfType<FunctionOpInterface>()) {
        name = parent.getName();
      }
      assignIfUnset(kernel_op, name, builder);
    });

    // Classifies functions by name.
    module.walk([&](Operation *op) {
      if (auto func = dyn_cast<FunctionOpInterface>(op)) {
        if (func.isExternal()) {
          return;
        }
        assignIfUnset(func, func.getName(), builder);
      }
    });
  }
};

} // namespace

namespace mlir {
namespace neura {
std::unique_ptr<Pass> createRaichuClassifyTargetPass() {
  return std::make_unique<RaichuClassifyTargetPass>();
}
} // namespace neura
} // namespace mlir
