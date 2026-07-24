#ifndef COMMON_ACCELERATOR_ATTRS_H
#define COMMON_ACCELERATOR_ATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace accel {

// Common attribute key.
constexpr llvm::StringRef kAcceleratorAttr = "accelerator";

// Common accelerator targets.
constexpr llvm::StringRef kNeuraTarget = "neura";
constexpr llvm::StringRef kGpuTarget = "gpu";
constexpr llvm::StringRef kTpuTarget = "tpu";

// RAICHU heterogeneous dual-tier CGRA targets.
//
// A kernel is classified onto one of the two RAICHU fabric tiers. The
// Memory CGRA (M-CGRA) handles memory-bound, irregular-access rendering
// stages (ray marching, hash-grid encoding), while the Compute CGRA
// (C-CGRA) handles compute-bound arithmetic stages (MLP). This decision is
// orthogonal to kAcceleratorAttr (which stays "neura" so the shared mapping
// pipeline still applies); it only records which fabric spec the kernel is
// mapped onto when deriving its II.
constexpr llvm::StringRef kRaichuTargetAttr = "raichu.target";
constexpr llvm::StringRef kMCgraTarget = "m-cgra";
constexpr llvm::StringRef kCCgraTarget = "c-cgra";

} // namespace accel
} // namespace mlir

#endif // COMMON_ACCELERATOR_ATTRS_H
