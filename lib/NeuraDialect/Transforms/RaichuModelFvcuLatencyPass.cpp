#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

#define GEN_PASS_DEF_RAICHUMODELFVCULATENCY
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

// Models the execution latency of the RAICHU C-CGRA Fused Vector-Compare Unit
// (FVCU, paper Section 4.4). The FVCU has a fixed number of FP32 FMA lanes
// (fvcu_width, default 6). A vector dot product / partial-dot of width K
// therefore occupies the FVCU for ceil(K / fvcu_width) cycles. This latency is
// written to the op's "latency" attribute, which the mapper consumes to reserve
// the PE for the right number of cycles when computing the II.
//
// This is the compute-side counterpart of raichu-model-ga-latency (which models
// the M-CGRA GA unit width). Without it, a wide fvc_dot3 would be scheduled as
// a single-cycle op, drastically overstating the C-CGRA's throughput.

// Extracts the vector width (element count) from a shaped operand type; returns
// 1 for scalars / unknown shapes.
static int64_t getVectorWidth(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    if (shaped.hasRank() && shaped.getNumElements() > 0) {
      return shaped.getNumElements();
    }
  }
  return 1;
}

struct RaichuModelFvcuLatencyPass
    : public PassWrapper<RaichuModelFvcuLatencyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RaichuModelFvcuLatencyPass)

  RaichuModelFvcuLatencyPass() = default;
  RaichuModelFvcuLatencyPass(const RaichuModelFvcuLatencyPass &pass)
      : PassWrapper<RaichuModelFvcuLatencyPass, OperationPass<ModuleOp>>(pass) {}

  Option<int> fvcuWidth{
      *this, "fvcu-width",
      llvm::cl::desc("Number of FP32 FMA lanes in the FVCU (C-CGRA PE)."),
      llvm::cl::init(6)};

  Option<int> memWidth{
      *this, "mem-width",
      llvm::cl::desc("Vector-load memory port width (elements per cycle)."),
      llvm::cl::init(8)};

  StringRef getArgument() const override { return "raichu-model-fvcu-latency"; }
  StringRef getDescription() const override {
    return "Sets FVCU primitive execution latency from the FVCU lane width.";
  }

  void setLatencyByWidth(Operation *op, int64_t width, int lanes) {
    if (lanes <= 0) {
      lanes = 1;
    }
    int64_t latency = (width + lanes - 1) / lanes; // ceil(width / lanes)
    if (latency < 1) {
      latency = 1;
    }
    op->setAttr("latency", IntegerAttr::get(IntegerType::get(op->getContext(),
                                                             32),
                                            static_cast<int>(latency)));
  }

  void setFvcuLatency(Operation *op, int64_t width) {
    setLatencyByWidth(op, width, fvcuWidth.getValue());
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // fvc_dot3: width taken from the first (vector) operand.
    module.walk([&](neura::FVCDot3Op dot) {
      setFvcuLatency(dot, getVectorWidth(dot.getA().getType()));
    });
    // fvc_partial_dot6: width = number of multiply-accumulate lanes = pairs.
    module.walk([&](neura::FVCPartialDot6Op p) {
      int64_t pairs = p.getOperands().size() / 2;
      setFvcuLatency(p, pairs > 0 ? pairs : 1);
    });
    module.walk([&](neura::FVCReduce6Op r) {
      // Width = vector lanes of a single vector operand, or the number of
      // scalar lane operands when fused from scalar reductions.
      int64_t width = r.getInputs().size();
      if (width == 1) {
        width = getVectorWidth(r.getInputs()[0].getType());
      }
      setFvcuLatency(r, width);
    });
    // Vector loads pay ceil(K / mem_width) memory cycles.
    module.walk([&](neura::VectorLoadOp v) {
      setLatencyByWidth(v, getVectorWidth(v.getResult().getType()),
                        memWidth.getValue());
    });
  }
};

} // namespace

namespace mlir {
namespace neura {
std::unique_ptr<Pass> createRaichuModelFvcuLatencyPass() {
  return std::make_unique<RaichuModelFvcuLatencyPass>();
}
} // namespace neura
} // namespace mlir
