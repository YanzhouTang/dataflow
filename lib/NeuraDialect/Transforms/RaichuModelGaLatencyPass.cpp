#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

#define GEN_PASS_DEF_RAICHUMODELGALATENCY
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

// Models the execution latency of the RAICHU M-CGRA Gather/Scatter (GA) unit
// (paper Section 4.1). The GA unit issues ga_width random-address requests per
// cycle through the shared-SRAM crossbar, so a batched gather/scatter of B rows
// occupies the GA unit for ceil(B / ga_width) cycles. This latency is recorded
// on the op's "latency" attribute, which the mapper consumes to reserve the GA
// tile for the right number of cycles when computing the II.
//
// A single batched gather therefore remains one operation (not B loads) while
// still paying a memory-proportional cost, which is exactly the "8 loads -> 1
// gather" tradeoff the M-CGRA is designed to exploit.

// Extracts the batch (row) count from a gather/scatter result or operand type.
// Handles plain shaped types (memref/tensor) directly; returns 1 when the batch
// size cannot be determined (degenerate scalar gather == a single load).
static int64_t getBatchRows(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    if (shaped.hasRank() && shaped.getRank() >= 1 &&
        !shaped.isDynamicDim(0)) {
      return shaped.getDimSize(0);
    }
  }
  return 1;
}

struct RaichuModelGaLatencyPass
    : public PassWrapper<RaichuModelGaLatencyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RaichuModelGaLatencyPass)

  RaichuModelGaLatencyPass() = default;
  RaichuModelGaLatencyPass(const RaichuModelGaLatencyPass &pass)
      : PassWrapper<RaichuModelGaLatencyPass, OperationPass<ModuleOp>>(pass) {}

  Option<int> gaWidth{
      *this, "ga-width",
      llvm::cl::desc("Number of random-address requests the GA unit issues per "
                     "cycle (M-CGRA shared-SRAM crossbar width)."),
      llvm::cl::init(8)};

  StringRef getArgument() const override { return "raichu-model-ga-latency"; }
  StringRef getDescription() const override {
    return "Sets gather/scatter execution latency from the M-CGRA GA unit "
           "width.";
  }

  void setGaLatency(Operation *op, int64_t batch_rows) {
    int width = gaWidth > 0 ? gaWidth.getValue() : 1;
    int64_t latency = (batch_rows + width - 1) / width; // ceil(batch / width)
    if (latency < 1) {
      latency = 1;
    }
    op->setAttr("latency", IntegerAttr::get(IntegerType::get(op->getContext(),
                                                             32),
                                            static_cast<int>(latency)));
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](neura::GatherOp gather) {
      setGaLatency(gather, getBatchRows(gather.getResult().getType()));
    });
    module.walk([&](neura::ScatterOp scatter) {
      // Scatter writes as many rows as its index/value batch carries.
      setGaLatency(scatter, getBatchRows(scatter.getValues().getType()));
    });
  }
};

} // namespace

namespace mlir {
namespace neura {
std::unique_ptr<Pass> createRaichuModelGaLatencyPass() {
  return std::make_unique<RaichuModelGaLatencyPass>();
}
} // namespace neura
} // namespace mlir
