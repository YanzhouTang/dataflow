#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "Conversion/ConversionPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraTypes.h"

std::string filename = "opgraph.dot";
std::error_code EC;
llvm::raw_fd_ostream os(filename, EC, llvm::sys::fs::OF_Text);

// This pass pipeline can convert all the other dialects into the Neura dialect
void mlir::neura::registerNeuraConversionPassPipeline() {
  PassPipelineRegistration<>(
      "neura-conversion", "Convert all dialects to Neura dialect",
      [](OpPassManager &pm) {
        pm.addPass(mlir::neura::createAssignAcceleratorPass());
        // Convert all the other dialects into the Neura dialect
        pm.addPass(mlir::createLowerArithToNeuraPass());
        pm.addPass(mlir::createLowerLlvmToNeuraPass());
        pm.addPass(mlir::createPrintOpGraphPass(os));

        pm.addPass(mlir::neura::createCanonicalizeCastPass());
        pm.addPass(mlir::neura::createPromoteFuncArgToConstPass());
        pm.addPass(mlir::neura::createFoldConstantPass());
        pm.addPass(mlir::neura::createCanonicalizeLiveInPass());
        pm.addPass(mlir::neura::createLeveragePredicatedValuePass());
        pm.addPass(mlir::createPrintOpGraphPass(os));

        pm.addPass(mlir::neura::createTransformCtrlToDataFlowPass());
        pm.addPass(mlir::neura::createFoldConstantPass());
        pm.addPass(mlir::neura::createFusePatternPass());
        pm.addPass(mlir::neura::createInsertDataMovPass());
        pm.addPass(mlir::createPrintOpGraphPass(os));
      });
}
