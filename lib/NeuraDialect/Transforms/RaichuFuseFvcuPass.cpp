#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_DEF_RAICHUFUSEFVCU
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

// C-CGRA operation fusion for the Fused Vector-Compare Unit (RAICHU paper,
// Section 4.2.2 / Figure 6c). Scalar multiply-add chains that form short dot
// products are collapsed into a single FVCU primitive so that the mapper sees
// one dense op instead of a spread-out scalar tree.
//
// This pattern recognizes a vector dot product expressed as a vector multiply
// followed by a horizontal reduction:
//
//   %p = neura.vfmul %a, %b               // elementwise a * b
//   %d = neura.vector.reduce.add %p       // sum of lanes
//
// and rewrites it to a single FVCU primitive:
//
//   %d = neura.fvc_dot3 %a, %b
//
// The FVCU is fed by vector loads, so the fused op keeps a two-operand vector
// fan-in that matches both the hardware datapath and the PE input-port budget.
struct FuseFVCDot3Pattern : public OpRewritePattern<neura::VectorReduceAddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::VectorReduceAddOp reduce,
                                PatternRewriter &rewriter) const override {
    auto vfmul = reduce.getInput().getDefiningOp<neura::VFMulOp>();
    if (!vfmul) {
      return failure();
    }

    // The vector product must feed only this reduction so that erasing it is
    // safe.
    if (!vfmul->hasOneUse()) {
      return failure();
    }

    auto dot3 = rewriter.create<neura::FVCDot3Op>(
        reduce.getLoc(), reduce.getType(), vfmul.getLhs(), vfmul.getRhs());

    rewriter.replaceOp(reduce, dot3.getResult());
    rewriter.eraseOp(vfmul);
    return success();
  }
};

// FVCU gate mode G3 (AABB slab intersection test). The ray-box test reduces the
// per-axis entry/exit distances with a nested min/max tree:
//   tnear = max(max(a,b),c)   tfar = min(min(a,b),c)
// The FVCU compare tree does each 3-way reduction in one op, so we fuse the
// nested neura.fmax / neura.fmin (both in value-value form) into fvc_max3 /
// fvc_min3. This is the intersection-test acceleration the FVCU provides.
struct FuseFVCMax3Pattern : public OpRewritePattern<neura::FMaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::FMaxOp outer,
                                PatternRewriter &rewriter) const override {
    Value c = outer.getRhs();
    if (!c) {
      return failure(); // needs the binary (value-value) form
    }
    auto inner = outer.getLhs().getDefiningOp<neura::FMaxOp>();
    if (!inner || !inner.getRhs() || !inner->hasOneUse()) {
      return failure();
    }
    auto max3 = rewriter.create<neura::FVCMax3Op>(
        outer.getLoc(), outer.getType(), inner.getLhs(), inner.getRhs(), c);
    rewriter.replaceOp(outer, max3.getResult());
    rewriter.eraseOp(inner);
    return success();
  }
};

struct FuseFVCMin3Pattern : public OpRewritePattern<neura::FMinOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::FMinOp outer,
                                PatternRewriter &rewriter) const override {
    Value c = outer.getRhs();
    if (!c) {
      return failure();
    }
    auto inner = outer.getLhs().getDefiningOp<neura::FMinOp>();
    if (!inner || !inner.getRhs() || !inner->hasOneUse()) {
      return failure();
    }
    auto min3 = rewriter.create<neura::FVCMin3Op>(
        outer.getLoc(), outer.getType(), inner.getLhs(), inner.getRhs(), c);
    rewriter.replaceOp(outer, min3.getResult());
    rewriter.eraseOp(inner);
    return success();
  }
};

// FVCU gate mode G4 (pairwise compare-and-swap). A min/max pair over the same
// two operands -- e.g. the per-axis slab bounds `tmin = fmin(t1,t2)`,
// `tmax = fmax(t1,t2)` of the AABB test, or an ordering step in 3DGS sorting --
// is one FVCU compare-swap producing (lo, hi). Rooted on the FMinOp; the
// matching FMaxOp over the same (a,b) is replaced by the same fused op.
struct FuseFVCCmpSwapPattern : public OpRewritePattern<neura::FMinOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::FMinOp fmin,
                                PatternRewriter &rewriter) const override {
    Value a = fmin.getLhs();
    Value b = fmin.getRhs();
    if (!b) {
      return failure();
    }
    // Find an fmax over the same operand pair (either order).
    neura::FMaxOp fmax = nullptr;
    for (Operation *user : a.getUsers()) {
      auto m = dyn_cast<neura::FMaxOp>(user);
      if (m && m.getRhs() &&
          ((m.getLhs() == a && m.getRhs() == b) ||
           (m.getLhs() == b && m.getRhs() == a))) {
        fmax = m;
        break;
      }
    }
    if (!fmax) {
      return failure();
    }
    auto cs = rewriter.create<neura::FVCCmpSwapOp>(
        fmin.getLoc(), TypeRange{fmin.getType(), fmax.getType()}, a, b);
    rewriter.replaceOp(fmax, cs.getHi());
    rewriter.replaceOp(fmin, cs.getLo());
    return success();
  }
};

// FVCU gate mode G2 (six-lane reduce). A six-value sum a+b+c+d+e+f collapses
// into one FVCU reduce6 -- the reduce stage that completes dot products in NeRF
// / low-rank NeRF. After `fuse-pattern` a left-associative 6-sum appears as a
// spine of fadd_fadd accumulators over a terminal fadd:
//   %s0 = fadd(l0, l1)
//   %s1 = fadd_fadd(%s0, l2, l3)
//   %s2 = fadd_fadd(%s1, l4, l5)   <- root
// so we walk operand-0 (the accumulator) collecting the six leaves.
struct FuseFVCReduce6Pattern : public OpRewritePattern<neura::FAddFAddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::FAddFAddOp root,
                                PatternRewriter &rewriter) const override {
    // Must be the top of the reduction (not feeding another add accumulator).
    for (Operation *user : root.getResult().getUsers()) {
      if (isa<neura::FAddOp, neura::FAddFAddOp>(user)) {
        return failure();
      }
    }
    SmallVector<Value> leaves;
    SmallVector<Operation *> inner;
    Operation *cur = root;
    while (cur && leaves.size() < 6) {
      if (cur != root && !cur->hasOneUse()) {
        return failure();
      }
      if (auto ff = dyn_cast<neura::FAddFAddOp>(cur)) {
        leaves.push_back(ff.getC());
        leaves.push_back(ff.getB());
        if (cur != root) {
          inner.push_back(cur);
        }
        cur = ff.getA().getDefiningOp();
      } else if (auto fa = dyn_cast<neura::FAddOp>(cur)) {
        if (!fa.getRhs()) {
          return failure();
        }
        leaves.push_back(fa.getRhs());
        leaves.push_back(fa.getLhs());
        inner.push_back(cur);
        cur = nullptr; // terminal add: both operands are leaves
      } else {
        return failure();
      }
    }
    if (leaves.size() != 6 || cur != nullptr) {
      return failure();
    }
    auto r6 = rewriter.create<neura::FVCReduce6Op>(root.getLoc(),
                                                   root.getType(), leaves);
    rewriter.replaceOp(root, r6.getResult());
    for (Operation *add : inner) {
      if (add->use_empty()) {
        rewriter.eraseOp(add);
      }
    }
    return success();
  }
};

// FVCU gate mode G1 (six-lane partial dot / MAC). A six-element multiply-
// accumulate chain Σ a_i·b_i -- after `fuse-pattern` a spine of fmul_fadd
// accumulators over a terminal fmul:
//   %p0 = fmul(a0, b0)
//   %p1 = fmul_fadd(a1, b1, %p0)
//   ...
//   %p5 = fmul_fadd(a5, b5, %p4)   <- root (result then used by +bias etc.)
// collapses into one neura.fvc_partial_dot6 over the interleaved factor pairs.
// This is the NeRF / low-rank NeRF matrix-vector dot the FVCU G1 targets.
struct FuseFVCPartialDot6Pattern : public OpRewritePattern<neura::FMulFAddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::FMulFAddOp root,
                                PatternRewriter &rewriter) const override {
    // Top of the MAC chain: root's result must not be the accumulator (c) of
    // another fmul_fadd.
    for (Operation *user : root.getResult().getUsers()) {
      if (auto ff = dyn_cast<neura::FMulFAddOp>(user)) {
        if (ff.getC() == root.getResult()) {
          return failure();
        }
      }
    }
    SmallVector<Value> factors; // [a0,b0,a1,b1,...]
    SmallVector<Operation *> inner;
    Operation *cur = root;
    while (cur) {
      if (auto ff = dyn_cast<neura::FMulFAddOp>(cur)) {
        if (cur != root && !cur->hasOneUse()) {
          return failure();
        }
        factors.push_back(ff.getA());
        factors.push_back(ff.getB());
        if (cur != root) {
          inner.push_back(cur);
        }
        cur = ff.getC().getDefiningOp();
        if (!cur) {
          return failure();
        }
      } else if (auto fm = dyn_cast<neura::FMulOp>(cur)) {
        if (!fm.getRhs() || !cur->hasOneUse()) {
          return failure();
        }
        factors.push_back(fm.getLhs());
        factors.push_back(fm.getRhs());
        inner.push_back(cur);
        cur = nullptr; // terminal product
      } else {
        return failure();
      }
      if (factors.size() > 12) {
        return failure();
      }
    }
    if (factors.size() != 12) {
      return failure(); // exactly six products (6-lane partial dot)
    }
    auto pd6 = rewriter.create<neura::FVCPartialDot6Op>(root.getLoc(),
                                                        root.getType(), factors);
    rewriter.replaceOp(root, pd6.getResult());
    for (Operation *op : inner) {
      if (op->use_empty()) {
        rewriter.eraseOp(op);
      }
    }
    return success();
  }
};

struct RaichuFuseFvcuPass
    : public PassWrapper<RaichuFuseFvcuPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RaichuFuseFvcuPass)

  StringRef getArgument() const override { return "raichu-fuse-fvcu"; }
  StringRef getDescription() const override {
    return "Fuses scalar arithmetic motifs into RAICHU C-CGRA FVCU primitives.";
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseFVCDot3Pattern>(&getContext());
    patterns.add<FuseFVCMax3Pattern>(&getContext());
    patterns.add<FuseFVCMin3Pattern>(&getContext());
    patterns.add<FuseFVCCmpSwapPattern>(&getContext());
    patterns.add<FuseFVCReduce6Pattern>(&getContext());
    // NOTE: the scalar 6-MAC -> fvc_partial_dot6 fusion (FuseFVCPartialDot6Pattern)
    // is intentionally NOT registered: a 12-scalar-port op exceeds the PE input
    // budget and does not map. G1 (six-element MAC) is realized through the
    // mapping-efficient vector-load-fed FVCU dot path (2 vector operands),
    // matching the paper's "FVCU supported by vector loads" (see the
    // vector_load + FVCU dot demonstration in hash_nerf).
    FrozenRewritePatternSet frozen(std::move(patterns));

    module_op.walk([&](Operation *op) {
      if (!op->getRegions().empty()) {
        for (Region &region : op->getRegions()) {
          if (failed(applyPatternsGreedily(region, frozen))) {
            signalPassFailure();
          }
        }
      }
    });
  }
};

} // namespace

namespace mlir {
namespace neura {
std::unique_ptr<Pass> createRaichuFuseFvcuPass() {
  return std::make_unique<RaichuFuseFvcuPass>();
}
} // namespace neura
} // namespace mlir
