// Minimal M-CGRA mapping smoke test: a straight-line kernel containing a
// gather primitive should map onto the RAICHU M-CGRA and yield a compiled_ii.
//
// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --architecture-spec=%S/../arch_spec/raichu_m_cgra.yaml \
// RUN:   -o %t-mapped.mlir
// RUN: FileCheck %s --input-file=%t-mapped.mlir

// CHECK: compiled_ii
func.func @gather_kernel() -> i64 {
  %table = llvm.mlir.constant(0 : i64) : i64
  %idx = llvm.mlir.constant(1 : i64) : i64
  %r = neura.gather %table[%idx] : i64, i64 -> i64
  return %r : i64
}
