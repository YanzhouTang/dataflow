// RUN: neura-interpreter %s --verbose | FileCheck %s

func.func @test_sel_with_comparison() -> f32 {
  %a = arith.constant 5.0 : f32
  %b = arith.constant 3.0 : f32
  %cond = "neura.fcmp"(%a, %b) {cmpType = "gt"} : (f32, f32) -> i1

  %true_val = arith.constant 100.0 : f32
  %false_val = arith.constant 200.0 : f32
  %res = "neura.sel"(%cond, %true_val, %false_val) : (i1, f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 100.000000

  return %res : f32
}

func.func @test_sel_with_comparison_false() -> f32 {
  %a = arith.constant 2.0 : f32
  %b = arith.constant 3.0 : f32
  %cond = "neura.fcmp"(%a, %b) {cmpType = "gt"} : (f32, f32) -> i1

  %true_val = arith.constant 100.0 : f32
  %false_val = arith.constant 200.0 : f32
  %res = "neura.sel"(%cond, %true_val, %false_val) : (i1, f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 200.000000

  return %res : f32
}

func.func @test_sel_nested_with_comparison() -> f32 {
  %a = arith.constant 2.0 : f32
  %b = arith.constant 3.0 : f32
  %cond1 = "neura.fcmp"(%a, %b) {cmpType = "gt"} : (f32, f32) -> i1

  %true_val1 = arith.constant 100.0 : f32
  %false_val1 = arith.constant 200.0 : f32
  %sel1 = "neura.sel"(%cond1, %true_val1, %false_val1) : (i1, f32, f32) -> f32

  %c = arith.constant 5.0 : f32
  %d = arith.constant 1.0 : f32
  %cond2 = "neura.fcmp"(%c, %d) {cmpType = "gt"} : (f32, f32) -> i1

  %true_val2 = arith.constant 300.0 : f32
  %res = "neura.sel"(%cond2, %true_val2, %sel1) : (i1, f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 300.000000

  return %res : f32
}