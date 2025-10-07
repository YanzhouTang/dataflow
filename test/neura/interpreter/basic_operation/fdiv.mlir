// RUN: neura-interpreter %s --verbose | FileCheck %s

func.func @test_fdiv_positive() -> f32 {
  %a = arith.constant 10.0 : f32
  %b = arith.constant 2.0 : f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 5.000000
  return %res : f32
}

func.func @test_fdiv_negative_dividend() -> f32 {
  %a = arith.constant -10.0 : f32
  %b = arith.constant 2.0 : f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: -5.000000
  return %res : f32
}

func.func @test_fdiv_negative_divisor() -> f32 {
  %a = arith.constant 10.0 : f32
  %b = arith.constant -2.0 : f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: -5.000000
  return %res : f32
}

func.func @test_fdiv_two_negatives() -> f32 {
  %a = arith.constant -10.0 : f32
  %b = arith.constant -2.0 : f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 5.000000
  return %res : f32
}

func.func @test_fdiv_by_zero() -> f32 {
  %a = arith.constant 5.0 : f32
  %b = arith.constant 0.0 : f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: nan
  return %res : f32
}

func.func @test_fdiv_zero_dividend() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 5.0 : f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %res : f32
}

func.func @test_fdiv_with_embed_predicate_true() -> f32 {
  %a = "neura.constant"() {value = 15.0 : f32} : () -> f32
  %b = "neura.constant"() {value = 3.0 : f32} : () -> f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 5.000000
  return %res : f32
}

func.func @test_fdiv_f64() -> f64 {
  %a = arith.constant 10.5 : f64
  %b = arith.constant 2.5 : f64
  %res = "neura.fdiv"(%a, %b) : (f64, f64) -> f64
  // CHECK: [neura-interpreter]  → Output: 4.200000
  return %res : f64
}

func.func @test_fdiv_decimal() -> f32 {
  %a = arith.constant 2.0 : f32
  %b = arith.constant 0.5 : f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 4.000000
  return %res : f32
}

func.func @test_fdiv_large_numbers() -> f32 {
  %a = arith.constant 1.0e20 : f32
  %b = arith.constant 1.0e10 : f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 10000000000.000000
  return %res : f32
}

func.func @test_fdiv_near_zero() -> f32 {
  %a = arith.constant 1.0e-20 : f32
  %b = arith.constant 1.0e-10 : f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %res : f32
}