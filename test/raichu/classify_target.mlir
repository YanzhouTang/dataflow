// RUN: mlir-neura-opt %s --raichu-classify-target | FileCheck %s

// Ray marching -> M-CGRA.
// CHECK: func.func @ray_march_func
// CHECK-SAME: raichu.target = "m-cgra"
func.func @ray_march_func(%arg0: f32) -> f32 {
  return %arg0 : f32
}

// Hash-grid encoding -> M-CGRA.
// CHECK: func.func @hash_encoder_func
// CHECK-SAME: raichu.target = "m-cgra"
func.func @hash_encoder_func(%arg0: f32) -> f32 {
  return %arg0 : f32
}

// MLP -> C-CGRA.
// CHECK: func.func @nerf_mlp_func
// CHECK-SAME: raichu.target = "c-cgra"
func.func @nerf_mlp_func(%arg0: f32) -> f32 {
  return %arg0 : f32
}

// Pixel composition / blending -> left unclassified for now.
// CHECK: func.func @pixel_composition_func
// CHECK-NOT: raichu.target
func.func @pixel_composition_func(%arg0: f32) -> f32 {
  return %arg0 : f32
}
