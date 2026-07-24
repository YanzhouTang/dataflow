#!/usr/bin/env bash
# =============================================================================
# RAICHU Compiler — Artifact Evaluation (AE) driver
#
# Reproduces the per-kernel Initiation Interval (II) and schedule makespan
# (steps) for every kernel of the five RAICHU rendering methods, by compiling
# each kernel to the Neura dialect and mapping it onto the RAICHU M-CGRA / C-CGRA
# with the in-tree mapper (`mlir-neura-opt --map-to-accelerator`).
#
# These per-kernel (II, steps) are the compiler subproject's deliverable to the
# RAICHU simulator: frame latency = II * (trip_count - 1) + steps, with
# trip_count supplied by the scene/model workload (see latency_analysis/*.md).
#
# Usage (from inside the neura dev container, at the repo root `dataflow/`):
#   bash raichu_ae/run_ae.sh                 # run all methods
#   bash raichu_ae/run_ae.sh 3dgs            # run one method (mnerf|hnerf|lnerf|3dgs|rt)
#
# Environment overrides:
#   OPT      path to mlir-neura-opt   (default ./build/tools/mlir-neura-opt/mlir-neura-opt)
#   LLVM     path to clang++/mlir-translate bin dir (default /workspace/llvm-project/build/bin)
#   TIMEOUT  per-kernel mapping timeout in seconds (default 600)
# =============================================================================
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
OPT="${OPT:-./build/tools/mlir-neura-opt/mlir-neura-opt}"
LLVM="${LLVM:-/workspace/llvm-project/build/bin}"
TIMEOUT="${TIMEOUT:-900}"
ARCHDIR="test/arch_spec"
CROOT="test/c2llvm2mlir"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

CFLAGS="-S -emit-llvm -O3 -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -ffp-contract=off"
# Full front-to-back pipeline used for the C-source rendering kernels.
PIPE_CPP=(--assign-accelerator --lower-llvm-to-neura --fuse-pattern
          --raichu-fuse-fvcu --raichu-model-fvcu-latency
          --promote-input-arg-to-const --fold-constant --canonicalize-return
          --canonicalize-live-in --leverage-predicated-value
          --transform-ctrl-to-data-flow --fold-constant --insert-data-mov)

[ -x "$OPT" ] || { echo "ERROR: mlir-neura-opt not found at $OPT (build the project first)"; exit 1; }

pass=0; fail=0
printf "%-8s %-24s %-8s %-8s %-8s %-6s\n" "METHOD" "KERNEL" "CGRA" "II" "steps" "check"
printf -- "---------------------------------------------------------------------\n"

extract_ii()    { grep -oE "compiled_ii = [0-9]+" "$1" 2>/dev/null | head -1 | grep -oE "[0-9]+"; }
extract_steps() { local s; s=$(grep -oE "time_step = [0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1); [ -n "$s" ] && echo $((s+1)); }

# Map a C-source kernel: compile -> import -> map. Args: method kernel arch bt ms expII
run_cpp() {
  local method=$1 kernel=$2 arch=$3 bt=$4 ms=$5 exp=$6
  local src; src=$(find "$CROOT" -name "$kernel.cpp" | head -1)
  local out="$WORK/$kernel.mapped.mlir"
  if [ -z "$src" ]; then printf "%-8s %-24s %-8s %-8s\n" "$method" "$kernel" "-" "NOFILE"; ((fail++)); return; fi
  "$LLVM/clang++" $CFLAGS "$src" -o "$WORK/$kernel.ll" 2>/dev/null
  "$LLVM/mlir-translate" --import-llvm "$WORK/$kernel.ll" -o "$WORK/$kernel.mlir" 2>/dev/null
  timeout "$TIMEOUT" "$OPT" "$WORK/$kernel.mlir" "${PIPE_CPP[@]}" \
    --map-to-accelerator="mapping-strategy=heuristic backtrack-config=$bt max-steps=$ms" \
    --architecture-spec="$ARCHDIR/raichu_${arch}_cgra.yaml" -o "$out" >/dev/null 2>&1
  report "$method" "$kernel" "$arch-cgra" "$out" "$exp"
}

# Map a hand-written .mlir kernel by executing the pipeline in its `// RUN:` line.
run_mlir() {
  local method=$1 kernel=$2 rel=$3 exp=$4
  local f="test/raichu/$rel"
  local out="$WORK/$kernel.mapped.mlir"
  if [ ! -f "$f" ]; then printf "%-8s %-24s %-8s %-8s\n" "$method" "$kernel" "-" "NOFILE"; ((fail++)); return; fi
  local cmd
  cmd=$(grep '// RUN:' "$f" | grep -v 'FileCheck' | sed 's|^// RUN:[[:space:]]*||' | tr -d '\\' | tr '\n' ' ')
  cmd=${cmd//mlir-neura-opt/$OPT}
  cmd=${cmd//%s/$f}
  cmd=${cmd//%S/$(dirname "$f")}
  cmd=${cmd//%t-mapped.mlir/$out}
  timeout "$TIMEOUT" bash -c "$cmd" >/dev/null 2>&1
  local cgra="c-cgra"; grep -q "raichu_m_cgra" <<< "$cmd" && cgra="m-cgra"
  report "$method" "$kernel" "$cgra" "$out" "$exp"
}

report() {
  local method=$1 kernel=$2 cgra=$3 out=$4 exp=$5
  local ii steps chk
  ii=$(extract_ii "$out"); steps=$(extract_steps "$out")
  if [ -z "$ii" ]; then ii="FAIL"; chk="x"; ((fail++));
  elif [ "$exp" = "-" ] || [ "$ii" = "$exp" ]; then chk="ok"; ((pass++));
  else chk="~$exp"; ((pass++)); fi
  printf "%-8s %-24s %-8s %-8s %-8s %-6s\n" "$method" "$kernel" "$cgra" "$ii" "${steps:-–}" "$chk"
}

M="${1:-all}"

if [ "$M" = all ] || [ "$M" = mnerf ]; then
  run_cpp M-NeRF mnerf_sample c simple 8 6
  run_cpp M-NeRF mnerf_posenc c simple 8 6
  run_cpp M-NeRF mnerf_mlp    c simple 8 14
  run_cpp M-NeRF mnerf_accum  c simple 8 7
fi
if [ "$M" = all ] || [ "$M" = hnerf ]; then
  run_mlir H-NeRF hash_gather  mcgra_hash_gather_map_ii.mlir 8
  run_mlir H-NeRF interp       ccgra_interp_map_ii.mlir       7
  run_mlir H-NeRF mlp_vec_dot  vec_matmul_body.mlir           2
fi
if [ "$M" = all ] || [ "$M" = lnerf ]; then
  run_cpp L-NeRF lnerf_line_gather c simple 8 6
  run_cpp L-NeRF lnerf_vm_product  c simple 8 6
  run_cpp L-NeRF lnerf_vm_reduce   c simple 8 6
  run_cpp L-NeRF lnerf_mlp         c simple 8 11
  run_cpp L-NeRF lnerf_appear_proj c simple 8 10
fi
if [ "$M" = all ] || [ "$M" = 3dgs ]; then
  run_cpp 3DGS rot_m_from_quat    c simple 8 9
  run_cpp 3DGS sigma3d_from_m     c simple 8 7
  run_cpp 3DGS cov2d_project      c simple 8 13
  run_cpp 3DGS preprocess_full    c simple 8 11
  run_cpp 3DGS bbox_radius        c simple 8 10
  run_cpp 3DGS sh_color           c simple 8 14
  run_cpp 3DGS tile_range_binning m simple 8 10
  run_cpp 3DGS sort_binning       c simple 8 7
  run_cpp 3DGS sort_partition     m simple 8 7
  run_cpp 3DGS sort_stage         m greedy 8 5
  run_cpp 3DGS blend_alpha_exp    c simple 8 12
  run_cpp 3DGS blend_stage2       c simple 8 7
  run_cpp 3DGS blend_early_term   c simple 8 11
  run_cpp 3DGS intersect_shape    c simple 8 14
  run_cpp 3DGS subtile_bitmap     c simple 8 13
fi
if [ "$M" = all ] || [ "$M" = rt ]; then
  run_cpp RT rt_ray_gen        c simple 8 11
  run_cpp RT rt_aabb           c simple 8 10
  run_cpp RT rt_compact        m simple 8 7
  run_cpp RT rt_triangle_setup c simple 8 12
  run_cpp RT rt_tri_vt         c simple 6 13
  run_cpp RT rt_tri_reduce     c simple 8 7
  run_cpp RT rt_shade          c simple 8 9
fi

printf -- "---------------------------------------------------------------------\n"
printf "PASS=%d  FAIL=%d\n" "$pass" "$fail"
[ "$fail" -eq 0 ]
