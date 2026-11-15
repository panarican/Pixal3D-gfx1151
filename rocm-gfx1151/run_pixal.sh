#!/bin/bash
# Usage: rocm-gfx1151/run_pixal.sh <image> <out.glb>
# Runs inference with smart VRAM + timings, samples GPU memory (VRAM + GTT) every 2 s into vram.log,
# and prints the timings and peak GPU memory at the end.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMG="$(realpath -m "$1")"; OUT="$(realpath -m "$2")"; LOG="$REPO/run.log"; VRAM="$REPO/vram.log"
CARD=${CARD:-$(ls -d /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1 | xargs -r dirname)}
rm -f "$OUT" "$VRAM"; : > "$LOG"; start=$(date +%s)
if [ -n "$CARD" ]; then
  ( while true; do echo "$(date +%s) $(cat $CARD/mem_info_vram_used) $(cat $CARD/mem_info_gtt_used)" | awk '{printf "%d %.2f\n", $1, ($2+$3)/2^30}' >> "$VRAM"; sleep 2; done ) &
  SAMPLER=$!
fi
# oom_score_adj 500: if the machine runs out of memory, the kernel stops this job before other processes
bash -c "echo 500 > /proc/self/oom_score_adj; source '$REPO/rocm-gfx1151/env.sh' && cd '$REPO' && PIXAL_TIMING=1 exec python inference.py --image '$IMG' --output '$OUT' --smart_vram" > "$LOG" 2>&1
rc=$?; end=$(date +%s); [ -n "${SAMPLER:-}" ] && kill $SAMPLER 2>/dev/null
echo "exit=$rc wall=$((end-start))s"
[ -s "$VRAM" ] && awk 'NR==1{b=$2} {if($2>m)m=$2} END{printf "GPU memory: before %.1f GiB, peak %.1f GiB (+%.1f)\n", b, m, m-b}' "$VRAM"
grep -E '^\[TIMING\] [a-z]|Traceback' "$LOG" | cut -c1-160
ls -la "$OUT" 2>/dev/null
