#!/usr/bin/env bash
# Show progress of all running mebench experiments + recently finished/failed ones.
# Usage: watch -n 5 bash scripts/watch_runs.sh
#        bash scripts/watch_runs.sh  (one-shot)

set -u
cd "$(dirname "$0")/.."

# --- Find unique running configs (any ppid; orphans included) ---
mapfile -t RUNNING_CFGS < <(
    pgrep -af "python[0-9]* -m mebench run --config" \
        | sed -nE 's|.*--config ([^ ]+\.yaml).*|\1|p' \
        | sort -u
)

# --- GPU summary ---
gpu_line=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
load=$(awk '{print $1, $2, $3}' /proc/loadavg)

echo "=== $(date '+%H:%M:%S')  running=${#RUNNING_CFGS[@]}  GPU=${gpu_line:-N/A}  load=${load} ==="

if [ "${#RUNNING_CFGS[@]}" -eq 0 ]; then
    echo "  (no mebench experiments running)"
fi

parse_progress() {
    # Args: $1=path to experiment.log (or stdout log)
    local log="$1"
    # Crash detection
    if grep -q "OutOfMemoryError\|CUDA out of memory" "$log" 2>/dev/null; then
        echo "OOM"
        return
    fi
    if tail -50 "$log" 2>/dev/null | grep -q "^Traceback\|Error:"; then
        echo "CRASHED"
        return
    fi
    # Completion
    if grep -q "Attack run complete\|=== Run Summary ===\|All seeds completed" "$log" 2>/dev/null; then
        echo "DONE"
        return
    fi
    # Try [Trainer] step= line (SubstituteTrainer)
    local trainer
    trainer=$(grep "\[Trainer\]" "$log" 2>/dev/null | tail -1)
    if [ -n "$trainer" ]; then
        echo "$trainer" | grep -oE "step=[0-9]+ val=[0-9.]+ best=[0-9.]+@[0-9]+ patience=[0-9]+/[0-9]+"
        return
    fi
    # SwiftThief phases
    local sw
    sw=$(grep -oE "\[SwiftThief\] (CL|KD): *[0-9]+%" "$log" 2>/dev/null | tail -1)
    if [ -n "$sw" ]; then
        echo "$sw"
        return
    fi
    # MARICH scoring
    local mr
    mr=$(grep -oE "\[MARICH\] [^[:cntrl:]]*[0-9]+%[^[:cntrl:]]*" "$log" 2>/dev/null | tail -1 | cut -c1-60)
    if [ -n "$mr" ]; then
        echo "$mr"
        return
    fi
    # DFMS stages (no-query stages 1-4, alternate stage 5)
    local df_stage df_prog
    df_stage=$(grep "\[DFMSHL\] Stage" "$log" 2>/dev/null | tail -1 | grep -oE "Stage [0-9]+/[0-9]+: [^(]+")
    df_prog=$(grep -oE "\[DFMSHL\] \[(Evaluation|Progress)\][^[:cntrl:]]*" "$log" 2>/dev/null | tail -1 | cut -c1-80)
    if [ -n "$df_prog" ]; then
        echo "$df_prog"
        return
    fi
    if [ -n "$df_stage" ]; then
        echo "DFMS $df_stage"
        return
    fi
    # DFME/DS/MAZE/DisGuide generic data-free Progress lines
    local pg
    pg=$(grep -oE "step=[0-9]+ acc_gt=[0-9.]+ agreement=[0-9.]+" "$log" 2>/dev/null | tail -1)
    if [ -n "$pg" ]; then
        echo "$pg"
        return
    fi
    # Query Progress (random/activethief before training)
    local qp
    qp=$(grep -oE "\[Query Progress\] Used: [0-9]+ / Remaining: [0-9]+" "$log" 2>/dev/null | tail -1)
    if [ -n "$qp" ]; then
        echo "$qp"
        return
    fi
    echo "(starting)"
}

get_acc() {
    # Args: $1=run name
    local summary
    summary=$(ls -t runs/"$1"/*/seed_*/summary.json 2>/dev/null | head -1)
    [ -n "$summary" ] || { echo ""; return; }
    python3 -c "
import json, sys
d = json.load(open('$summary'))
chk = d.get('checkpoints', {})
keys = sorted(chk.keys(), key=lambda k: int(k))
if not keys: print(''); sys.exit()
v = chk[keys[-1]].get('track_b', chk[keys[-1]])
print('acc=%.4f agr=%.4f' % (v.get('acc_gt', float('nan')), v.get('agreement', float('nan'))))
" 2>/dev/null
}

# --- Running experiments ---
for cfg in "${RUNNING_CFGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    # Find most recent log for this run name
    log=$(ls -t runs/"$name"/*/seed_*/experiment.log 2>/dev/null | head -1)
    if [ -z "$log" ]; then
        # Fallback: stdout log under logs/adamw_rerun/
        log=$(ls -t logs/adamw_rerun/"$name".log 2>/dev/null | head -1)
    fi
    if [ -z "$log" ]; then
        printf "  ? %-60s (no log yet)\n" "$name"
        continue
    fi
    status=$(parse_progress "$log")
    printf "  … %-60s %s\n" "$name" "$status"
done

# --- Recently finished (last 30 min, not currently running) ---
echo ""
echo "--- Recently finished (last 30 min) ---"
running_set=$(printf '%s\n' "${RUNNING_CFGS[@]}" | sed -E 's|configs/matrix/||; s|\.yaml$||')
find runs -maxdepth 4 -name summary.json -mmin -30 2>/dev/null \
    | sort -r \
    | while read -r summary; do
        name=$(echo "$summary" | sed -E 's|^runs/([^/]+)/.*|\1|')
        # Skip if currently running
        if echo "$running_set" | grep -qx "$name"; then continue; fi
        acc=$(python3 -c "
import json
d = json.load(open('$summary'))
chk = d.get('checkpoints', {})
keys = sorted(chk.keys(), key=lambda k: int(k))
if not keys:
    print('no_checkpoint'); raise SystemExit
v = chk[keys[-1]].get('track_b', chk[keys[-1]])
print('acc=%.4f agr=%.4f' % (v.get('acc_gt', float('nan')), v.get('agreement', float('nan'))))
" 2>/dev/null)
        printf "  ✓ %-60s %s\n" "$name" "$acc"
    done | head -20

# --- Recently failed (OOM/Traceback in last 30 min) ---
echo ""
echo "--- Recent failures ---"
running_set_log=$(printf '%s\n' "${RUNNING_CFGS[@]}" | sed -E 's|configs/matrix/||; s|\.yaml$||')
for log in $(find logs/adamw_rerun -maxdepth 1 -name "*.log" -mmin -60 2>/dev/null); do
    name=$(basename "$log" .log)
    if grep -q "OutOfMemoryError\|CUDA out of memory" "$log" 2>/dev/null; then
        printf "  ✗ %-60s OOM\n" "$name"
    elif tail -50 "$log" 2>/dev/null | grep -qE "^Traceback"; then
        # only report if not still running
        if ! echo "$running_set_log" | grep -qx "$name"; then
            printf "  ✗ %-60s CRASHED\n" "$name"
        fi
    fi
done | sort -u | head -20
