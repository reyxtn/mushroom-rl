#!/usr/bin/env bash
#
# Pick the best walk and run checkpoints from an H2 sweep folder, finished or
# still running. Reads each seed's <stage>.best (written when a stage ends),
# so seeds still training are simply skipped. Never touches the sweep itself;
# copies the winners into <sweep>/picked/.
#
#   ./h2_pick_best.sh                                  # newest sweep under ./runs
#   ./h2_pick_best.sh runs/h2_sweep_20261009_065700_ab12
#   MIN_ALIVE=0.95 ./h2_pick_best.sh <sweep>
#
# Selection: among seeds whose best epoch kept alive_frac >= MIN_ALIVE, the one
# with the lowest lin_vel_err. Unlike J, which barely differs between seeds,
# this is the number that says how well the policy does what it is told.
# Falls back to the highest alive_frac if no seed passes MIN_ALIVE.

MIN_ALIVE="${MIN_ALIVE:-0.9}"

set -uo pipefail
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

SWEEP="${1:-$(ls -d runs/h2_sweep_* 2>/dev/null | sort | tail -1)}"
[[ -n "$SWEEP" && -d "$SWEEP" ]] || die "no sweep folder given and none found under ./runs"
SWEEP="$(cd "$SWEEP" && pwd)"
echo "sweep: $SWEEP"
echo

TABLE="$(mktemp)"
printf 'seed\tstage\tbest_epoch\tJ\tmean_ep_len\tlin_vel_err\talive_frac\tstatus\n' > "$TABLE"
for dir in "$SWEEP"/seed_*; do
    [[ -d "$dir" ]] || continue
    seed="${dir##*_}"
    status="$(cat "$dir/status" 2>/dev/null || echo "running")"
    for stage in stand walk run; do
        [[ -s "$dir/$stage.best" && -f "$dir/$stage.msh" ]] || continue
        read -r e j l v a < "$dir/$stage.best"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$seed" "$stage" "$e" "$j" "$l" "$v" "$a" "$status" >> "$TABLE"
    done
done

if [[ "$(wc -l < "$TABLE")" -le 1 ]]; then
    rm -f "$TABLE"
    die "no finished stages yet in $SWEEP"
fi
column -t -s $'\t' "$TABLE" 2>/dev/null || cat "$TABLE"
echo

mkdir -p "$SWEEP/picked"
for stage in walk run; do
    best="$(awk -F'\t' -v s="$stage" -v m="$MIN_ALIVE" '
        NR > 1 && $2 == s {
            ok = ($7 + 0 >= m + 0)
            if (ok && (!fo || $6 + 0 < bv + 0)) { fo = 1; bv = $6; rowo = $0 }
            if (!fa || $7 + 0 > ba + 0)        { fa = 1; ba = $7; rowa = $0 }
        }
        END { if (fo) print "ok\t" rowo; else if (fa) print "fallback\t" rowa }
    ' "$TABLE")"
    if [[ -z "$best" ]]; then
        echo "best $stage: none finished yet"
        continue
    fi
    IFS=$'\t' read -r how seed _ e j l v a _ <<< "$best"
    src="$SWEEP/seed_$seed/$stage.msh"
    dst="$SWEEP/picked/h2_${stage}_seed${seed}.msh"
    cp "$src" "$dst"
    {
        echo "task=$stage seed=$seed best_epoch=$e"
        echo "J=$j mean_ep_len=$l lin_vel_err=$v alive_frac=$a"
        echo "picked by lowest lin_vel_err with alive_frac >= $MIN_ALIVE ($how)"
        cat "$SWEEP/config.txt" 2>/dev/null
        echo "source=$src"
    } > "${dst%.msh}.txt"
    [[ "$how" == fallback ]] && note=" (no seed reached alive_frac $MIN_ALIVE; picked the most robust)" || note=""
    echo "best $stage: seed $seed, epoch $e, lin_vel_err=$v alive=$a ep_len=$l$note"
    echo "    -> $dst"
done
cp "$TABLE" "$SWEEP/picked/summary.tsv"
rm -f "$TABLE"
