#!/usr/bin/env bash
# Re-run every experiment behind the report, in order, in one environment (the Docker image):
#   docker compose run -d --name sa-exp stringart bash experiments/run_all.sh
#   docker logs -f sa-exp
# Needs the datasets and face models in the volumes (see compose.yaml).
set -euo pipefail
for s in evaluate_dataset evaluate_refine evaluate_color evaluate_palette evaluate_exposure; do
    echo "== ${s} $(date -u +%H:%M:%S)"
    # pipefail makes a failing experiment fail the run; grep itself may match nothing.
    python "experiments/${s}.py" 2>&1 | { grep --line-buffered -v -e '\[ WARN' -e 'loading data' || true; }
done
echo "== ALL-DONE $(date -u +%H:%M:%S)"
