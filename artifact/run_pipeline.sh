#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

mode="${1:-full}"

run_preprocessing() {
    local config_path="$1"
    local output_root="$2"
    shift
    shift

    python scripts/01_preprocessing/generate_models.py \
        --config "$config_path" \
        --output-dir "$output_root/models" \
        "$@"
}

config_options=()

set_config_options() {
    config_options=()
    local config_name
    for config_name in "$@"; do
        config_options+=(--config "$config_name")
    done
}

run_preprocessing_stages() {
    local output_root="$1"
    shift
    set_config_options "$@"

    python scripts/01_preprocessing/evaluate_models.py \
        --models-dir "$output_root/models" \
        "${config_options[@]}"
    python scripts/01_preprocessing/compute_suspiciousness.py \
        --models-dir "$output_root/models" \
        --output-dir "$output_root/suspiciousness" \
        "${config_options[@]}"
    python scripts/01_preprocessing/transform_datasets.py \
        --models-dir "$output_root/models" \
        --sus-dir "$output_root/suspiciousness" \
        --output-dir "$output_root/datasets" \
        "${config_options[@]}"
}

run_classification() {
    local output_root="$1"
    shift
    set_config_options "$@"

    python scripts/02_classification/train_classifiers.py \
        --dataset-dir "$output_root/datasets" \
        --output-dir "$output_root/results" \
        "${config_options[@]}"
}

run_interpretation() {
    local output_root="$1"
    shift
    set_config_options "$@"

    python scripts/02_classification/interpret_features.py \
        --results-dir "$output_root/results" \
        "${config_options[@]}"
}

run_rq1_rq2() {
    run_preprocessing configs/artifact/baseline_1000.yaml outputs \
        --dataset mnist --dataset fmnist
    run_preprocessing_stages outputs baseline_1000
    run_classification outputs baseline_1000
}

run_rq3() {
    if [[ ! -d outputs/results/baseline_1000 ]]; then
        echo "RQ3 requires the classifiers from rq1-rq2." >&2
        echo "Run: bash artifact/run_pipeline.sh rq1-rq2" >&2
        exit 1
    fi
    run_interpretation outputs baseline_1000
}

run_rq4() {
    run_preprocessing configs/artifact/rq4_25.yaml outputs --dataset mnist
    run_preprocessing configs/artifact/rq4_50.yaml outputs --dataset mnist
    run_preprocessing_stages outputs rq4_25 rq4_50
    run_classification outputs rq4_25 rq4_50
}

run_all() {
    run_rq1_rq2
    run_rq3
    run_rq4
}

case "$mode" in
    check)
        python artifact/check_environment.py
        ;;
    smoke)
        run_preprocessing configs/artifact/smoke.yaml outputs/smoke \
            --dataset mnist
        run_preprocessing_stages outputs/smoke smoke
        run_classification outputs/smoke smoke
        run_interpretation outputs/smoke smoke
        ;;
    rq1-rq2)
        run_rq1_rq2
        ;;
    rq3)
        run_rq3
        ;;
    rq4)
        run_rq4
        ;;
    all)
        python artifact/check_environment.py
        run_all
        ;;
    full)
        python artifact/check_environment.py --expect-device cuda
        run_all
        ;;
    classify)
        python scripts/02_classification/train_classifiers.py
        python scripts/02_classification/interpret_features.py
        ;;
    *)
        echo "Usage: $0 {check|smoke|rq1-rq2|rq3|rq4|all|full|classify}" >&2
        exit 2
        ;;
esac
