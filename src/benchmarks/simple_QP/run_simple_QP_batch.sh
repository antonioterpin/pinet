#!/usr/bin/env bash

for id in "dc3_simple_1" "dc3_simple_2" "dc3_nonconvex_1" "dc3_nonconvex_2"; do
    for config in "benchmark_config_manual"; do
        for seed in 0 1 2 3 4; do
            PYTHONPATH=src python src/benchmarks/simple_QP/run_simple_QP.py \
            --id "$id" \
            --config "$config" \
            --seed "$seed"
        done
    done
done
