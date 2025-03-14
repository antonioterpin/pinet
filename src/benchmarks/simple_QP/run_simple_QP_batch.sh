#!/usr/bin/env bash

for id in "dc3_simple_1.yaml" "dc3_nonconvex_1.yaml"; do
    for config in "simple_QP_bicgstab_low.yaml" "simple_QP_fpi_low.yaml" "simple_QP_unroll_low.yaml"; do
        PYTHONPATH=src python src/benchmarks/simple_QP/run_simple_QP.py \
            --id "$id" \
            --config "$config"
    done
done
