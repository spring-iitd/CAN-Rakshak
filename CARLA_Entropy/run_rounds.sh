#!/bin/bash
set -e

TOTAL_ROUNDS=4

for ((i=0; i<$TOTAL_ROUNDS; i++))
do
    echo "======================================="
    echo "Running Round $i"
    echo "======================================="
    python driver_dos_CARLA.py --round $i
done
