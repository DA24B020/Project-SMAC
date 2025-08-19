#!/usr/bin/env bash
#
# assn2b.sh — Q2 wrapper: generate, sort, plot 20 random points & connections

# Exit on any error
set -e

# 1) Run the Python generator/plotter
python3 assn2b.py

# 2) Notify
echo "Done: points saved in assn2b.txt, plot in assn2b.png"
