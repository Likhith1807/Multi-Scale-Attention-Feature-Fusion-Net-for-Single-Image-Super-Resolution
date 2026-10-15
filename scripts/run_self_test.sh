#!/usr/bin/env bash
# Run the complete test suite
set -euo pipefail

echo "=========================================================="
echo " Running MSAF-Net Comprehensive Verification & Unit Tests "
echo "=========================================================="

python self_test.py
echo "All tests passed successfully!"
