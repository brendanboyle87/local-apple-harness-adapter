#!/bin/sh
set -eu
mkdir -p /logs/verifier
if [ ! -L /workspace/answer.txt ] && [ -f /workspace/answer.txt ] \
    && printf 'harbor ready\n' | cmp -s - /workspace/answer.txt; then
    printf '1\n' > /logs/verifier/reward.txt
else
    printf '0\n' > /logs/verifier/reward.txt
fi
