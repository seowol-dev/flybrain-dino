#!/usr/bin/env bash
# 모든 시나리오를 병렬로 실행하고 results/ 에 저장
cd "$(dirname "$0")/.."
mkdir -p results
run() { uv run flybrain run "$@" > "results/log_${*// /_}.txt" 2>&1; }
run feeding --duration 30 &
run feeding --duration 30 --silence-type CB0701 --name feeding_MN9_silenced &
run wind --duration 4 &
run touch --duration 3 &
run odor --duration 20 &
run thermal --duration 15 &
run looming --duration 3 &
wait
uv run python examples/sugar_vs_bitter.py > results/log_sugar_vs_bitter.txt 2>&1
uv run python examples/quickstart.py > results/log_quickstart.txt 2>&1
grep -h -A1 "완료\|Traceback" results/log_*.txt
tail -5 results/log_sugar_vs_bitter.txt
