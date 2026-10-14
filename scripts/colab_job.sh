#!/usr/bin/env bash
# Run all inference experiments on one Colab A100 runtime.
#   colab new -s infer --gpu A100
#   colab upload -s infer scripts/colab_job.sh /content/colab_job.sh
#   echo 'import subprocess; subprocess.run(["bash", "/content/colab_job.sh"])' | colab exec -s infer
#
# Each experiment is independent: one failing is logged and the rest still run,
# and results.tgz is written on exit whatever happens, so a crash never loses finished results.
set -uo pipefail
REPO=${REPO:-https://github.com/kishorhange111/llm-inference-optimization}
BRANCH=${BRANCH:-main}
VLLM_VERSION=${VLLM_VERSION:-0.11.0}  # pinned: vLLM's CLI and flags change between releases
DIR=/content/llm-inference-optimization
trap 'tar -czf /content/results.tgz -C "$DIR" results 2>/dev/null; echo "packed results ($(date -u +%T))"' EXIT
set -e
# Clone into place even if DIR already holds restored results/checkpoints (a resumed runtime).
if [ ! -d "$DIR/.git" ]; then
  mkdir -p "$DIR" && git clone --depth 1 -b "$BRANCH" "$REPO" /content/_clone && cp -a /content/_clone/. "$DIR"/ && rm -rf /content/_clone
else
  git -C "$DIR" fetch --depth 1 origin "$BRANCH" && git -C "$DIR" reset --hard FETCH_HEAD
fi
cd "$DIR"
pip install -q -e ".[gpu]"
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p results
set +e

run() {  # $1 = log name, rest = command; failures are recorded, not fatal
  local name=$1; shift
  echo "=== $name ($(date -u +%T))"
  "$@" 2>&1 | tee "results/$name.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || echo "FAILED: $name" | tee -a results/failures.txt
}
run kv python scripts/run_experiments.py kv
run quant python scripts/run_experiments.py quant --model Qwen/Qwen2.5-7B-Instruct
run attention python scripts/run_experiments.py attention --model Qwen/Qwen2.5-1.5B-Instruct
run compile python scripts/run_experiments.py compile --model Qwen/Qwen2.5-1.5B-Instruct
run speculative python scripts/run_experiments.py speculative --model Qwen/Qwen2.5-7B-Instruct --draft Qwen/Qwen2.5-0.5B-Instruct

# Online serving: vLLM (continuous batching + PagedAttention) in bf16, then the AWQ 4-bit checkpoint.
pip install -q "vllm==$VLLM_VERSION"
serve() {  # $1 = model, $2 = label
  python -m vllm.entrypoints.openai.api_server --model "$1" --max-model-len 4096 --gpu-memory-utilization 0.9 --port 8000 > "results/vllm_$2.log" 2>&1 &
  local pid=$! ready=0
  for _ in $(seq 1 120); do
    if curl -sf localhost:8000/v1/models >/dev/null; then ready=1; break; fi
    kill -0 $pid 2>/dev/null || break  # the server died; no point waiting
    sleep 5
  done
  if [ $ready = 1 ]; then
    run "serve_$2" python scripts/run_experiments.py serve --model "$1" --tokenizer Qwen/Qwen2.5-7B-Instruct --label "$2"
  else
    echo "FAILED: vLLM server for $2 did not start" | tee -a results/failures.txt
    tail -30 "results/vllm_$2.log"
  fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 5
}
serve Qwen/Qwen2.5-7B-Instruct bf16
serve Qwen/Qwen2.5-7B-Instruct-AWQ awq
echo "DONE $(date -u +%T)"; cat results/failures.txt 2>/dev/null
