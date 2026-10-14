#!/usr/bin/env bash
# Run all inference experiments on one Colab A100 runtime.
#   colab new -s infer --gpu A100
#   colab upload -s infer scripts/colab_job.sh /content/colab_job.sh
#   echo 'import subprocess; subprocess.run(["bash", "/content/colab_job.sh"])' | colab exec -s infer
set -euo pipefail
REPO=${REPO:-https://github.com/kishorhange111/llm-inference-optimization}
BRANCH=${BRANCH:-main}
DIR=/content/llm-inference-optimization
if [ ! -d "$DIR" ]; then git clone --depth 1 -b "$BRANCH" "$REPO" "$DIR"; else git -C "$DIR" fetch --depth 1 origin "$BRANCH" && git -C "$DIR" reset --hard FETCH_HEAD; fi
cd "$DIR"
pip install -q -e ".[gpu]"
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p results

python scripts/run_experiments.py kv
python scripts/run_experiments.py quant --model Qwen/Qwen2.5-7B-Instruct 2>&1 | tee results/quant.log
python scripts/run_experiments.py attention --model Qwen/Qwen2.5-1.5B-Instruct 2>&1 | tee results/attention.log
python scripts/run_experiments.py compile --model Qwen/Qwen2.5-1.5B-Instruct 2>&1 | tee results/compile.log
python scripts/run_experiments.py speculative --model Qwen/Qwen2.5-7B-Instruct --draft Qwen/Qwen2.5-0.5B-Instruct 2>&1 | tee results/speculative.log

# Online serving: vLLM (continuous batching + PagedAttention) in bf16, then the AWQ 4-bit checkpoint.
pip install -q "vllm>=0.6"
serve() {  # $1 = model, $2 = label
  python -m vllm.entrypoints.openai.api_server --model "$1" --max-model-len 4096 --gpu-memory-utilization 0.9 --port 8000 > "results/vllm_$2.log" 2>&1 &
  PID=$!
  for i in $(seq 1 120); do curl -sf localhost:8000/v1/models >/dev/null && break; sleep 5; done
  python scripts/run_experiments.py serve --model "$1" --tokenizer Qwen/Qwen2.5-7B-Instruct --label "$2" 2>&1 | tee -a results/serve.log
  kill $PID; wait $PID 2>/dev/null || true; sleep 5
}
serve Qwen/Qwen2.5-7B-Instruct bf16
serve Qwen/Qwen2.5-7B-Instruct-AWQ awq
tar -czf /content/results.tgz results
echo DONE
