## Infini-LLM

LLama-3 with the major focus on updating the "KV-cache" mechanism for extended context processing support in the resource-constrained systems.


### Basic Model Usage

```python
from InfiniLLM import Llama
from transformers import AutoTokenizer

# Load model
model = Llama.LLama.build(
    chkpt_dir="/path/to/checkpoint",
    max_seq_len=8192,
    max_batch_size=1,
    use_cache=True,
    attn_method="window",  # Options: "full", "window", "linear"
    window_size=512
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/checkpoint")

# Generate text
prompt = "Artificial intelligence is transforming the way we"
inputs = tokenizer(prompt, return_tensors="pt")
# ... generation code ...
```
### Run All Benchmarks

```bash
python run_benchmarks.py --model_path $LLAMA_DIR --benchmark all \
    --seq_len 16384 \
    --attn_method window \
    --window_size 512
```

## Benchmark Results Structure

Results are saved in the following structure:

```
benchmark_results/
├── lm_eval/
│   ├── hellaswag_arc_easy_etc_results.json
│   └── ...
├── scbench/
│   ├── string_needle_multi_turn_result.json
│   ├── semantic_qa_multi_request_result.json
│   └── ...
└── ruler/
    ├── niah_single_ctx4096.json
    ├── niah_multi_ctx8192.json
    ├── ruler_summary.json
    └── ...
```

## Advanced Usage

### Custom Benchmark Configuration

```python
from benchmarks.lm_eval import LMEvalHarnessRunner
from benchmarks.scbench import SCBenchRunner
from benchmarks.ruler import RULERBenchmark, RULERTaskConfig, RULERTaskType

# LM Eval Harness
runner = LMEvalHarnessRunner(model, tokenizer_path, output_dir="results")
results = runner.run(tasks=["hellaswag"], num_fewshot=5)

# SCBench
scbench = SCBenchRunner(model, tokenizer_path)
result = scbench.run_multi_turn_session(
    shared_context="...",
    queries=["query1", "query2"],
    expected_answers=["ans1", "ans2"]
)

# RULER
ruler = RULERBenchmark(model, tokenizer_path)
config = RULERTaskConfig(
    task_type=RULERTaskType.NIAH_SINGLE,
    context_length=8192,
    num_needles=1
)
result = ruler.run_task(config, num_examples=10)
```

## Project Structure

```
Infini-LLM/
├── InfiniLLM/           # Core model implementation
│   └── model/
│       ├── Llama.py     # Main Llama model
│       └── ops.py       # Operations
├── benchmarks/          # Benchmark integrations
│   ├── lm_eval/        # LM Evaluation Harness
│   ├── scbench/        # SCBench
│   └── ruler/          # RULER
├── Llama_analysis.py    # Analysis and visualization
├── run_benchmarks.py    # Main benchmark runner
└── requirements.txt     # Dependencies
```

## Environment Variables

```bash
# Set model checkpoint directory
export LLAMA_DIR=/path/to/llama/checkpoint

# Run with environment variable
python run_benchmarks.py --benchmark all
```


