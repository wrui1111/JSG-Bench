# JSG-Bench

Reproducibility code and annotation-schema examples for the Joint Spatial--Semantic Grounding Benchmark (JSG-Bench).

This repository contains the experiment and evaluation source code needed for JSG-Core, JSG-Scale, JSG-Xfer, the TD-EGFA reference pipeline, the native SIDA adapter audit, spatial controls, criterion-relaxation analyses, paired contrasts, multi-VLM diagnostics, and inter-annotator agreement (IAA) evaluation.

## Repository scope

Included:

- executable experiment and evaluation source code;
- the frozen `jsg-bench-audit-v1.0` contract;
- four small IAA schema examples, each containing one real, one full-synthetic, and one tampered record;
- download helpers and exact dependency versions.

Not included:

- SID-Set images, masks, full split files, or full semantic annotations;
- model checkpoints or model caches;
- raw model responses, intermediate requests, candidate crops, logs, reports, metrics, figures, or other result artifacts;
- SIDA source code, which remains in its upstream repository.

The exclusions above are also enforced by `.gitignore`.

## Layout

```text
annotation_examples/  Three-record IAA schema examples only
configs/              Frozen JSG audit contract
experiments/          Experiment, adapter, audit, and analysis code
scripts/              Dataset and model download helpers
requirements.txt      Recorded Python environment for JSG/TD-EGFA
```

## Environment

The recorded JSG/TD-EGFA runs used Python 3.9.6. Create a clean environment from the repository root:

```bash
git clone https://github.com/wrui1111/JSG-Bench.git
cd JSG-Bench
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p experiments/runs experiments/reports models external/downloads
```

PyTorch installation can be platform-specific. If the pinned wheel is unavailable for the local CUDA, ROCm, or Apple platform, install the matching PyTorch 2.8 build first and then run:

```bash
python -m pip install -r requirements.txt --no-deps
```

## Download SID-Set

SID-Set is hosted by its authors at `saberzl/SID_Set` on Hugging Face. The following command downloads and exports the validation split into the directory layout expected by the experiment scripts:

```bash
python scripts/export_sid_set_from_hf.py \
  --split validation \
  --output-root SID_Set
```

For a small download smoke test:

```bash
python scripts/export_sid_set_from_hf.py \
  --split validation \
  --output-root SID_Set \
  --limit 12
```

The upstream dataset may also be downloaded without export:

```bash
hf download saberzl/SID_Set \
  --repo-type dataset \
  --local-dir downloads/SID_Set
```

The four files under `annotation_examples/` document the IAA schemas only. They are not replacements for the complete human semantic annotations used to compute the paper's final metrics.

## Download model checkpoints

Primary Qwen2.5-VL-3B checkpoint:

```bash
python scripts/download_qwen25vl.py \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir models/Qwen2.5-VL-3B-Instruct
```

ModelScope fallback for the same checkpoint:

```bash
python scripts/download_qwen25vl_modelscope.py \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir models/Qwen2.5-VL-3B-Instruct
```

Optional multi-VLM diagnostic models used through the Ollama runner:

```bash
ollama pull qwen2.5vl:7b
ollama pull openbmb/minicpm-v4.5:latest
```

Run these models with `experiments/ollama_vlm_infer.py`. The diagnostic runs used temperature zero and retained the model's native Ollama packaging.

SIDA native-interface audit dependencies:

```bash
git clone https://github.com/hzlsaber/SIDA external/SIDA

hf download saberzl/SIDA-7B-description \
  --local-dir models/SIDA-7B-description

curl -fL \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
  -o models/sam_vit_h_4b8939.pth
```

Install SIDA's own dependency file inside a separate environment because its pinned stack differs from the JSG/TD-EGFA environment:

```bash
python3.9 -m venv .venv-sida
source .venv-sida/bin/activate
python -m pip install --upgrade pip
python -m pip install -r external/SIDA/requirements.txt
```

No checkpoint downloaded by these commands is tracked by this repository.

## External stress-test data

The CASIA2 split helper uses KaggleHub and writes all downloaded files under the ignored `external/downloads/` directory:

```bash
python experiments/download_external_datasets.py casia_split --project-root .
```

IMD2020 must be obtained from its official distributor. After download, build its manifest with:

```bash
python experiments/build_imd2020_manifest.py \
  --root external/downloads/IMD2020 \
  --output external/datasets/IMD2020/manifest.jsonl \
  --summary-output external/datasets/IMD2020/manifest_summary.json
```

## Minimal TD-EGFA workflow

Prepare local SID split manifests and annotation templates. This command does not modify original images:

```bash
python experiments/prepare_sid_experiment.py --sid-root SID_Set
```

Generate the frozen input requests and pixel-level evidence for a development split:

```bash
python experiments/training_free_pipeline.py make-requests \
  --sid-root SID_Set \
  --manifest SID_Set/splits/dev.jsonl \
  --per-class 100 \
  --output experiments/runs/dev_300_requests.jsonl

python experiments/training_free_pipeline.py lowlevel \
  --sid-root SID_Set \
  --manifest SID_Set/splits/dev.jsonl \
  --per-class 100 \
  --output experiments/runs/dev_300_lowlevel.jsonl
```

Run Qwen2.5-VL inference and aggregate the structured outputs:

```bash
python experiments/qwen25vl_sid_infer.py \
  --requests experiments/runs/dev_300_requests.jsonl \
  --model-path models/Qwen2.5-VL-3B-Instruct \
  --output experiments/runs/dev_300_qwen25vl_outputs.jsonl \
  --device auto \
  --max-new-tokens 512

python experiments/training_free_pipeline.py aggregate \
  --vlm-output experiments/runs/dev_300_qwen25vl_outputs.jsonl \
  --lowlevel experiments/runs/dev_300_lowlevel.jsonl \
  --output experiments/runs/dev_300_predictions.jsonl

python experiments/training_free_pipeline.py evaluate \
  --sid-root SID_Set \
  --manifest SID_Set/splits/dev.jsonl \
  --predictions experiments/runs/dev_300_predictions.jsonl \
  --output experiments/reports/dev_300_metrics.json
```

All generated requests, outputs, metrics, and reports remain local and are ignored by Git.

## Audit and analysis entry points

The frozen contract is `configs/jsg_audit_contract_v1.json`. Once the required local predictions and human references have been placed at the paths expected by the scripts, the principal audit commands are:

```bash
python experiments/build_pixel5d_audit_traces.py
python experiments/export_jsg_audit_traces.py
python experiments/joint_grounding_analysis.py
python experiments/evaluate_sida_jsg_core.py
python experiments/spatial_construct_controls.py
python experiments/criterion_relaxation_analysis.py
python experiments/paired_benchmark_contrasts.py
python experiments/joint_threshold_multivlm_analysis.py
```

These commands intentionally write only to ignored run/report directories.

## IAA examples

Each file in `annotation_examples/` contains exactly three records in this order:

1. `real`
2. `full_synthetic`
3. `tampered`

Within each task, the reference and second-pass template use the same three image IDs. See `annotation_examples/README.md` for the exact mapping. Full IAA files and completed agreement results are not distributed here.

To compute agreement after independently completing a second-pass file:

```bash
python experiments/compute_iaa_agreement.py \
  --reference path/to/reference.jsonl \
  --second-pass path/to/completed_second_pass.jsonl \
  --fields target_scope dominant_artifact_type \
  --output experiments/reports/iaa_agreement.json
```

IAA overlap samples must not be used to tune prompts, thresholds, or decision rules.

## Verification

Run the source-level tests and syntax checks without downloading models:

```bash
python -m unittest experiments.test_training_free_pipeline
python -m compileall -q experiments scripts
```
