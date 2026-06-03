# Video2LoRA

Video2LoRA trains a hypernetwork that converts a video into LoRA weights for a
frozen vision-language model. The generated adapter lets the model answer later
text prompts without feeding the video tokens again.

<p align="center">
  <img src="assets/video2lora-animated-diagram.svg" alt="Animated Video2LoRA method diagram" width="900">
</p>

## Install

Install `uv`.

```bash
git clone https://github.com/MananSuri27/video2lora.git
cd video2lora
uv sync
```

For local video path resolution, set:

```bash
export VIDEO2LORA_DATA_ROOT=$PWD/data/video2lora
```

If unset, `VIDEO2LORA_DATA_ROOT` defaults to `data/video2lora`.

## Checkpoints

Download the released checkpoints from Hugging Face:

```bash
uv run huggingface-cli download MananSuri27/Video2LoRA-SmolVLM-ckpts \
  --local-dir checkpoints/Video2LoRA-SmolVLM-ckpts
```

The repo contains:

```text
checkpoints/Video2LoRA-SmolVLM-ckpts/
  video2lora-smolvlm2-500m-best-ce.pt
  video2lora-smolvlm2-2.2b-best-ce.pt
```


## Inference

Create a JSONL manifest with one row per video. For example:

```json
{"id":"sample-0001","video_path":"/path/to/video.mp4","prompt":"Describe what is happening in this video.","task_type":"caption"}
```

```bash
uv run python -m scripts.video2lora.infer \
  --checkpoint checkpoints/Video2LoRA-SmolVLM-ckpts/video2lora-smolvlm2-500m-best-ce.pt \
  --manifest /path/to/manifest.jsonl \
  --output outputs/tiny_generations.jsonl
```

For the 2.2B checkpoint, change `--checkpoint` to:

```bash
checkpoints/Video2LoRA-SmolVLM-ckpts/video2lora-smolvlm2-2.2b-best-ce.pt
```

The output JSONL includes the original row fields plus `prediction` and
`checkpoint`.

## Data Format

Training and inference use JSONL manifests. Each line is one example:

```json
{
  "id": "sample-0001",
  "video_path": "raw/finevideo/sample.mp4",
  "task_type": "caption",
  "prompt": "Describe what is happening in this video.",
  "target_text": "A person is cooking.",
  "dataset": "finevideo",
  "split": "train",
  "metadata": {}
}
```

Relative `video_path` values are resolved against `VIDEO2LORA_DATA_ROOT`.
Absolute paths are used as-is.

## Generate Teacher Data

The final CE training recipe uses cached teacher-generated targets. Starting
from readable manifests, generate teacher targets with:

```bash
export VIDEO2LORA_DATA_ROOT=$PWD/data/video2lora

CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.video2lora.generate_finevideo_teacher_targets \
  --input-manifest $VIDEO2LORA_DATA_ROOT/processed/finevideo/train.jsonl \
  --output-manifest $VIDEO2LORA_DATA_ROOT/processed/finevideo/train.teacher_visual_smolvlm.jsonl \
  --smolvlm-name-or-path HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --per-device-batch-size 8 \
  --max-frames 12 \
  --video-size-longest-edge 384
```

To shard generation manually, run the same command per GPU with
`--num-shards N --shard-index R`, then merge with `--merge-only`.

## Train

The main CE training entrypoint is:

```bash
uv run accelerate launch \
  --config_file accelerate_config.yaml \
  --num_processes 4 \
  -m scripts.video2lora.train_smolvlm_stage1 \
  --smolvlm-name-or-path HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --train-manifest data/video2lora/processed/finevideo/train.teacher_visual_smolvlm.jsonl \
  --val-manifest data/video2lora/processed/finevideo/val.teacher_visual_smolvlm.jsonl \
  --val-core-manifest data/video2lora/processed/finevideo/val.teacher_visual_smolvlm.jsonl \
  --output-dir runs/video2lora-smoke \
  --per-device-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --max-steps 1000 \
  --target-modules down_proj \
  --lora-r 16 \
  --latent-size 512 \
  --n-latent-queries 8 \
  --num-blocks 9 \
  --max-frames 12 \
  --video-size-longest-edge 384 \
  --kl-weight 0.0 \
  --wandb-mode disabled
```

The self-distillation variant is also included:

```bash
uv run accelerate launch \
  --config_file accelerate_config.yaml \
  --num_processes 4 \
  -m scripts.video2lora.train_smolvlm_stage1_selfdistill \
  --smolvlm-name-or-path HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --train-manifest data/video2lora/processed/finevideo/train.jsonl \
  --val-manifest data/video2lora/processed/finevideo/val.jsonl \
  --val-gen-manifest data/video2lora/processed/finevideo/val_gen_100.jsonl \
  --output-dir runs/video2lora-selfdistill \
  --wandb-mode disabled
```

## Build FineVideo Manifests

If you have raw FineVideo metadata/videos laid out under
`$VIDEO2LORA_DATA_ROOT/raw/finevideo`, build Stage 1 manifests with:

```bash
uv run python -m scripts.video2lora.build_finevideo_stage1_manifest
```
