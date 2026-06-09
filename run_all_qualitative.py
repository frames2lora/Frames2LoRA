import os
import sys
import urllib.request
from pathlib import Path
import torch

# Add repository root and src directory to sys.path to enable local imports
repo_root = str(Path(__file__).parent.resolve())
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
src_dir = str(Path(repo_root) / "src")
if os.path.exists(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dataclasses import dataclass
from typing import List, Optional
from huggingface_hub import hf_hub_download

# Import core modules
from ctx_to_lora.modeling.lora_layer import apply_lora_to_layers
from ctx_to_lora.modeling.lora_merger import combine_lora
from scripts.video2lora.train_smolvlm_stage1 import build_stage1_model
from scripts.video2lora.train_smolvlm_online import (
    prepare_smolvlm_inputs,
    extract_l2l_fused_text_features,
)

@dataclass
class TrainArgs:
    smolvlm_name_or_path: str
    train_manifest: str = ""
    val_manifest: str = ""
    output_dir: str = ""
    lora_r: int = 16
    lora_dropout: float = 0.0
    target_modules: Optional[List[str]] = None
    latent_size: int = 512
    dropout_rate: float = 0.0
    n_latent_queries: int = 8
    num_blocks: int = 9
    num_self_attn_per_block: int = 0
    video_fps: Optional[float] = None
    max_frames: int = 12
    video_size_longest_edge: int = 384
    video_load_backend: str = "auto"
    internalization_prompt: str = "Internalize this video for later captioning."
    kl_weight: float = 0.0
    generation_max_new_tokens: int = 128

# 1. Qualitative Examples Database (from video2lora.github.io/script.js)
examples = [
    {
        "videoId": "media/benchmarks/carebench/v_00014063_0.mp4",
        "question": "Describe the video in as much useful visual detail as possible. Include the main activity, visible people or objects, scene context, appearance, and any important visual details that help explain what is happening.",
        "reference": "This video depicts a scene of a man lighting a pipe with a lighter. The man in the video is smoking a pipe held in his mouth, supported by his left hand, while his right hand grips the lighter. His right forearm features a large black tattoo. He has short, thick hair that is a deep brown color and is dressed in a loose-fitting black tank top. He is seated next to a window, which has a wooden frame and blue curtains, with a brick wall behind him and a wooden door on the right. The door has a square pattern, adorned with silver hinges and a doorknob. In the video, he first ignites the lighter with his right hand and then brings the flame to the pipe, holding it in that position for several seconds. Throughout this time, his left hand remains steady on the pipe, and his gaze is fixed intently on it, ensuring that the pipe is fully lit before setting the lighter down. He then continues to hold the pipe with his left hand and begins to smoke. The video is shot from the front, clearly illustrating how relaxed he is while smoking at home.",
        "source": "CaReBench: Caption"
    },
    {
        "videoId": "media/benchmarks/carebench/v_00016555_0.mp4",
        "question": "Describe the key visible events in chronological order. Include all important actions and changes you can observe, with enough detail to distinguish each event clearly.",
        "reference": "Little boy watering plants outdoors; Using watering can to pour water into flower pot; Shifting camera angle from side view to rear view; Tapping edge of flower pot a few times; Setting down watering can",
        "source": "CaReBench: Events"
    },
    {
        "videoId": "media/benchmarks/plm/f522598789220c70_122_155.mp4",
        "question": "Does this look like the same posture she's holding?",
        "reference": "Yes, it appears you're mirroring the same posture. Your alignment, knee bend, and spine position match the demonstration, indicating proper form and engagement of the targeted muscle groups for optimal effectiveness and safety.",
        "source": "PLM-SGQA"
    },
    {
        "videoId": "media/benchmarks/plm/b5bdb7f254cb1727_369_400.mp4",
        "question": "Is he trying to tug?",
        "reference": "Yes, your dog is likely inviting a tug-of-war game. Holding the toy in his mouth and possibly looking at you or wagging his tail indicates he's ready to engage in a playful tug.",
        "source": "PLM-SGQA"
    },
    {
        "videoId": "media/benchmarks/vidcapbench/132065802449.mp4",
        "question": "What is the weather like in the scene? Answer only the question, in one sentence.",
        "reference": "Rainy day.",
        "source": "VidCapBench"
    },
    {
        "videoId": "media/benchmarks/vidcapbench/Tarsier_20.mp4",
        "question": "Which parts of the creature are highlighted in the video? Answer only the question, in one sentence.",
        "reference": "A close-up of its face, eyes, and hair.",
        "source": "VidCapBench"
    }
]

def main():
    print("=== Video2LoRA: Qualitative Examples Inference Script ===")
    
    # Download the qualitative videos if not present
    print("\n--- Downloading qualitative example videos ---")
    for item in examples:
        local_path = item["videoId"]
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            url = f"https://video2lora.github.io/{item['videoId']}"
            print(f"Downloading {local_path} from {url}...")
            try:
                urllib.request.urlretrieve(url, local_path)
                print(f"Saved to {local_path}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")
        else:
            print(f"Found local video file: {local_path}")

    # Download hypernetwork weights
    print("\n--- Downloading 2.2B Video2LoRA checkpoint ---")
    checkpoint_dir = Path("checkpoints/Video2LoRA-SmolVLM-ckpts")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        ckpt_path = hf_hub_download(
            repo_id="MananSuri27/Video2LoRA-SmolVLM-ckpts",
            filename="video2lora-smolvlm2-2.2b-best-ce.pt",
            local_dir=str(checkpoint_dir)
        )
        print(f"Checkpoint ready at: {ckpt_path}")
    except Exception as e:
        print(f"Error downloading checkpoint: {e}")
        return

    # Device selection (support MPS for Mac acceleration)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\nTargeting device: {device}")

    # Model parameters matching 2.2B SmolVLM2 preset
    train_args = TrainArgs(
        smolvlm_name_or_path="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        target_modules=["down_proj"]
    )

    print("\nLoading SmolVLM2 base model and initializing modulated structure...")
    try:
        model, raw_model, processor, tokenizer = build_stage1_model(train_args, device=device)
        
        print("Loading hypernetwork state dictionary...")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        raw_model.eval()
        print("Model initialization successful!")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Loop and run inference for all examples
    for idx, item in enumerate(examples):
        print(f"\n==================================================")
        print(f"EXAMPLE {idx+1}/{len(examples)}: {item['source']}")
        print(f"==================================================")
        print(f"Video: {item['videoId']}")
        print(f"Question: {item['question']}")
        print(f"Ground Truth: {item['reference']}\n")

        # 1. Perform internalization forward pass
        internalize_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": item["videoId"]},
                        {"type": "text", "text": train_args.internalization_prompt}
                    ]
                }
            ]
        ]
        
        with torch.no_grad():
            vlm_inputs = prepare_smolvlm_inputs(
                processor,
                internalize_messages,
                device,
                video_fps=train_args.video_fps,
                max_frames=train_args.max_frames,
                video_size_longest_edge=train_args.video_size_longest_edge,
                video_load_backend=train_args.video_load_backend
            )
            
            ctx_features, ctx_attn_mask, ctx_position_ids = extract_l2l_fused_text_features(
                raw_model,
                vlm_inputs,
                num_target_layers=model.hypernet.n_layers
            )
            
            generated_loras, _ = model.generate_weights(
                ctx_ids=None,
                ctx_features=ctx_features,
                ctx_attn_mask=ctx_attn_mask,
                ctx_position_ids=ctx_position_ids
            )
            
            generated_loras = combine_lora(
                generated_loras,
                torch.ones(1, dtype=torch.int32, device=device),
                lora_bias=model.hypernet.get_head_bias() if model.hypernet.config.use_bias else None
            )

        # 2. Run Base Model (with visual tokens in context)
        base_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": item["videoId"]},
                        {"type": "text", "text": item["question"]}
                    ]
                }
            ]
        ]
        
        with torch.no_grad():
            base_inputs = prepare_smolvlm_inputs(
                processor,
                base_messages,
                device,
                video_fps=train_args.video_fps,
                max_frames=train_args.max_frames,
                video_size_longest_edge=train_args.video_size_longest_edge,
                video_load_backend=train_args.video_load_backend
            )
            
            base_generated_ids = raw_model.generate(
                **base_inputs,
                max_new_tokens=train_args.generation_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            base_prediction = tokenizer.decode(
                base_generated_ids[0][base_inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

        # 3. Run Video2LoRA (zero visual tokens in context, using generated LoRA)
        apply_lora_to_layers(
            model.base_model,
            model.hypernet.layer_indices,
            generated_loras,
            torch.ones(1, dtype=torch.int32, device=device),
            position_ids=None
        )

        prompt_ids = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": item["question"]}]
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.base_model.generate(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=train_args.generation_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            video2lora_prediction = tokenizer.decode(
                generated_ids[0][prompt_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

        # Reset LoRA hooks
        model.reset()

        print(f"--> Base Model: {base_prediction}")
        print(f"--> Video2LoRA: {video2lora_prediction}")

if __name__ == "__main__":
    main()
