import os
import json
import time
import glob
import random
import numpy as np
import torch
import av
import re
import pandas as pd
import argparse
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from utils import out_pattern, read_segmented_video_pyav, set_seed, parse_args, read_video_pyav, load_frame_annotations, \
    get_frame_range_for_video, load_prompt, render_prompt, get_target_indices, read_frames_pyav, read_segmented_frames_pyav

    
def load_qwen_model(model_id = "Qwen/Qwen3-VL-8B-Instruct", ban_tokens=["<think>", "</think>"]):

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.float16,
        # attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    ).eval()
    
    if ban_tokens is not None:
        ban_ids = [tid for t in ban_tokens
                   if (tid := processor.tokenizer.convert_tokens_to_ids(t)) is not None]
        bad_words_ids = [[tid] for tid in ban_ids] if ban_ids else None
        return model, processor, bad_words_ids
    else:
        return model, processor, None
    
def infer_one_video(frames, processor, model, task_type, bad_words_ids=None, prompt_name="structured_esv_edv", idx_low=57, idx_high=80, prompt_file="/home/dili10/scripts/vlm_benchmark/prompts.yaml"):
    
    if task_type == "video":

        NUM = len(frames)
        raw_cfg = load_prompt(prompt_name, prompt_file=prompt_file)

        if isinstance(raw_cfg, dict) and "conversation" in raw_cfg:
            conv_template = raw_cfg["conversation"]
        else:
            conv_template = raw_cfg

        conversation = render_prompt(conv_template, NUM=NUM, idx_low=idx_low, idx_high=idx_high)

    elif task_type == "frame":

        raw_cfg = load_prompt(prompt_name, prompt_file=prompt_file)

        if isinstance(raw_cfg, dict) and "conversation" in raw_cfg:
            conv_template = raw_cfg["conversation"]
        else:
            conv_template = raw_cfg

        conversation = render_prompt(conv_template)
    
    prompt = processor.apply_chat_template(
        conversation, 
        add_generation_prompt=True)

    with torch.inference_mode():

        if task_type == "video":
            inputs = processor(
                text=prompt,
                videos=[frames],
                return_tensors="pt"
            ).to(model.device)
        elif task_type == "frame":
            inputs = processor(
                text=prompt,
                images=[frames[0], frames[1]],
                return_tensors="pt"
            ).to(model.device)

        out = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=30,
            bad_words_ids=bad_words_ids
        )

    start = inputs["input_ids"].shape[1]
    text = processor.decode(out[0][start:], skip_special_tokens=True).strip()
    return text


def main():

    args = parse_args()
    model_id = args.model_id
    video_path = args.video_path
    annotation_file = args.annotation_file
    save_path = args.save_path
    video_type = args.video_type
    prompt_name = args.prompt_name
    task = args.task_type
    prompt_file = args.prompt_file

    prompt_end = prompt_name.split('_', 1)[-1]

    model_name = model_id.split("/")[-1]
    video_dir = video_path
    annotation_file = annotation_file
    # save_path = os.path.join(save_path, f"results_{model_name}_{video_type}_{prompt_end}_{task}_random_seed_{seed}.jsonl")
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    OUT_PAT = out_pattern(prompt_end=prompt_end)

    # all_videos = sorted(glob.glob(os.path.join(video_dir, "*.avi")))
    # videos = all_videos[:100]

    # Load annotations
    annotations_df = load_frame_annotations(annotation_file)

    video_names = (
            annotations_df["FileName"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

    videos = video_names[:100]

    model, processor, bad_words_ids = load_qwen_model(
        model_id = model_id, 
        ban_tokens=["<think>", "</think>"])
        
    for i in range(10):
        seed = 2020 + i
        set_seed(seed)

        save_path_file = os.path.join(save_path, f"results_{model_name}_{video_type}_{prompt_end}_{task}_random_seed_{seed}.jsonl")
        os.makedirs(os.path.dirname(save_path_file), exist_ok=True)

        with open(save_path_file, "w", encoding="utf-8") as f:
            for vid_idx, vn in enumerate(videos):
                
                vp = os.path.join(video_dir, vn)
                container = av.open(vp)
                total_frames = container.streams.video[0].frames

                if task == "video":
                    if annotations_df is not None:                  
                        idx_low, idx_high, edv_idx, esv_idx, start_idx = get_frame_range_for_video(vp, annotations_df, total_frames=total_frames)
                    else:
                        # suppose here 50 frames contains at least one cardiac cycle
                        idx_low, idx_high, edv_idx, esv_idx, start_idx = 0, max(int(container.streams.video[0].frames), 50), None, None, 0

                    NUM = int(idx_high - idx_low) + 1
                    indices = np.arange(idx_low+start_idx, start_idx+idx_high + 1).astype(int)

                    if video_type == "echo":
                        frames = read_video_pyav(container, indices)
                    elif video_type == "segmented_echo":
                        frames = read_segmented_video_pyav(container, indices)


                    for run_id in range(1):
                        # set_seed(2025)

                        t0 = time.time()
                        out_text = infer_one_video(
                            frames=frames, 
                            processor=processor, 
                            model=model, 
                            task_type=task,
                            bad_words_ids=bad_words_ids, 
                            idx_low=idx_low, 
                            idx_high=idx_high,
                            prompt_name=prompt_name,
                            prompt_file=prompt_file
                        )
                        #print("Output text:", out_text)
                        dt = time.time() - t0

                        m = OUT_PAT.search(out_text or "")
                        if m:
                            if prompt_end == "esv_edv":
                                pred_esv = int(m.group(1))
                                pred_edv = int(m.group(2))
                            elif prompt_end == "edv_esv":
                                pred_edv = int(m.group(1))
                                pred_esv = int(m.group(2))
                            elif prompt_end == "edv":
                                pred_edv = int(m.group(1))
                                pred_esv = None
                            elif prompt_end == "esv":
                                pred_esv = int(m.group(1))
                                pred_edv = None
                        else:
                            pred_edv = None
                            pred_esv = None

                        rec = {
                            "video_index": vid_idx,
                            "video_path": vp,
                            "run_id": run_id,
                            "model_name": model_name,
                            "prompt_name": prompt_name,
                            "num_frames": NUM,
                            "idx_range": [idx_low, idx_high],

                            "start_idx": start_idx,
                            "gt_edv": int(edv_idx) if edv_idx is not None else None,
                            "gt_esv": int(esv_idx) if esv_idx is not None else None,
                            "pred_edv": pred_edv,
                            "pred_esv": pred_esv,

                            "output": out_text,
                            "latency_sec": round(dt, 4)
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    
                elif task == "frame":
                    if annotations_df is not None:
                        edv_idx, esv_idx = get_target_indices(annotations_df, vp)
                    else:
                        raise ValueError("Frame-level task requires annotations.")
                    
                    if video_type == "echo":
                        frames, edv_pos, esv_pos = read_frames_pyav(container, edv_idx, esv_idx)
                    elif video_type == "segmented_echo":
                        frames, edv_pos, esv_pos = read_segmented_frames_pyav(container, edv_idx, esv_idx)

                    for run_id in range(1):
                        # set_seed(2025)

                        t0 = time.time()
                        out_text = infer_one_video(
                            frames=frames, 
                            processor=processor, 
                            model=model,
                            bad_words_ids=bad_words_ids,
                            task_type=task,
                            prompt_name=prompt_name,
                            prompt_file=prompt_file
                        )

                        dt = time.time() - t0

                        m = OUT_PAT.search(out_text or "")

                        if m:
                            if prompt_end == "edv":
                                pred_edv = int(m.group(1))
                                pred_esv = None

                            elif prompt_end == "esv":
                                pred_esv = int(m.group(1))
                                pred_edv = None
                        else:
                            pred_edv = None
                            pred_esv = None

                        rec = {
                            "video_index": vid_idx,
                            "video_path": vp,
                            "run_id": run_id,
                            "model_name": model_name,
                            "prompt_name": prompt_name,

                            "gt_edv": edv_pos,
                            "gt_esv": esv_pos,
                            "pred_edv": pred_edv,
                            "pred_esv": pred_esv,

                            "output": out_text,
                            "latency_sec": round(dt, 4)
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    # print(f"[{vid_idx:03d}] run {run_id} -> {out_text}  ({dt:.2f}s)")

if __name__ == "__main__":
    main()
