import os
import random
import numpy as np
import torch
import av
import re
import pandas as pd
import argparse
import yaml
from jinja2 import Template

def set_seed(s=2025):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)   

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True,
                        help="Model ID.")
    parser.add_argument("--annotation_file", type=str, default=None,
                        help="CSV file containing frame annotations.")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the video file.")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save the results JSONL file.")
    parser.add_argument("--video_type", type=str, default="echo",
                        help="Type of the video files (e.g., avi, mp4).")
    parser.add_argument("--prompt_name", type=str, default="structured_esv_edv",
                        help="Name of the prompt to use from prompts.yaml.")
    parser.add_argument("--task_type", type=str, default="video",
                        help="Type of the task (e.g., video, frame).")
    parser.add_argument("--prompt_file", type=str, default="/home/dili10/scripts/vlm_benchmark/prompts.yaml",
                        help="Path to the prompt YAML file.")
    return parser.parse_args()

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)

    # print(f"Frames decoded: {len(frames)} for indices from {start_index} to {end_index}")
    
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def read_frames_pyav(container, edv_idx, esv_idx, shuffle=True):
    """
    Read exactly two specific frames (EDV and ESV) from the video using PyAV.

    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        edv_idx (int): Frame index of EDV   (largest LV cavity).
        esv_idx (int): Frame index of ESV   (smallest LV cavity).
        shuffle (bool): If True, randomly shuffle the order of the two
                        frames when returning them.

    Returns:
        frames (np.ndarray): Array of two frames with shape (2, H, W, 3).
                             frames[0] -> model frame index 0
                             frames[1] -> model frame index 1
        edv_pos (int): Position of EDV in `frames`, either 0 or 1.
        esv_pos (int): Position of ESV in `frames`, either 0 or 1.
    """

    target_indices = {edv_idx, esv_idx}
    max_idx = max(target_indices)

    container.seek(0)
    frames_dict = {}

    for i, frame in enumerate(container.decode(video=0)):
        if i > max_idx:
            break

        if i in target_indices:
            frames_dict[i] = frame.to_ndarray(format="rgb24")

            if len(frames_dict) == 2:
                break

    # original order
    order = [edv_idx, esv_idx]

    # random shuffle the order of the two frames
    if shuffle:
        random.shuffle(order) 

    frame0 = frames_dict[order[0]]
    frame1 = frames_dict[order[1]]
    frames = np.stack([frame0, frame1], axis=0)  # (2, H, W, 3)

    # Calculate the positions of EDV and ESV in the returned array
    edv_pos = order.index(edv_idx)  # 0 or 1
    esv_pos = order.index(esv_idx)  # 0 or 1

    return frames, edv_pos, esv_pos


def load_frame_annotations(csv_path):
    """Load frame annotations from CSV."""
    df = pd.read_csv(csv_path)
    return df

def get_target_indices(annotations_df, video_path):
    """Get target frame indices from annotations DataFrame."""
    video_name = os.path.basename(video_path)
    video_data = annotations_df[annotations_df['FileName'] == video_name]
    edv_idx = video_data.loc[video_data['Label'] == 'EDV', 'Frame'].iloc[0]
    esv_idx = video_data.loc[video_data['Label'] == 'ESV', 'Frame'].iloc[0]
    return int(edv_idx), int(esv_idx)

def get_frame_range_for_video(video_path, annotations_df, total_frames):
    """
    Extract idx_low and idx_high for a specific video.
    Returns the frame range [idx_low, idx_high] based on EDV and ESV frames.
    """
    video_name = os.path.basename(video_path)
    video_data = annotations_df[annotations_df['FileName'] == video_name]

    if video_data.empty:
        raise ValueError(f"Warning: {video_name} not found in annotations")

    edv_series = video_data.loc[video_data['Label'] == 'EDV', 'Frame']
    esv_series = video_data.loc[video_data['Label'] == 'ESV', 'Frame']

    if not edv_series.empty and not esv_series.empty:
        edv = int(edv_series.iloc[0])
        esv = int(esv_series.iloc[0])

        # extend range by one third of the cardiac cycle on each side
        one_third_cycle_len = (abs(edv - esv) * 2) // 3

        # recenter frame indices starting from 0
        start_idx = min(edv, esv) - one_third_cycle_len 
        
        if start_idx < 0:
            start_idx = 0

        new_edv = edv - start_idx
        new_esv = esv - start_idx
        if (max(new_edv, new_esv) + one_third_cycle_len) < total_frames:
            idx_high = max(new_edv, new_esv) + one_third_cycle_len
        else:
            idx_high = total_frames - 1
        return 0, int(idx_high), new_edv, new_esv, start_idx
    else:
        raise ValueError(f"Warning: EDV or ESV frame not found for {video_name}")
    
def load_prompt(prompt_name, prompt_file="/home/dili10/scripts/vlm_benchmark/prompts.yaml"):
    """Load prompt from YAML file."""
    
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    if prompt_name not in prompts:
        raise ValueError(f"Prompt {prompt_name} not found in prompts.yaml")
    return prompts[prompt_name]



def render_prompt(prompt_text, **kwargs):
    """Recursively render only `text` fields inside the conversation structure."""
    if isinstance(prompt_text, dict):
        new_node = {}
        for k, v in prompt_text.items():
            if k == "text" and isinstance(v, str):
                # Only render Jinja template inside text fields
                new_node[k] = Template(v).render(**kwargs)
            else:
                new_node[k] = render_prompt(v, **kwargs)
        return new_node

    elif isinstance(prompt_text, list):
        return [render_prompt(x, **kwargs) for x in prompt_text]
    else:
        return prompt_text
    

def out_pattern(prompt_end="esv_edv"):
    if prompt_end == "edv_esv":
        return re.compile(r"EDV\s*=\s*(-?\d+)\s*,\s*ESV\s*=\s*(-?\d+)", re.IGNORECASE)
    elif prompt_end == "esv_edv":
        return re.compile(r"ESV\s*=\s*(-?\d+)\s*,\s*EDV\s*=\s*(-?\d+)", re.IGNORECASE)
    elif prompt_end == "edv":
        return re.compile(r"EDV\s*=\s*(-?\d+)", re.IGNORECASE)
    elif prompt_end == "esv":
        return re.compile(r"ESV\s*=\s*(-?\d+)", re.IGNORECASE)
    

def read_segmented_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded cropped frames of shape (num_frames, 112, 112, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]

    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))

    # crop only video with segmentation
    cropped_frames = [img[0:112, 112:224, :] for img in frames]

    return np.stack(cropped_frames)


def read_segmented_frames_pyav(container, edv_idx, esv_idx, shuffle=True):
    """
    Read exactly two specific frames (EDV and ESV) from the video using PyAV.

    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        edv_idx (int): Frame index of EDV   (largest LV cavity).
        esv_idx (int): Frame index of ESV   (smallest LV cavity).
        shuffle (bool): If True, randomly shuffle the order of the two
                        frames when returning them.

    Returns:
        frames (np.ndarray): Array of two frames with shape (2, H, W, 3).
                             frames[0] -> model frame index 0
                             frames[1] -> model frame index 1
        edv_pos (int): Position of EDV in `frames`, either 0 or 1.
        esv_pos (int): Position of ESV in `frames`, either 0 or 1.
    """

    target_indices = {edv_idx, esv_idx}
    max_idx = max(target_indices)

    container.seek(0)
    frames_dict = {}

    for i, frame in enumerate(container.decode(video=0)):
        if i > max_idx:
            break

        if i in target_indices:
            img = frame.to_ndarray(format="rgb24")

            cropped = img[0:112, 112:224, :]   # (112,112,3)
            frames_dict[i] = cropped

            if len(frames_dict) == 2:
                break

    # original order
    order = [edv_idx, esv_idx]

    # random shuffle the order of the two frames
    if shuffle:
        random.shuffle(order) 

    frame0 = frames_dict[order[0]]
    frame1 = frames_dict[order[1]]
    frames = np.stack([frame0, frame1], axis=0)  # (2, H, W, 3)

    # Calculate the positions of EDV and ESV in the returned array
    edv_pos = order.index(edv_idx)  # 0 or 1
    esv_pos = order.index(esv_idx)  # 0 or 1

    return frames, edv_pos, esv_pos