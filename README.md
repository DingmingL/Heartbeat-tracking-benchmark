# Can Vision-Language Models Track a Heartbeat?
## A Benchmark on Frame-Level Echocardiogram Understanding

This repository contains the code for our work:

>**Can Vision-Language Models Track a Heartbeat? A Benchmark on Frame-Level Echocardiogram Understanding** \
> **MIDL 2026 (under review)**   
> **Dingming Liu et al.** 

---

## 1. Overview

We propose a benchmark for frame-level understanding of echocardiogram videos, 
focusing on EDV/ESV frame localization, segmentation overlays, and non-medical 
control tasks. We evaluate 6 open-source VLMs (Qwen3-VL 8B/32B, LLaVA-Interleave, LLaVA-NeXT-Video 7B/34B and Gemma-3n) 
and compare them against Monte Carlo random baselines.

---

## 2. Repository structure

```text
examples/   # Examples of raw result JSON files and images
prompts/    # Prompts used in benchmark
src/        # Scripts for running models and evaluations
scripts/    # Codes for annotation file generation, Monte Carlo baseline calculation and metrics caluculation/visualization.
```
---

## 3. Setup

### 3.1 Clone this repository

```bash
git clone https://github.com/DingmingL/Heartbeat-tracking-benchmark.git
cd Heartbeat-tracking-benchmark/
```
### 3.2 Create environment

```bash
conda env create -f environment.yml
conda activate <ENV_NAME>
```
---

## 4. Running the benchmark

### 4.1 Prepare data

1. Request access and download the dataset from: https://echonet.github.io/dynamic/
2. (If you want segmented version) Run segmentation using: https://github.com/echonet/dynamic.git

### 4.2 Annotation file generation

See in scripts/annotation_generation.ipynb

### 4.3 Run the benchmark

Example command:

```bash
python run_gemma-3n.py \
--model_id google/gemma-3n-e4b-it \
--video_path your/path/to/videos/ \
--annotation_file your/path/to/segmented_frame_locate_annotation.csv \
--save_path your/path/to/benchmark_results/Gemma-3n \
--video_type echo \
--task_type video \
--prompt_file prompts/prompts.yaml \
--prompt_name gemma_edv
```
---

## 5. Metrics and plots generation

See in scripts/metrics.ipynb

---

## 6. License

### 6.1 Code

The code in this repository is released under the MIT license.  
See [MIT](./LICENSE) for details.
