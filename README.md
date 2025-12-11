# Can Vision-Language Models Track a Heartbeat?
## A Benchmark on Frame-Level Echocardiogram Understanding

This repository contains the code for our work:

> **Title.** Can Vision-Language Models Track a Heartbeat? A Benchmark on Frame-Level Echocardiogram Understanding
> **Venue.** MIDL 2026 (under review)  
> **Authors.** Dingming Liu et al.

---

## 1. Overview

We propose a benchmark for frame-level understanding of echocardiogram videos, 
focusing on EDV/ESV frame localization, segmentation overlays, and non-medical 
control tasks. We evaluate 6 open-source VLMs (Qwen3-VL 8B/32B, LLaVA-Interleave, LLaVA-NeXT-Video 7B/34B and Gemma-3n) 
and compare them against Monte Carlo random baselines.

---

## 2. Repository structure

```text
src/        # Scripts for running models and evaluations
scripts/    # Codes for result visualization and metrics caluculation
