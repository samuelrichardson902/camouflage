# Naturalistic Adversarial Camouflage for Vehicles

This repository contains the source code for a style-constrained adversarial optimization pipeline. The project demonstrates how Neural Style Transfer (NST) can be used to hide adversarial perturbations within semantically plausible textures (e.g., mud) to evade both AI object detectors and human suspicion.

## Project Overview
Traditional physical adversarial attacks often rely on conspicuous geometric noise. This framework integrates a neural renderer into a differentiable pipeline to optimize textures that:
1.  Blind AI: Reduce YOLOv8 detection confidence.
2.  Evade Humans: Appear as normal weathering (mud) using NST constraints.


## Repository Structure
- `naturalBatchTrainer.ipynb`: Core optimization script.
- `/models`: Pre-trained neural renderer.
- `/natural_checkpoints & noise_checkpoints`: The generated adversarial patches at different epochs in their optimisation.
- `eval.ipynb`: Data analysis, and statistical testing (ANOVA/T-Tests).
- `/textures`: Styling references (mud/dirt).

## Requirements
- Python 3.7.16
- Tensorflow 2.11.0
- Keras 2.11.0
- Numpy 1.21.6
- Ultralytics (YOLOv8)
- OpenCV, Matplotlib, Seaborn
- carla version 0.9.15

## Results Summary
- **Upper Bound (Noise):** 82.11% ASR
- **Naturalistic (Mud):** 60.64% ASR (Cohen's $d = 1.67$)
- **Cost of Naturalness:** 21.47% reduction in efficacy for perceptual stealth.
