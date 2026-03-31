# VR Mini Project — Multi-Object Apparel Detection and Instance Segmentation

## Team

- MT2025032
- MT2025040
- MT2025081
- MT2025111

## Hugging Face repository

https://huggingface.co/3-pi-radians/VRMP1_MT2025032_MT2025040_MT2025081_MT2025111/commit/d01f4a053f6f54b8c06f4f1f669c563bd7f081b9

## Dataset

DeepFashion2 — Top-5 categories selected:

- short_sleeve_top, trousers, shorts, long_sleeve_top, skirt

## Task A: Classification

Models: ResNet-50, EfficientNet-B0, MobileNetV3
Best model: ResNet-50 (Macro-F1: 0.867)

## Task B: Detection + Segmentation

Models: YOLOv8-seg, Mask R-CNN, U-Net
Best model: YOLOv8-seg (Mask mAP@0.5: 0.907)

## Repository Structure

- preprocessing/ — data pipeline notebooks
- task_a_classification/ — classification training notebooks
- task_b_detection_segmentation/ — detection/segmentation notebooks
- inference/ — predictor.py, validator, requirements

## How to Run Inference

1. Install requirements: pip install -r inference/requirements.txt
2. Place model weights in inference/model_files/
3. Run: python inference/validator_local.py
