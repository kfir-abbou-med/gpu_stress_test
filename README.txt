# GPU Stress Test Processor - README

## Overview

The **GPU Stress Test Processor** is a Python-based benchmark tool for evaluating GPU performance through advanced image processing and deep learning models. Key algorithms include:

1. **Semantic Segmentation** (DeepLabV3)
2. **Object Detection** (Faster R-CNN)
3. **Feature Extraction** (ResNet50)

This tool leverages AI workloads to stress-test GPUs and provides a visual assessment of performance.

## Features
 
- **Batch Video Processing**: Efficiently processes frames in batches.
- **AI Model Integration**: Pretrained models for computer vision tasks.
- **Performance Insights**: Reports processing speed and FPS.
- **Enhanced Visualization**: Outputs videos with segmentation masks and bounding boxes.

## Requirements

### Hardware

- CUDA-enabled GPU (e.g., NVIDIA GPUs).

### Software

- Python 3.7+
- PyTorch 1.9+ (with CUDA support)
- OpenCV 4.5+
- torchvision
- NumPy

## Benchmark Workflow

1. **Initialization**:
   - Identifies GPU and reports VRAM.
   - Loads pretrained models (DeepLabV3, Faster R-CNN, ResNet50).
2. **Frame Preprocessing**:
   - Converts frames to 512x512 tensors for model compatibility.
3. **AI Inference**:
   - Runs segmentation, object detection, and feature extraction.
   - Creates overlays for segmentation masks and detections.
4. **Output Video**:
   - Generates a processed video with visualized results.

### Key Outputs

- **Visual Enhancements**:
  - Segmentation overlays and bounding boxes.
  - Feature vector data (e.g., mean value).
- **Performance Metrics**:
  - Total processing time.
  - Average FPS.

## Code Overview

### Core Components

- **`GPUStressTestProcessor`**:
  - Orchestrates video processing, AI inference, and result visualization.
- **`process_frame_batch(frames)`**:
  - Processes frames in batches using all models.
- **`process_video(video_path, output_path)`**:
  - Handles video input/output and frame buffering.

### AI Models

- **DeepLabV3**:
  - Provides semantic segmentation.
- **Faster R-CNN**:
  - Detects objects and generates bounding boxes.
- **ResNet50**:
  - Extracts feature vectors.

## Performance Insights

The benchmark reports:

- **Total Time**: Duration to process the input video.
- **Average FPS**: Frames processed per second, indicating GPU efficiency.
