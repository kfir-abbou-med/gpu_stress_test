import os
import sys
import cv2
import torch
import numpy as np
import time
from torchvision import transforms, models
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import autocast

class GPUStressTestProcessor:
    def __init__(self):
        print("\n=== Initializing Video Processor ===")
        print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_models()
        self.transform = self.get_transform()
        self.frame_buffer = []
        self.buffer_size = 16
        self.total_frames_processed = 0
        self.start_time = None

    def get_transform(self):
        """
        Creates a composition of image transformations for preprocessing.
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def initialize_models(self):
        print("\nLoading AI models...")
        print("Loading segmentation model (DeepLabV3)...")
        self.segmentation_model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(self.device)
        print("Loading feature extraction model (ResNet50)...")
        self.resnet = models.resnet50(pretrained=True).to(self.device)
        print("Loading object detection model (Faster R-CNN)...")
        self.detection_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(self.device)
        
        self.segmentation_model.eval()
        self.resnet.eval()
        self.detection_model.eval()
        print("All models loaded successfully!")

    @torch.cuda.amp.autocast()
    def process_frame_batch(self, frames):
        print("\n--- Processing Batch ---")
        batch_start = time.time()
        
        # Convert frames to tensors
        frame_tensors = torch.stack([self.transform(frame) for frame in frames]).to(self.device)

        # 1. Semantic Segmentation (DeepLabV3)
        print("Running semantic segmentation...")
        with torch.no_grad():
            segmentation_output = self.segmentation_model(frame_tensors)['out']
            segmentation_masks = segmentation_output.argmax(dim=1).cpu().numpy()
        
        # 2. Object Detection (Faster R-CNN)
        print("Running object detection...")
        with torch.no_grad():
            detection_results = self.detection_model(frame_tensors)
        
        # 3. Feature Extraction (ResNet50)
        print("Running feature extraction...")
        with torch.no_grad():
            feature_vectors = self.resnet(frame_tensors)
        
        # 4. Visualization and Combination
        processed_frames = []
        for i, frame in enumerate(frames):
            # Add segmentation mask as a heatmap
            mask = segmentation_masks[i].astype(np.uint8)
            mask_colored = cv2.applyColorMap(mask * 15, cv2.COLORMAP_JET)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Resize the segmentation mask to match the original frame's dimensions
            mask_colored_resized = cv2.resize(mask_colored, (frame_bgr.shape[1], frame_bgr.shape[0]))

            # Overlay the resized mask on the original frame
            overlaid = cv2.addWeighted(frame_bgr, 0.6, mask_colored_resized, 0.4, 0)

            # Add detection bounding boxes
            for detection in detection_results[i]['boxes']:
                box = detection.int().cpu().numpy()
                cv2.rectangle(overlaid, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            
            # Add feature vector info (placeholder text)
            text = f"Feature Vec Mean: {feature_vectors[i].mean().item():.2f}"
            cv2.putText(overlaid, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            processed_frames.append(overlaid)

        total_batch_time = time.time() - batch_start
        print(f"\nBatch processing completed in {total_batch_time:.2f}s")
        print(f"Average time per frame: {total_batch_time/len(frames):.2f}s")
        
        return processed_frames

    def process_video(self, video_path, output_path):
        try:
            print(f"\n=== Starting Video Processing ===")
            print(f"Input video: {video_path}")
            print(f"Output video: {output_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Error opening video file")

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"\nVideo Properties:")
            print(f"Total frames: {total_frames}")
            print(f"FPS: {fps}")
            print(f"Resolution: {frame_width}x{frame_height}")
            print(f"Duration: {total_frames/fps:.2f} seconds")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            self.start_time = time.time()
            while cap.isOpened():
                while len(self.frame_buffer) < self.buffer_size:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    # Convert BGR to RGB and add to buffer
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.frame_buffer.append(frame_rgb)

                if len(self.frame_buffer) == 0:
                    break

                self.total_frames_processed += len(self.frame_buffer)
                processed_batch = self.process_frame_batch(self.frame_buffer)

                for processed_frame in processed_batch:
                    out.write(processed_frame)

                self.frame_buffer = []

        except Exception as e:
            print(f"Error: {e}")
        finally:
            cap.release()
            out.release()

        total_time = time.time() - self.start_time
        print(f"\n=== Processing Complete ===")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Average FPS: {self.total_frames_processed/total_time:.2f}")
        print(f"Output saved to: {output_path}")

def main():
    processor = GPUStressTestProcessor()
    processor.process_video("./videos/4k.mov", "./output/output.mp4")

if __name__ == "__main__":
    main()
