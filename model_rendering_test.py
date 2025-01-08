import torch
import time
import psutil
import GPUtil
from torch import nn
import torch.nn.functional as F
from datetime import datetime

class LightModel(nn.Module):
    def __init__(self, input_size):
        super(LightModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * (input_size//4)**3, (input_size//2)**3)
    
    def forward(self, x):
        x = F.max_pool3d(F.relu(self.conv1(x)), 2)
        x = F.max_pool3d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class MediumModel(nn.Module):
    def __init__(self, input_size):
        super(MediumModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * (input_size//4)**3, (input_size//2)**3)
    
    def forward(self, x):
        x = F.max_pool3d(F.relu(self.conv1(x)), 2)
        x = F.max_pool3d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def get_gpu_info():
    """Get GPU information"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return f"{gpu.name} - Memory: {gpu.memoryTotal}MB"
        return "No GPU detected"
    except:
        return "Unable to get GPU information"

def run_benchmark(input_sizes=[16, 32], batch_sizes=[1, 2], num_iterations=5):
    """Run benchmark and return summary statistics"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning benchmark on: {device}")
    print(f"GPU: {get_gpu_info()}")
    
    total_time = 0
    total_fps = 0
    test_count = 0
    benchmark_start = time.time()
    
    models = {
        'Light': LightModel,
        'Medium': MediumModel
    }
    
    for model_name, model_class in models.items():
        for input_size in input_sizes:
            for batch_size in batch_sizes:
                try:
                    print(f"\nTesting {model_name} - Input size: {input_size}, Batch: {batch_size}")
                    
                    # Initialize model and data
                    model = model_class(input_size).to(device)
                    input_data = torch.randn(batch_size, 1, input_size, input_size, input_size).to(device)
                    
                    # Warm-up
                    for _ in range(2):
                        model(input_data)
                    
                    # Benchmark
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    for _ in range(num_iterations):
                        model(input_data)
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    test_time = end_time - start_time
                    avg_time = test_time / num_iterations
                    fps = 1.0 / avg_time
                    
                    # Add to totals
                    total_time += test_time
                    total_fps += fps
                    test_count += 1
                    
                    # Clean up
                    del model, input_data
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    print(f"Error: {e}")
                    torch.cuda.empty_cache()
                    continue
    
    benchmark_end = time.time()
    total_benchmark_time = benchmark_end - benchmark_start
    
    return {
        'total_benchmark_time': total_benchmark_time,
        'average_fps': total_fps / test_count if test_count > 0 else 0,
        'tests_completed': test_count
    }

if __name__ == "__main__":
    # Run benchmark
    print("Starting GPU Benchmark...")
    print("=" * 50)
    
    results = run_benchmark(
        input_sizes=[64, 128],  # Test with these input sizes
        batch_sizes=[4, 8],    # Test with these batch sizes
        num_iterations=5       # Number of iterations per test
    )
    
    print("\nFinal Results:")
    print("=" * 50)
    print(f"Total Benchmark Time: {results['total_benchmark_time']:.2f} seconds")
    print(f"Average FPS: {results['average_fps']:.2f}")
    print(f"Tests Completed: {results['tests_completed']}")
    print("=" * 50)