import torch
import time
import psutil
import GPUtil
from torch import nn
import torch.nn.functional as F
from datetime import datetime

class AggressiveModel(nn.Module):
    def __init__(self, input_size):
        super(AggressiveModel, self).__init__()
        # More complex architecture with more layers
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(64)
        
        # Dynamically calculate the flattened size
        test_input = torch.zeros(1, 1, input_size, input_size, input_size)
        with torch.no_grad():
            x = F.max_pool3d(self.bn1(F.relu(self.conv1(test_input))), 2)
            x = F.max_pool3d(self.bn2(F.relu(self.conv2(x))), 2)
            x = F.max_pool3d(self.bn3(F.relu(self.conv3(x))), 2)
            flattened_size = x.view(1, -1).size(1)
        
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, input_size**3 // 4)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.max_pool3d(self.bn1(F.relu(self.conv1(x))), 2)
        x = F.max_pool3d(self.bn2(F.relu(self.conv2(x))), 2)
        x = F.max_pool3d(self.bn3(F.relu(self.conv3(x))), 2)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

def adaptive_memory_benchmark(max_input_size=128, max_batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning Adaptive Memory Benchmark on: {device}")
    
    # Get GPU memory details
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"Total GPU Memory: {total_memory / (1024**3):.2f} GB")
    
    # Adaptive benchmark parameters
    results = []
    current_input_size = 16
    current_batch_size = 1
    
    while current_input_size <= max_input_size:
        while current_batch_size <= max_batch_size:
            try:
                # Memory management
                torch.cuda.empty_cache()
                
                # Create model and input
                model = AggressiveModel(current_input_size).to(device).half()
                input_data = torch.randn(
                    current_batch_size, 1, current_input_size, 
                    current_input_size, current_input_size, 
                    dtype=torch.half, device=device
                )
                
                # Check memory usage
                memory_allocated = torch.cuda.memory_allocated(device)
                memory_cached = torch.cuda.memory_reserved(device)
                print(f"\nTest Configuration:")
                print(f"Input Size: {current_input_size}")
                print(f"Batch Size: {current_batch_size}")
                print(f"Memory Allocated: {memory_allocated / (1024**2):.2f} MB")
                print(f"Memory Cached: {memory_cached / (1024**2):.2f} MB")
                
                # Warm-up
                for _ in range(3):
                    with torch.no_grad():
                        output = model(input_data)
                
                # Benchmark
                torch.cuda.synchronize()
                start_time = time.time()
                
                iterations = 10
                for _ in range(iterations):
                    with torch.no_grad():
                        output = model(input_data)
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                # Calculate performance metrics
                total_time = end_time - start_time
                avg_time = total_time / iterations
                fps = 1.0 / avg_time if avg_time > 0 else float('inf')
                
                # Store results
                results.append({
                    'input_size': current_input_size,
                    'batch_size': current_batch_size,
                    'avg_time': avg_time,
                    'fps': fps,
                    'memory_allocated': memory_allocated
                })
                
                print(f"Performance: {fps:.2f} FPS")
                print(f"Average Time: {avg_time*1000:.2f} ms")
                
            except RuntimeError as e:
                print(f"Memory Error: {e}")
                break
            except Exception as e:
                print(f"Unexpected Error: {e}")
                break
            
            # Increment batch size
            current_batch_size *= 2
        
        # Reset batch size and increment input size
        current_batch_size = 1
        current_input_size *= 2
    
    # Print detailed results
    print("\n=== Benchmark Results ===")
    for result in results:
        print(f"Input Size: {result['input_size']}, "
              f"Batch Size: {result['batch_size']}, "
              f"FPS: {result['fps']:.2f}, "
              f"Avg Time: {result['avg_time']*1000:.2f} ms")

if __name__ == "__main__":
    adaptive_memory_benchmark(max_input_size=128, max_batch_size=16)