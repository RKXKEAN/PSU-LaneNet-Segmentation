import time
import os
import sys
import psutil
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thop import profile
from models.custom_lane_net import get_model

def get_model_size_mb(model, filepath="temp_model.pth"):
    torch.save(model.state_dict(), filepath)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if os.path.exists(filepath):
        os.remove(filepath)
    return size_mb

def benchmark_inference(model, input_size=(1, 3, 384, 640), device="cuda", num_warmup=10, num_runs=50):
    """
    Comprehensive inference benchmarking:
      - Memory Footprint (VRAM / RAM)
      - Parameter count & FLOPs
      - Latency (ms) & Throughput (FPS)
    """
    model.eval()
    actual_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    model = model.to(actual_device)
    dummy_input = torch.randn(*input_size).to(actual_device)

    # 1. Parameter and FLOPs counting
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    flops = macs * 2 # Standard conversion: 1 MAC = 2 FLOPs
    model_size_mb = get_model_size_mb(model)

    device_name = "CPU"
    gpu_allocated_mb = 0.0
    gpu_reserved_mb = 0.0

    if actual_device == "cuda":
        device_name = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Measure baseline memory
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / (1024 * 1024)

        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model(dummy_input)
        torch.cuda.synchronize()

        # Timed runs
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = []
        for _ in range(num_runs):
            starter.record()
            with torch.no_grad():
                _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender)) # milliseconds

        gpu_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        gpu_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

    else:
        model = model.to("cpu")
        dummy_input = dummy_input.to("cpu")

        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model(dummy_input)

        # Timed runs
        timings = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_input)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0) # milliseconds

    process = psutil.Process()
    ram_usage_mb = process.memory_info().rss / (1024 * 1024)

    mean_latency = float(np.mean(timings))
    std_latency = float(np.std(timings))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    summary = {
        "device": device_name,
        "input_resolution": f"{input_size[2]}x{input_size[3]}",
        "parameters": int(params),
        "parameters_m": float(params / 1e6),
        "model_file_size_mb": float(model_size_mb),
        "gmacs": float(macs / 1e9),
        "gflops": float(flops / 1e9),
        "gpu_peak_allocated_mb": float(gpu_allocated_mb),
        "gpu_peak_reserved_mb": float(gpu_reserved_mb),
        "ram_usage_mb": float(ram_usage_mb),
        "mean_latency_ms": mean_latency,
        "std_latency_ms": std_latency,
        "fps": fps
    }
    return summary

if __name__ == "__main__":
    model = get_model(num_classes=6)
    print("=== GPU Benchmark ===")
    if torch.cuda.is_available():
        gpu_res = benchmark_inference(model, device="cuda")
        for k, v in gpu_res.items():
            print(f"  {k}: {v}")
    print("\n=== CPU Benchmark ===")
    cpu_res = benchmark_inference(model, device="cpu", num_runs=20)
    for k, v in cpu_res.items():
        print(f"  {k}: {v}")
