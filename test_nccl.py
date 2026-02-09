#!/usr/bin/env python3
"""
NCCL Diagnostic Script for Multi-Node GPU Serving

This script helps identify NCCL configuration issues that can cause
multi-node GPU serving failures. Run this script on each node to verify
NCCL function before deploying distributed workloads.

Usage: python3 multi-node-nccl-check.py
"""
import os
import sys
import socket
import torch
from datetime import datetime

def log(msg):
    """Log messages with timestamp for better debugging."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def print_environment_info():
    """Print relevant environment information for debugging."""
    log("=== Environment Information ===")
    log(f"Hostname: {socket.gethostname()}")
    log(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    # Print all NCCL-related environment variables.
    nccl_vars = [var for var in os.environ.keys() if var.startswith('NCCL_')]
    if nccl_vars:
        log("NCCL Environment Variables:")
        for var in sorted(nccl_vars):
            log(f"  {var}: {os.environ[var]}")
    else:
        log("No NCCL environment variables set")

def check_cuda_availability():
    """Verify CUDA is available and functional."""
    log("\n=== CUDA Availability Check ===")
    
    if not torch.cuda.is_available():
        log("ERROR: CUDA not available")
        return False
    
    device_count = torch.cuda.device_count()
    log(f"CUDA device count: {device_count}")
    log(f"PyTorch version: {torch.__version__}")
    
    # Check NCCL availability in PyTorch.
    try:
        import torch.distributed as dist
        if hasattr(torch.distributed, 'nccl'):
            log(f"PyTorch NCCL available: {torch.distributed.is_nccl_available()}")
    except Exception as e:
        log(f"Error checking NCCL availability: {e}")
    
    return True

def test_individual_gpus():
    """Test that each GPU is working individually."""
    log("\n=== Individual GPU Tests ===")
    
    for gpu_id in range(torch.cuda.device_count()):
        log(f"\n--- Testing GPU {gpu_id} ---")
        
        try:
            torch.cuda.set_device(gpu_id)
            device = torch.cuda.current_device()
            
            log(f"Device {device}: {torch.cuda.get_device_name(device)}")
            
            # Print device properties.
            props = torch.cuda.get_device_properties(device)
            log(f"  Compute capability: {props.major}.{props.minor}")
            log(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
            
            # Test basic CUDA operations.
            log("  Testing basic CUDA operations...")
            tensor = torch.ones(1000, device=f'cuda:{gpu_id}')
            result = tensor.sum()
            log(f"  Basic CUDA test passed: sum = {result.item()}")
            
            # Test cross-GPU operations if multiple GPUs are available.
            if torch.cuda.device_count() > 1:
                log("  Testing cross-GPU operations...")
                try:
                    other_gpu = (gpu_id + 1) % torch.cuda.device_count()
                    test_tensor = torch.randn(10, 10, device=f'cuda:{gpu_id}')
                    tensor_copy = test_tensor.to(f'cuda:{other_gpu}')
                    log(f"  Cross-GPU copy successful: GPU {gpu_id} -> GPU {other_gpu}")
                except Exception as e:
                    log(f"  Cross-GPU copy failed: {e}")
            
            # Test memory allocation.
            log("  Testing large memory allocations...")
            try:
                large_tensor = torch.zeros(1000, 1000, device=f'cuda:{gpu_id}')
                log("  Large memory allocation successful")
                del large_tensor
            except Exception as e:
                log(f"  Large memory allocation failed: {e}")
                
        except Exception as e:
            log(f"ERROR testing GPU {gpu_id}: {e}")
            import traceback
            log(f"Traceback:\n{traceback.format_exc()}")

def test_nccl_initialization():
    """Test NCCL initialization and basic operations."""
    log("\n=== NCCL Initialization Test ===")
    
    try:
        import torch.distributed as dist
        
        # Set up single-process NCCL environment.
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        
        log("Attempting single-process NCCL initialization...")
        dist.init_process_group(
            backend='nccl',
            rank=0,
            world_size=1
        )
        
        log("Single-process NCCL initialization successful!")
        
        # Test basic NCCL operation.
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            tensor = torch.ones(10, device=device)
            
            # This is a no-op with world_size=1 but exercises NCCL
            dist.all_reduce(tensor)
            log("NCCL all_reduce test successful!")
        
        dist.destroy_process_group()
        log("NCCL cleanup successful!")
        
    except Exception as e:
        log(f"NCCL initialization failed: {e}")
        import traceback
        log(f"Full traceback:\n{traceback.format_exc()}")

def main():
    """Main diagnostic routine."""
    log("Starting NCCL Diagnostic Script")
    log("=" * 50)
    
    print_environment_info()
    
    if not check_cuda_availability():
        sys.exit(1)
    
    test_individual_gpus()
    test_nccl_initialization()
    
    log("\n" + "=" * 50)
    log("NCCL diagnostic script completed")
    log("If you encountered errors, check the specific error messages above")
    log("and refer to the troubleshooting guide for solutions.")

if __name__ == "__main__":
    main()
