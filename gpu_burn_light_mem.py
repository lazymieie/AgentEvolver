import time
import torch
import torch.multiprocessing as mp

# ========== 可调参数 ==========
DTYPE = torch.float16         # 更省显存；也可改成 torch.bfloat16
MATRIX_SIZE = 2048            # 1024 / 2048 / 4096
SLEEP_EVERY = 0               # 每轮后 sleep 秒数，0 表示不停跑
PRINT_EVERY = 50              # 每多少轮打印一次
NUM_GPUS = 8                  # 占用前 8 张卡
USE_TENSOR_CORES = True
# ==============================


def worker(gpu_id: int):
    assert torch.cuda.is_available(), "没有检测到 CUDA GPU"
    assert gpu_id < torch.cuda.device_count(), (
        f"请求使用 GPU {gpu_id}，但当前只有 {torch.cuda.device_count()} 张卡"
    )

    torch.cuda.set_device(gpu_id)

    if USE_TENSOR_CORES:
        torch.set_float32_matmul_precision("high")

    device = torch.device(f"cuda:{gpu_id}")
    gpu_name = torch.cuda.get_device_name(device)

    print(f"[GPU {gpu_id}] Using GPU: {gpu_name}", flush=True)

    with torch.no_grad():
        a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device, dtype=DTYPE)
        b = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device, dtype=DTYPE)
        c = torch.empty(MATRIX_SIZE, MATRIX_SIZE, device=device, dtype=DTYPE)

        # 预热
        for _ in range(10):
            c = a @ b
            a, b = b, c
        torch.cuda.synchronize(device)

        mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
        print(
            f"[GPU {gpu_id}] Start compute. "
            f"Matrix={MATRIX_SIZE}x{MATRIX_SIZE}, dtype={DTYPE}, "
            f"allocated={mem_alloc:.2f} MB",
            flush=True
        )

        step = 0
        start_time = time.time()

        while True:
            c = a @ b
            a, b = b, c
            step += 1

            if step % PRINT_EVERY == 0:
                torch.cuda.synchronize(device)
                elapsed = time.time() - start_time
                mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
                print(
                    f"[GPU {gpu_id}] step={step}, elapsed={elapsed:.1f}s, "
                    f"allocated={mem_alloc:.2f}MB, reserved={mem_reserved:.2f}MB",
                    flush=True
                )

            if SLEEP_EVERY > 0:
                time.sleep(SLEEP_EVERY)


def main():
    assert torch.cuda.is_available(), "没有检测到 CUDA"
    total_gpus = torch.cuda.device_count()
    assert total_gpus >= NUM_GPUS, f"当前只有 {total_gpus} 张 GPU，不足 {NUM_GPUS} 张"

    print(f"Detected {total_gpus} GPUs, launching {NUM_GPUS} workers...", flush=True)

    mp.spawn(worker, nprocs=NUM_GPUS, join=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()