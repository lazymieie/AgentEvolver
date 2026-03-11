import time
import argparse
import multiprocessing as mp

import torch
from pynvml import (
    nvmlInit, nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
)

def run_one_gpu(
    gpu: int,
    start_below: float,
    stop_at_or_above: float,
    hard_disable_at: float,
    poll_s: float,
    burst_s: float,
    dtype: str,
    size: int,
    streams: int,
    mem_free_gb: float,
):
    # NVML init
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu)

    torch.cuda.set_device(gpu)
    dev = torch.device(f"cuda:{gpu}")

    if dtype == "fp16":
        dt = torch.float16
    elif dtype == "bf16":
        dt = torch.bfloat16
    else:
        dt = torch.float32

    # 检查可用显存
    if mem_free_gb > 0:
        info = nvmlDeviceGetMemoryInfo(handle)
        free_gb = info.free / (1024**3)
        if free_gb < mem_free_gb:
            print(f"[GPU {gpu}] free_mem={free_gb:.1f}GB < {mem_free_gb}GB, autofill disabled")
            nvmlShutdown()
            return

    # 【关键优化 1】预先分配输出矩阵 c，彻底杜绝循环中的显存申请/释放开销
    a = torch.randn((size, size), device=dev, dtype=dt)
    b = torch.randn((size, size), device=dev, dtype=dt)
    c = torch.empty((size, size), device=dev, dtype=dt)

    # warmup
    for _ in range(5):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize(dev)

    last_print = 0.0

    while True:
        util = float(nvmlDeviceGetUtilizationRates(handle).gpu)
        
        # 状态打印：移到循环开头，确保打印的是当前最真实的 utilization
        now = time.time()
        if now - last_print > 1.0:
            print(f"[GPU {gpu}] util={util:5.1f}%  (start<{start_below}, stop>={stop_at_or_above})")
            last_print = now

        if util >= hard_disable_at:
            time.sleep(poll_s)
            continue

        if util < start_below:
            t_end = time.time() + burst_s
            
            # burst 循环
            while time.time() < t_end:
                # 【关键优化 2】一次性下发多个计算任务（复用输出内存 c），塞满 GPU 流水线
                # 不用多 stream 也能轻松跑满单卡
                for _ in range(5): 
                    torch.matmul(a, b, out=c)
                
                # 等这一小批任务执行完再同步，既防止 CPU 跑太快撑爆队列，又保证了 GPU 处于高负载
                torch.cuda.synchronize(dev)

                # 检查是否已经达标，达标则提前退出 burst
                util_now = float(nvmlDeviceGetUtilizationRates(handle).gpu)
                if util_now >= stop_at_or_above or util_now >= hard_disable_at:
                    break
        else:
            time.sleep(poll_s)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--start_below", type=float, default=60.0, help="only run when util < this")
    p.add_argument("--stop_at_or_above", type=float, default=60.0, help="stop burst when util >= this")
    p.add_argument("--hard_disable_at", type=float, default=95.0, help="never run when util >= this (acts like '100%')")
    p.add_argument("--poll_s", type=float, default=0.3, help="poll interval when idle")
    p.add_argument("--burst_s", type=float, default=0.2, help="how long to compute per trigger")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--size", type=int, default=12288, help="matmul size")
    p.add_argument("--streams", type=int, default=2, help="streams per GPU")
    p.add_argument("--mem_free_gb", type=float, default=0.0, help="only start if free mem >= this (GB); 0 disables")
    args = p.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    print(f"[threshold] gpus={gpus} start_below={args.start_below} stop_at_or_above={args.stop_at_or_above} hard_disable_at={args.hard_disable_at}")

    mp.set_start_method("spawn", force=True)
    procs = []
    for g in gpus:
        pr = mp.Process(
            target=run_one_gpu,
            args=(
                g, args.start_below, args.stop_at_or_above, args.hard_disable_at,
                args.poll_s, args.burst_s, args.dtype, args.size, args.streams, args.mem_free_gb
            ),
            daemon=True,
        )
        pr.start()
        procs.append(pr)

    try:
        for pr in procs:
            pr.join()
    except KeyboardInterrupt:
        print("\n[threshold] stopping...")

if __name__ == "__main__":
    main()