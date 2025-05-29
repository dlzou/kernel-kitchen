import argparse
import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

from conv2d.conv2d_warp import conv2d_basic_setup
from utils import get_device, set_seed, verbose_allclose


wp.init()

TORCH_DEVICE = get_device()
WP_DEVICE = wp.device_from_torch(TORCH_DEVICE)

MODES = ["test", "bench"]

SETUP_FNS = {
    "conv2d_basic": conv2d_basic_setup,
}


def conv2d_ref(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    stride = 1,
    padding = 0,
) -> torch.Tensor:
    return F.conv2d(
        input_tensor,
        weight,
        stride=stride,
        padding=padding,
    )


def test(
    setup_fn,
    input_shape = (8, 3, 32, 32),
    weight_shape = (6, 3, 5, 5),
    stride = 1,
    padding = 0,
):
    print("\n******** TEST CONFIG ********")
    params = {**locals()}
    print(f"{params=}\n")

    set_seed()

    input_tensor = torch.randn(input_shape, device=TORCH_DEVICE, dtype=torch.float64)
    weight = torch.randn(weight_shape, device=TORCH_DEVICE, dtype=torch.float64)

    input_tensor_wp = wp.from_torch(input_tensor)
    weight_wp = wp.from_torch(weight)
    fn = setup_fn(input_tensor_wp, weight_wp, stride, padding)
    output_tensor_wp = fn()
    output_tensor = wp.to_torch(output_tensor_wp)
    
    output_tensor_ref = conv2d_ref(input_tensor, weight, stride, padding)
    
    print(f"{output_tensor.shape=}")
    print(f"{output_tensor_ref.shape=}")
    mismatch = verbose_allclose(output_tensor, output_tensor_ref, atol=1e-5)

    print("\n******** TEST RESULTS ********")
    print(f"{fn.__name__=}")
    if len(mismatch) == 0:
        print("SUCCESS")
    else:
        print("\n".join(mismatch))


def bench(
    setup_fn,
    warmup = 20,
    repeat = 100,
    l2_cache_mb = 72,
    input_shape = (8, 3, 256, 256),
    weight_shape = (6, 3, 5, 5),
    stride = 1,
    padding = 0,
):
    print("\n******** BENCH CONFIG ********")
    params = {**locals()}
    print(f"{params=}\n")

    set_seed()
    
    # Used to clear the L2 cache
    cache = wp.empty(int(l2_cache_mb * 1e6 / 4), dtype=wp.int32, device=WP_DEVICE)

    input_tensor = wp.from_torch(torch.randn(input_shape, device=TORCH_DEVICE, dtype=torch.float32))
    weight = wp.from_torch(torch.randn(weight_shape, device=TORCH_DEVICE, dtype=torch.float32))

    stream = wp.Stream(WP_DEVICE)
    start_events = [wp.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [wp.Event(enable_timing=True) for _ in range(repeat)]
    runtimes = np.zeros((repeat,))

    fn = setup_fn(input_tensor, weight)
    for _ in range(warmup):
        _ = fn()
        
    for i in range(repeat):
        cache.zero_()

        stream.record_event(start_events[i])
        _ = fn()
        stream.record_event(end_events[i])

        runtimes[i] = wp.get_event_elapsed_time(start_events[i], end_events[i]) 

    print("\n******** BENCH RESULTS ********")
    print(f"{fn.__name__=}")
    print(f"mean={np.mean(runtimes)*1000:.3f} µs")
    print(f"std={np.std(runtimes)*1000:.3f} µs")
    print(f"min={np.min(runtimes)*1000:.3f} µs")
    print(f"median={np.median(runtimes)*1000:.3f} µs")
    print(f"max={np.max(runtimes)*1000:.3f} µs")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a kernel.")
    parser.add_argument(
        "mode",
        type=str,
        help=f"One of {MODES}",
    )
    parser.add_argument(
        "--fn",
        type=str,
        required=True,
        help=f"One of {SETUP_FNS.keys()}",
    )
    parser.add_argument(
        "--params",
        type=json.loads,
        default={},
        help=f"Params to pass to function",
    )
    args = parser.parse_args()

    if args.mode == "test":
        if args.fn in SETUP_FNS.keys():
            test(SETUP_FNS[args.fn], **args.params)
        else:
            print(f"Invalid argument: --fn must be one of {SETUP_FNS.keys()}")
    elif args.mode == "bench":
        if args.fn in SETUP_FNS.keys():
            bench(SETUP_FNS[args.fn], **args.params)
        else:
            print(f"Invalid argument: --fn must be one of {SETUP_FNS.keys()}")
    else:
        print(f"Invalid argument: --mode must be one of {MODES}")


if __name__ == "__main__":
    sys.exit(main())
