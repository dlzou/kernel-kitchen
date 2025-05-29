import torch
import triton
import triton.language as tl


if torch.cuda.is_available():
    print(f"{torch.version.cuda=}")
    DEVICE = torch.device("cuda")
else:
    print("No CUDA available")
    DEVICE = torch.device("cpu")

GEMM_FLAVOR = "group_m"

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2") # Not present on RTX 4090


def get_autotune_config_basic():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5,
                      num_warps=2),
    ]


@triton.autotune(
    configs=get_autotune_config_basic(),
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel_basic(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Compute blocks of pointers to A and Busing broadcasting
    a_ptrs = A_ptr + stride_am * offs_am[:, None] + stride_ak * offs_k[None, :]
    b_ptrs = B_ptr + stride_bk * offs_k[:, None] + stride_bn * offs_bn[None, :]
    # Accumulate into a block of fp32 values for higher precision
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main loop
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k < K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += stride_ak * BLOCK_SIZE_K
        b_ptrs += stride_bk * BLOCK_SIZE_K

    # Convert accumulated fp32 values back to the original dtype
    c = acc.to(tl.float16)
    
    # Write back accumulated block to C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def get_autotune_config_group_m():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ]


@triton.autotune(
    configs=get_autotune_config_group_m(),
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel_group_m(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Change block launch order to take advantage of L2 cache.

    num_pids_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pids_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pids_in_group = GROUP_SIZE_M * num_pids_n
    group_id = pid // num_pids_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(GROUP_SIZE_M, num_pids_m - first_pid_m)
    
    pid_m = first_pid_m + ((pid % num_pids_in_group) % group_size_m)
    pid_n = (pid % num_pids_in_group) // group_size_m
    
    # Rest is same as basic gemm
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Compute blocks of pointers to A and Busing broadcasting
    a_ptrs = A_ptr + stride_am * offs_am[:, None] + stride_ak * offs_k[None, :]
    b_ptrs = B_ptr + stride_bk * offs_k[:, None] + stride_bn * offs_bn[None, :]
    # Accumulate into a block of fp32 values for higher precision
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main loop
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k < K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += stride_ak * BLOCK_SIZE_K
        b_ptrs += stride_bk * BLOCK_SIZE_K

    # Convert accumulated fp32 values back to the original dtype
    c = acc.to(tl.float16)
    
    # Write back accumulated block to C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


GEMM_KERNELS = {
    "basic": gemm_kernel_basic,
    "group_m": gemm_kernel_group_m,
}


def gemm(a: torch.Tensor, b: torch.Tensor, flavor="basic"):
    assert a.shape[1] == b.shape[0], "incompatible shapes"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    if flavor == "basic":
        # 2D launch kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    elif flavor == "group_m":
        # 1D launch kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    else:
        raise ValueError(f"Invalid flavor: {flavor}")

    GEMM_KERNELS[flavor][grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# UNIT TEST #


def unit_test():
    torch.manual_seed(0)
    M = 512
    N = 512
    K = 512
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

    triton_output = gemm(a, b, flavor=GEMM_FLAVOR)
    torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match for fp16 inputs")
    else:
        print("❌ Triton and Torch differ for fp16 inputs")
        print(f"{triton_output=}")
        print(f"{torch_output=}")
    
    if TORCH_HAS_FP8:
        torch.manual_seed(0)
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
        a = a.to(torch.float8_e5m2)
        # pre-transpose b for efficiency.
        b = b.T
        b = b.to(torch.float8_e5m2)
        triton_output = gemm(a, b, flavor=GEMM_FLAVOR)
        torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
        print(f"triton_output_with_fp8_inputs={triton_output}")
        print(f"torch_output_with_fp8_inputs={torch_output}")
        if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
            print("✅ Triton and Torch match for fp8 inputs")
        else:
            print("❌ Triton and Torch differ for fp8 inputs")


# BENCHMARK #

ref_lib = 'cuBLAS'

configs = []
for fp8_inputs in [False, True]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not DEVICE.type == "cuda"):
        continue
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],  # Label name for the lines
            line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name=f"gemm-{'fp8' if fp8_inputs else 'fp16'}-{GEMM_FLAVOR}",
            args={"fp8_inputs": fp8_inputs},
        ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gemm(a, b, flavor=GEMM_FLAVOR), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    unit_test()
    benchmark.run(print_data=True, save_path='./')