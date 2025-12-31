import torch
import triton

import cutlass 
import cutlass.cute as cute

from cutlass.cute.runtime import from_dlpack

@cute.kernel
def naive_elementwise_mul_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    # This kernel perform gC = gA + gB
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # Map thread index to logical index of input tensor
    m, n = gA.shape
    ni = thread_idx % n
    mi = thread_idx // n

    ############# Your code start #############

    ############# Your code end ###############

@cute.jit  # Just-in-time compilation decorator
def naive_elementwise_mul_host(
    mA: cute.Tensor,  # Input tensor A
    mB: cute.Tensor,  # Input tensor B
    mC: cute.Tensor,  # Output tensor C
):
    num_threads_per_block = 256

    m, n = mA.shape

    kernel = naive_elementwise_mul_kernel(mA, mB, mC)

    kernel.launch(
        grid=((m * n) // num_threads_per_block, 1, 1),  # Number of blocks in x,y,z
        block=(num_threads_per_block, 1, 1),  # Threads per block in x,y,z
    )


def naive_elementwise_mul():
    benchmark = True

    M, N = 16384, 8192

    a = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)  
    b = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)  
    c = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16) 

    a_ = from_dlpack(a, assumed_align=16)  
    b_ = from_dlpack(b, assumed_align=16)  
    c_ = from_dlpack(c, assumed_align=16) 

    naive_elementwise_mul_ = cute.compile(naive_elementwise_mul_host, a_, b_, c_)

    naive_elementwise_mul_(a_, b_, c_)

    torch.testing.assert_close(c, a * b)

    if benchmark:
        fn = lambda: naive_elementwise_mul_(a_, b_, c_)
        M, N = a.shape
        avg_time = triton.testing.do_bench(fn, warmup=2, rep=200)
        mem_bw = (M * N * 2 * a.element_size()) / (avg_time / 1000) / 1e9
        print(f"Kernel execution time: {avg_time:.4f} ms")
        print(f"Mem throughput: {mem_bw:.2f} GB/s")

@cute.kernel
def vectorized_elementwise_mul_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # Map thread index to logical index of input tensor in unit of vector
    m, n = gA.shape[1]  # thread-domain
    ni = thread_idx % n
    mi = thread_idx // n

    ############# Your code start #############

    ############# Your code end ###############


@cute.jit
def vectorized_elementwise_mul_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    threads_per_block = 256

    gA = cute.zipped_divide(mA, (?, ?))
    gB = cute.zipped_divide(mB, (?, ?))
    gC = cute.zipped_divide(mC, (?, ?))

    print("[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gA = {gA}")
    print(f"[DSL INFO]   gB = {gB}")
    print(f"[DSL INFO]   gC = {gC}")

    vectorized_elementwise_mul_kernel(gA, gB, gC).launch(
        grid=(cute.size(gC, mode=[1]) // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1),
    )


def vectorized_elementwise_mul():
    benchmark = True

    M, N = 16384, 8192

    a = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)  
    b = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)  
    c = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16) 

    a_ = from_dlpack(a, assumed_align=16)  
    b_ = from_dlpack(b, assumed_align=16)  
    c_ = from_dlpack(c, assumed_align=16) 

    vectorized_elementwise_mul_ = cute.compile(vectorized_elementwise_mul_host, a_, b_, c_)

    vectorized_elementwise_mul_(a_, b_, c_)

    torch.testing.assert_close(c, a * b)

    if benchmark:
        fn = lambda: vectorized_elementwise_mul_(a_, b_, c_)
        M, N = a.shape
        avg_time = triton.testing.do_bench(fn, warmup=2, rep=200)
        mem_bw = (M * N * 2 * a.element_size()) / (avg_time / 1000) / 1e9
        print(f"Kernel execution time: {avg_time:.4f} ms")
        print(f"Mem throughput: {mem_bw:.2f} GB/s")


if __name__ == "__main__":
    naive_elementwise_mul()
    vectorized_elementwise_mul()

