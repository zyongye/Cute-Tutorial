import functools

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.runtime as cute_rt
from cutlass import Float32
from cutlass.cute.nvgpu import cpasync


NUM_CTAS = 2

@cute.jit
def elementwise_add_cute(
    mA: cute.Tensor,
    mRes: cute.Tensor,
):
    BLOCK_M, BLOCK_N = 128, 128
    tiler = (BLOCK_M, BLOCK_N)
    smem_layout = cute.make_layout((BLOCK_M, BLOCK_N), stride=(BLOCK_N, 1))
    cta_layout_vmnk = cute.make_layout((1, 1, NUM_CTAS, 1))

    tma_copy_atom_A, tma_A = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SMulticastOp(),
        mA,
        smem_layout,
        cta_tiler=tiler,
        num_multicast=NUM_CTAS,
    )

    tma_store_atom_Res, tma_Res = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(),
        mRes,
        smem_layout,
        cta_tiler=tiler,
    )

    grid = cute.ceil_div(mA.shape, tiler)
    print(f"[DSL INFO] grid = {grid}")
    elementwise_add_kernel(
        mA,
        tma_A,
        tma_copy_atom_A,
        mRes,
        tma_Res,
        tma_store_atom_Res,
        smem_layout,
        tiler,
        cta_layout_vmnk,
    ).launch(
        # grid.x must be multiple of cluster size
        # cluster is folded into bidx
        grid=(grid[0] * NUM_CTAS, grid[1], 1),
        block=(128, 1, 1),
        cluster=(NUM_CTAS, 1, 1)
    )

@cute.kernel
def elementwise_add_kernel(
    mA: cute.Tensor,
    tma_mA: cute.Tensor,
    tma_copy_atom_A: cute.CopyAtom,
    mRes: cute.Tensor,
    tma_mRes: cute.Tensor,
    tma_store_atom_Res: cute.CopyAtom,
    smem_layout: cute.Layout,
    tiler: cute.IntTuple,
    cta_layout_vmnk: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
    # real bdix is after dividing cluster size
    bidx = bidx // NUM_CTAS

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(mA.element_type, smem_layout, byte_alignment=128)
    mbar = smem.allocate_array(cutlass.Int64)
    print(f"[DSL Info] sA = {sA}")

    if tidx == 0:
        cpasync.prefetch_descriptor(tma_copy_atom_A)
        # single thread required for mbar init, 
        # set to one because always one thread will perform the operation
        # all other thread will wait on mbar to become 0
        cute.arch.mbarrier_init(mbar, 1)
        cute.arch.mbarrier_expect_tx(mbar, cute.size_in_bytes(mA.element_type, smem_layout))

    cute.arch.mbarrier_init_fence()
    cute.arch.cluster_arrive(aligned=True)
    cute.arch.cluster_wait()

    # (BLOCK_M, BLOCK_N)
    gA = cute.local_tile(tma_mA, tiler, (bidx, bidy))
    print(f"[DSL INFO] gA = {gA}")
    gRes_tma = cute.local_tile(tma_mRes, tiler, (bidx, bidy))

    tAsA, tAgA = cpasync.tma_partition(
        tma_copy_atom_A, 
        cta_coord_vmnk[2],
        cute.make_layout(
            cute.size(cta_layout_vmnk, mode=[2])
        ),
        cute.group_modes(sA, 0, 2),
        cute.group_modes(gA, 0, 2),
    )

    print(f"[DSL INFO] tAgA = {tAgA}")    

    mcast_mask = cpasync.create_tma_multicast_mask(
        cta_layout_vmnk,
        cta_coord_vmnk,
        mcast_mode=2,
    )

    tRes_sRes, tRes_gRes = cpasync.tma_partition(
        tma_store_atom_Res,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 2),
        cute.group_modes(gRes_tma, 0, 2),
    )

    if warp_id == 0:
        cute.copy(
            tma_copy_atom_A,
            tAgA,
            tAsA,
            tma_bar_ptr=mbar,
            mcast_mask=mcast_mask,
        )

        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(mbar)

    # 0 means phase, since we only use it once, this is the initial phase
    # if we want to use it again, we need to flip the phase
    # and set expect_tx again
    cute.arch.mbarrier_wait(mbar, 0)

    for i in cutlass.range(cute.size(smem_layout, mode=0)):
        sA[(i, tidx)] = sA[(i, tidx)] + 1

    cute.arch.sync_threads()
    cute.arch.fence_proxy(
        kind="async.shared", 
        space="cta",
    )

    if cta_rank_in_cluster == 0 and warp_id == 0:
        cute.copy(
            tma_store_atom_Res,
            tAsA,
            tRes_gRes,
        )
    # cute.arch.cp_async_bulk_commit_group()
    # cute.arch.cp_async_bulk_wait_group(0)

    

@functools.cache
def elementwise_add_get_kernel(
    M, N
):
    A_fake = cute_rt.make_fake_tensor(Float32, shape=(M, N), stride=(N, 1), assumed_align=16)
    res_fake = cute_rt.make_fake_tensor(Float32, shape=(M, N), stride=(N, 1), assumed_align=16)

    return cute.compile(
        elementwise_add_cute,
        A_fake,
        res_fake,
        options="--enable-tvm-ffi --keep-ptx",
    )

def elementwise_add_tma(
    A: torch.Tensor,
) -> torch.Tensor:
    M, N = A.shape
    res = torch.empty_like(A)
    kernel = elementwise_add_get_kernel(M, N)
    kernel(A, res)
    return res

def main():
    device = "cuda"
    M, N = 1024, 1024
    A = torch.rand(M, N, dtype=torch.float32, device=device)
    res = elementwise_add_tma(A)
    torch.testing.assert_close(res, A + 1)

    # print(A)
    # print(res)

if __name__ == "__main__":
    main()

