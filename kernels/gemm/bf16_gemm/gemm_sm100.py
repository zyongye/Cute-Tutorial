import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.cute.runtime as cute_rt
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05, OperandMajorMode
import cutlass.utils.blackwell_helpers as sm100_utils


class DenseGEMMSM100:
    def __init__(self, dtype):

        self.mma_tiler_mnk = (128, 256, 64)
        self.dtype = dtype
        self.acc_dtype = cutlass.Float32
        self.ab_stages = 4
        self.acc_stage = 1
        self.thread_per_cta = 128
    
    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        # print(f"[DSL INFO]: mA = {mA}")
        # gA_tiled = cute.tiled_divide(mA, self.mma_tiler_mn)
        # gA_zipped = cute.zipped_divide(mA, self.mma_tiler_mn)
        # gA_flat = cute.flat_divide(mA, self.mma_tiler_mn)
        # gA_logical = cute.logical_divide(mA, self.mma_tiler_mn)
        # print(f"[DSL INFO]: gA_tiled = {gA_tiled}") 
        # print(f"[DSL INFO]: gA_zipped = {gA_zipped}") 
        # print(f"[DSL INFO]: gA_flat = {gA_flat}") 
        # print(f"[DSL INFO]: gA_logical = {gA_logical}") 

        op = tcgen05.MmaF16BF16Op(
            ab_dtype=self.dtype,
            acc_dtype=self.acc_dtype,
            instruction_shape=(128, 256, 16),
            cta_group=tcgen05.CtaGroup.ONE,
            a_src=tcgen05.OperandSource.SMEM,
            a_major_mode=OperandMajorMode.K,
            b_major_mode=OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)

        # print(f"[DSL INFO]: tiled_mma = {tiled_mma}")

        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler_mnk,
            mA.dtype,
            self.ab_stages
        )

        # print(f"[DSL INFO]: a_smem_layout = {a_smem_layout}")
        # print(f"[DSL INFO]: a_smem_layout.outer = {a_smem_layout.outer}")
        # print(f"[DSL INFO]: a_smem_layout.inner = {a_smem_layout.inner}")

        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler_mnk,
            mB.dtype,
            self.ab_stages
        )

        # print(f"[DSL INFO]: b_smem_layout = {b_smem_layout}")

        a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
        b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

        # print(f"[DSL INFO]: b_smem_layout_one_stafe = {b_smem_layout_one_stage}")

        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            op,
            mA,
            a_smem_layout_one_stage,
            self.mma_tiler_mnk,
            tiled_mma,
        )

        b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            op,
            mB,
            b_smem_layout,
            self.mma_tiler_mnk,
            tiled_mma,
        )
        # print(f"[DSL INFO]: a_tma_atom = {a_tma_atom}")
        # print(f"[DSL INFO]: a_tma_tensor = {a_tma_tensor}")

        grid_shape = cute.ceil_div((*mC.layout.shape, 1), self.mma_tiler_mnk[:2])
        # print(f"[DSL INFO]: grid_shape = {grid_shape}")
        self.kernel(
            tiled_mma,
            a_tma_atom,
            a_tma_tensor,
            b_tma_atom,
            b_tma_tensor,
            mC,
            a_smem_layout,
            b_smem_layout,
        ).launch(
            grid=grid_shape,
            block=(self.thread_per_cta, 1, 1),
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stages * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stages * 2]
            tmem_holding_buf: cutlass.Int32
        
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        bidx, bidy, _ = cute.arch.block_idx()
        mma_coord_mnk = (bidx, bidy, None)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = smem.allocate_tensor(
            element_type = self.dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner
        )
        sB = smem.allocate_tensor(
            element_type = self.dtype,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.thread_per_cta,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
        )
        num_tmem_cols = 512
        tmem.allocate(num_tmem_cols)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        
        num_tma_copy_bytes = cute.size_in_bytes(
            self.dtype, cute.select(a_smem_layout, mode=[0,1,2])
        ) + cute.size_in_bytes(self.dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))

        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.ab_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=num_tma_copy_bytes,
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        ).make_participants()

        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.thread_per_cta,
            ),
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        ).make_participants()

        gA_mkl = cute.local_tile(mA_mkl, self.mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
        print(f"[DSL INFO]: gA = {gA_mkl}")
        gB_nkl = cute.local_tile(mB_nkl, self.mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
        print(f"[DSL INFO]: gB = {gB_nkl}")
        gC_mnl = cute.local_tile(mC_mnl, self.mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))

        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        print(f"[DSL INFO]: tCgA = {tCgA}")
        
        tCrA = tiled_mma.make_fragment_A(sA)
        print(f"[DSL INFO]: tCrA = {tCrA}")
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler_mnk[:2])
        print(f"[DSL INFO]: acc_shape = {acc_shape}")
        tCtAcc = tiled_mma.make_fragment_C(acc_shape)

        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )

        print(f"[DSL INFO]: tAgA = {tAgA}")
        print(f"[DSL INFO]: tAsA = {tAsA}")

        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtACC = cute.make_tensor(tmem_ptr, tCtAcc.layout)
        print(f"[DSL INFO]: tCtACC = {tCtACC}")

        subtile_cnt = 4
        epi_tiler = ((cute.size(tCtACC, mode=[0, 0]), cute.size(tCtACC, mode=[0, 1]) // subtile_cnt), )
        print(f"[DSL INFO]: epi_tiler = {epi_tiler}")

        tCtACC_epi = cute.zipped_divide(tCtACC, epi_tiler)
        print(f"[DSL INFO]: tCtACC_epi = {tCtACC_epi}")
        gC_epi = cute.zipped_divide(tCgC, epi_tiler)
        print(f"[DSL INFO]: gC_epi = {gC_epi}")

        tmem_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
            cutlass.Float32,
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtACC_epi[None, 0])
        print(f"[DSL INFO]: tmem_tiled_copy = {tmem_tiled_copy}")
        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
        print(f"[DSL INFO]: tmem_thr_copy = {tmem_thr_copy}")

        tDtC = tmem_thr_copy.partition_S(tCtACC_epi)
        tDgC = tmem_thr_copy.partition_D(gC_epi)
        print(f"[DSL INFO]: tDtC = {tDtC}")
        print(f"[DSL INFO]: tDgC = {tDgC}")

        tCrACC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, self.acc_dtype)
        tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, self.dtype)
        print(f"[DSL INFO]: tCrC = {tCrC}")

        num_k_tiles = cute.size(gA_mkl, mode=[2])
        if warp_idx == 0:
            acc_empty = acc_producer.acquire_and_advance()

            for k_tile_idx in cutlass.range(num_k_tiles, prefetch_stages=self.ab_stages - 2):
                ab_empty = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a, 
                    tAgA[(None, ab_empty.count)],
                    tAsA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_b, 
                    tBgB[(None, ab_empty.count)],
                    tBsB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )

                ab_full = ab_consumer.wait_and_advance()
                num_k_blocks = cute.size(tCrA, mode=[2])

                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_coord = (None, None, k_block_idx, ab_full.index)
                    cute.gemm(
                        tiled_mma,
                        tCtACC,
                        tCrA[k_block_coord],
                        tCrB[k_block_coord],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                
                ab_full.release()
            acc_empty.commit()
        
        tmem.relinquish_alloc_permit()
        acc_full = acc_consumer.wait_and_advance()

        for i in cutlass.range(cute.size(tDtC, mode=[2])):
            cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrACC)
            tCrC.store(tCrACC.load().to(self.dtype))
            cute.autovec_copy(tCrC, tDgC[None, None, i])
        acc_full.release()

        pipeline.sync(barrier_id = 1)
        tmem.free(tmem_ptr)


def _gemm_bf16(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,     
):
    M, K = A.shape
    N, K = B.shape

    a_fake = cute_rt.make_fake_tensor(cute.BFloat16, (M, K), stride=(K, 1), assumed_align=16)
    b_fake = cute_rt.make_fake_tensor(cute.BFloat16, (N, K), stride=(K, 1), assumed_align=16)
    c_fake = cute_rt.make_fake_tensor(cute.BFloat16, (M, N), stride=(N, 1), assumed_align=16)

    compiled_kernel = cute.compile(
        DenseGEMMSM100(a_fake.dtype),
        a_fake,
        b_fake,
        c_fake,
        options="--enable-tvm-ffi",
    )

    compiled_kernel(A, B, C)

def gemm_bf16_cutedsl(A: torch.Tensor, B: torch.Tensor):
    M, _ = A.shape
    N, _ = B.shape

    C = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)

    _gemm_bf16(
        A, B, C,
    )
    
    return C


def main():
    M, N, K, L = 8192, 8192, 8192, 256
    device = "cuda"

    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    
    C = gemm_bf16_cutedsl(A, B)

    torch.testing.assert_close(C, A@B.T)
    

if __name__ == "__main__":
    main()


