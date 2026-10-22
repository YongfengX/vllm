# vLLM v0.30.0 → v0.32.0 变更总结（功能 + 算子）

> 面向 AI / 代码代理的结构化变更文档。每一条都给出 **PR 号、commit、源码位置**，方便直接 `git show` / 打开文件核对。
> 配套全量索引（1541 个提交逐条列出改动文件）：[`vllm_v0.30.0_to_v0.32.0_commit_index.md`](./vllm_v0.30.0_to_v0.32.0_commit_index.md)

## 0. 元信息与使用约定

| 项 | 值 |
| --- | --- |
| 起点 | `v0.30.0` = `ced6857afa`（2026-09-21，release 分支，在 main 的 `f2aad6aa70` 之后 cherry-pick 了 11 个提交） |
| 终点 | `v0.32.0` = `5df63a7203`（2026-10-17） |
| 范围 | `git log v0.30.0..v0.32.0` 共 **1541** 个提交；`git diff v0.30.0 v0.32.0` 共 **4357** 个文件，+337,882 / −96,364 行 |
| 提交分布（按标题标签） | Bugfix 477、ROCm 166、CI 122、Perf 90、Frontend 47、XPU 45、Rust Frontend 39、Model 37、Kernel 13、Quantization 12、Attention 12… |

### AI 使用约定

1. 本仓库默认只有 `main`，需要先拉官方 tag：
   ```bash
   git remote add upstream https://github.com/vllm-project/vllm.git
   git fetch --no-tags upstream tag v0.30.0 tag v0.32.0
   ```
2. 文中路径，除特别标注外，都是 **v0.32.0 中的最终位置**。
   PR 的「改动文件」用的是**提交当时的路径**（全量索引里也一样）。
3. ⚠️ **csrc 目录重组**（PR [#60721](https://github.com/vllm-project/vllm/pull/60721)，`d79b52b182`）：`csrc/libtorch_stable/**` 已整体迁到 `csrc/**`。
   所以早期提交里出现的 `csrc/libtorch_stable/X`，在 v0.32.0 中对应的是 `csrc/X`。详见 §2.1。
4. 查看任一 PR：`https://github.com/vllm-project/vllm/pull/<号>`；查看提交：`git show <commit>`。
5. 查某个文件在两个版本间的变化：`git diff v0.30.0 v0.32.0 -- <path>`，或 `git log --oneline v0.30.0..v0.32.0 -- <path>`。

---

## 1. 一页概览（最重要的变化）

### 算子 / Kernel

- `csrc/libtorch_stable/` 合并回 `csrc/`；共享头文件统一到 `csrc/core/`（[#60721](https://github.com/vllm-project/vllm/pull/60721)）。
- 新增 Attention 后端 3 个：`ULTRAQUANT`（4-bit KV，ROCm FlyDSL）、`ROCM_SEGMENTED_ATTN`（RDNA3/3.5/4）、`FLASHMLA_MEGA_ATTN_DSV41`（DeepSeek-V4.1）。另外 Attention 后端枚举现在可以**动态注册**（[#41091](https://github.com/vllm-project/vllm/pull/41091)）。
- **移除 cascade attention**（[#61126](https://github.com/vllm-project/vllm/pull/61126)）：`--disable-cascade-attn` 现在是 no-op 并给出弃用提示。
- **移除基于 torch.compile 的序列并行（SP）和 AsyncTP pass**（[#60517](https://github.com/vllm-project/vllm/pull/60517)）。
- **移除 AllSpark INT8 W8A16 GEMM**（[#58001](https://github.com/vllm-project/vllm/pull/58001)）。
- NVFP4 KV cache 扩展到 SM8x/SM12x（FlashInfer）、SM90（fa2）、SM12x XQA decode，以及 MiniMax-M3 MSA 稀疏路径。
- **Humming** 量化 kernel 库全面接入：linear、MoE、MXFP6、非对称 zero-point；在 SM90 上优先级高于 Marlin。
- MoE 新后端：FlashInfer CuteDSL MegaMoE、DeepGEMM MegaMoE 抽象、MoonEP（BF16）、AITER MegaMoEV2（ROCm），以及 MXFP4 oracle 新增 CUTLASS W4A4、per-token NVFP4 CuTe-DSL。
- 新通信算子：SM100/SM103 上的 MNNVL multimem 低 SM reduce-scatter；DSv4.1 的 all-reduce + mHC + MoE finalize 融合（最终是 CuTe-DSL 实现）。
- CPU 新算子：FP8 W8A8 linear/MoE、向量化采样（greedy/gumbel argmax）、aarch64 conv1d、POWER VSX 的 INT8 MoE 和 W4A16。
- Triton：新增**平台可覆盖的 Triton kernel 注册表** `vllm/triton_utils/dispatcher.py`；kernel 不再使用 `tl.make_block_ptr`，以兼容 triton-main。

### 功能

- Model Runner V2（MRV2）新增：草稿模型（draft model）投机解码、GPU N-gram 投机解码、自定义 logits processor、stock torch.compile 模式、Elastic EP、PP+PCP。
- 新增投机解码器 **LiLiCorr**；AR speculator 重命名为 `StandaloneAR` / `TargetDependentAR`；DFlash 支持异步调度。
- 新增 CLI：`vllm preload`、`vllm snapshot`、`vllm download-kernels`；新增端点 `/v1/systemone`（structured decisions）。
- 调度器：新增 `--max-num-active-seqs`，`--long-prefill-token-threshold` 支持自适应，重写 `skipped_waiting` 队列。
- 新增 **AuxOutput** 框架（routed experts 输出按 block 存储）、**KV hints** 请求封装、`shm tensor arena`。
- Sleep / RL：sleep 时可以释放 CUDA graph 池、workspace 和 KV 初始化状态；新增 NCCL M2N 与 ModelExpress 权重传输。
- 日志：新增 `--logging-config`、`--log-level`，内置 JSON formatter。
- **不再支持 Python 3.10**；Transformers 版本要求 `>=5.16.1,<5.20.0`；FlashInfer 0.7.0.post1；CUDA 版 torch 2.14。

---

## 2. 算子（Kernel）层面的变化

### 2.1 csrc 目录结构重组（影响所有 C++/CUDA 路径）

- **PR [#60721](https://github.com/vllm-project/vllm/pull/60721)** `d79b52b182` *Consolidate csrc shared headers and prune dead code*，共改 233 个文件，188 个是重命名：

| v0.30.0 路径 | v0.32.0 路径 |
| --- | --- |
| `csrc/libtorch_stable/*.cu` / `*.cuh`（如 `activation_kernels.cu`、`async_util.cuh`、`topk.cu`） | `csrc/*` |
| `csrc/libtorch_stable/attention/**`（`merge_attn_states.cu`、`dcp_utils/*`、`mla/cutlass_sm100_mla/*`） | `csrc/attention/**` |
| `csrc/libtorch_stable/quantization/{w8a8,marlin,machete,fp4,gptq,cutlass_w4a8,fused_kernels}/**` | `csrc/quantization/**` |
| `csrc/libtorch_stable/moe/**`（含 `marlin_moe_wna16`、`permute_unpermute_kernels`） | `csrc/moe/**` |
| `csrc/libtorch_stable/cutlass_extensions/**` | `csrc/cutlass_extensions/**` |
| `csrc/libtorch_stable/kimi_k3/**` | `csrc/kimi_k3/**` |
| `csrc/attention/{attention_dtypes.h,dtype_*.cuh,attention_generic.cuh}` | `csrc/core/vec_dtypes/{vec_dtypes.h,dtype_*.cuh,dtype_generic.cuh}` |
| `csrc/cuda_compat.h` | `csrc/core/cuda_compat.h` |
| `csrc/custom_quickreduce.cu`、`csrc/quickreduce/*` | `csrc/rocm/quickreduce/*`（新增 `csrc/rocm/quickreduce/torch_bindings.cpp`） |

- 删除：`csrc/libtorch_stable/{ops.h,torch_bindings.cpp,torch_utils.h,dispatch_utils.h}`、`csrc/cache.h`、`csrc/core/{batch_invariant.hpp,exception.hpp}`、`csrc/libtorch_stable/attention/attention_utils.cuh`、`csrc/libtorch_stable/mamba/static_switch.h`、`csrc/libtorch_stable/quantization/w8a8/int8/per_token_group_quant.cu`。
- 新增：`csrc/rocm/dispatch_utils.h`。同时改了 `CMakeLists.txt`、`cmake/cpu_extension.cmake`、`cmake/hipify.py`。

### 2.2 自定义算子注册表的增删

**C++ 侧 op schema（`csrc/**/torch_bindings.cpp`）**

| 变化 | op 名 | 实现位置 | 引入 PR |
| --- | --- | --- | --- |
| 新增 | `mnnvl_multimem_reduce_scatter` | `csrc/custom_all_gather_reduce_scatter.cu`、`csrc/custom_all_gather_reduce_scatter_ops.cpp`、`csrc/custom_all_gather_reduce_scatter.cuh` | [#55072](https://github.com/vllm-project/vllm/pull/55072) SM100/SM103 低 SM 占用 multimem reduce-scatter |
| 新增 | `moe_prepare_scatter` | `csrc/moe/moe_permute_unpermute_op.cu`、`csrc/moe/torch_bindings.cpp` | [#56685](https://github.com/vllm-project/vllm/pull/56685) Humming 集成 |
| 新增 | `fp8_scaled_mm_with_quant`、`quantize_fp8e4m3_vec`、`float8_linear_prepack_cpu` | `csrc/cpu/sgl-kernels/gemm_fp8_w8a8.cpp`、`csrc/cpu/sgl-kernels/moe_fp8_w8a8.cpp`、`csrc/cpu/torch_bindings.cpp` | [#49942](https://github.com/vllm-project/vllm/pull/49942) CPU FP8 W8A8 |
| 新增 | `greedy_argmax`、`fused_gumbel_argmax` | `csrc/cpu/sampling_kernels.cpp` | [#53913](https://github.com/vllm-project/vllm/pull/53913) CPU 向量化采样 |
| 新增 | `cpu_has_amx_fp8` | `csrc/cpu/cpu_isa.cpp` | [#49942](https://github.com/vllm-project/vllm/pull/49942)（[#58515](https://github.com/vllm-project/vllm/pull/58515) 让 CPU wheel 带上 AMX-FP8 构建） |
| 删除 | `allspark_w8a16_gemm`、`rearrange_kn_weight_as_n32k16_order` | 原 `csrc/libtorch_stable/quantization/gptq_allspark/*` | [#58001](https://github.com/vllm-project/vllm/pull/58001) |

**Python 侧 `torch.ops.vllm.*`（`direct_register_custom_op`）**

- 新增：
    - `dflash2_grouped_conv`（[#55960](https://github.com/vllm-project/vllm/pull/55960)）
    - `fused_qk_norm_mrope_and_unified_kv_cache_update`（[#50212](https://github.com/vllm-project/vllm/pull/50212)，ROCm MRoPE 融合）
    - `mrope`
    - `mhc_fused_post_pre_delayed_rms_norm_aiter`（[#58655](https://github.com/vllm-project/vllm/pull/58655)）
    - `qwen4_exp_qsa_prepare_with_output`、`qwen4_exp_amd_ple_ngram_ids`、`qwen4_exp_amd_ple_ngram_embedding_pinned`
    - `rocm_bf16x3_router_gemm_dispatch`（[#52668](https://github.com/vllm-project/vllm/pull/52668)）
    - `rocm_dsv41_mxfp8_wo_a_bmm`（[#58456](https://github.com/vllm-project/vllm/pull/58456)）
    - `rocm_mxfp8_block32_gemm`（[#58510](https://github.com/vllm-project/vllm/pull/58510)）
    - `xpu_apply_rotary_emb`、`xpu_layer_norm`、`xpu_fused_add_layer_norm`、`xpu_nemotron_layer_norm`、`xpu_fused_add_nemotron_layer_norm`（[#55721](https://github.com/vllm-project/vllm/pull/55721)、[#57172](https://github.com/vllm-project/vllm/pull/57172)）
- 删除：`fused_all_gather_flashinfer_fp4_matmul`、`fused_all_gather_flashinfer_scaled_matmul`、`fused_flashinfer_scaled_matmul_reduce_scatter`。它们随 AsyncTP pass 一起被移除（[#60517](https://github.com/vllm-project/vllm/pull/60517)）。

### 2.3 Attention 后端与 KV cache 格式

**后端注册表** `vllm/v1/attention/backends/registry.py`

| 变化 | 枚举 | 类 / 文件 | PR |
| --- | --- | --- | --- |
| 新增 | `ULTRAQUANT` | `vllm/v1/attention/backends/ultraquant_attn.py`；算子在 `vllm/v1/attention/ops/ultraquant/{format,reference,triton_dequant,triton_store,triton_unified_attention}.py`、`vllm/v1/attention/ops/flydsl_ultraquant_decode.py`、`vllm/v1/attention/ops/flydsl_kernels/ultraquant_decode_hd256.py` | [#57057](https://github.com/vllm-project/vllm/pull/57057) 4-bit KV（FP4 E2M1 + UE8M0 per-32 scale，K 做 Hadamard 旋转），decode 用 FlyDSL D=256，复用 TurboQuant 框架；新 cache dtype 为 `ultraquant_4bit`（`vllm/config/cache.py`） |
| 新增 | `ROCM_SEGMENTED_ATTN` | `vllm/v1/attention/backends/rocm_segmented_attn.py`、`vllm/v1/attention/ops/segmented_attention.py`、`vllm/model_executor/warmup/rocm_segmented_attn_autotune_warmup.py` | [#59132](https://github.com/vllm-project/vllm/pull/59132) token-major 分段 Triton attention，面向 RDNA3/3.5/4；新增配置 `enable_rocm_segmented_attn_autotune`（`vllm/config/kernel.py`） |
| 新增 | `FLASHMLA_MEGA_ATTN_DSV41` | `vllm/models/deepseek_v41/sparse_mla.py`（`FlashMLAMegaAttnBackend`）、`vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py` | [#56935](https://github.com/vllm-project/vllm/pull/56935) |
| 新能力 | `AttentionBackendEnum.register()` / `MambaAttentionBackendEnum.register()` | `vllm/v1/attention/backends/registry.py` | [#41091](https://github.com/vllm-project/vllm/pull/41091) 平台插件可以动态注册后端 |
| 新增 CPU 后端 | `zentorch_sdpa`、MLA prefill `zen_cpu_sdpa` | `vllm/v1/attention/backends/zentorch_sdpa.py`、`vllm/v1/attention/backends/mla/prefill/zen_cpu_sdpa.py` | [#54508](https://github.com/vllm-project/vllm/pull/54508)、[#54967](https://github.com/vllm-project/vllm/pull/54967) |
| 新增 | 平台可指定 MLA prefill 后端 | `vllm/platforms/interface.py` 等 | [#56959](https://github.com/vllm-project/vllm/pull/56959) |

**移除 cascade attention**：[#61126](https://github.com/vllm-project/vllm/pull/61126) `e1a5086256`

- 改动：`vllm/v1/attention/backend.py`，以及 `vllm/v1/attention/backends/{flash_attn,flash_attn_diffkv,flashinfer,flex_attention,rocm_aiter_fa,cpu_attn}.py` 等；`vllm/config/model.py`、`vllm/engine/arg_utils.py`（`--disable-cascade-attn` 改为 no-op）。

#### NVFP4 / FP8 KV cache 扩展

- [#46963](https://github.com/vllm-project/vllm/pull/46963) SM8x 和 SM12x 上通过 FlashInfer 使用 NVFP4 KV —— `vllm/v1/attention/backends/flashinfer.py`
- [#60620](https://github.com/vllm-project/vllm/pull/60620) SM90 + fa2 使用 NVFP4 KV —— `vllm/v1/attention/backends/flashinfer.py`
- [#60452](https://github.com/vllm-project/vllm/pull/60452) SM12x 上用 XQA 做 NVFP4 KV decode —— `vllm/v1/attention/backends/flashinfer.py`
- [#59300](https://github.com/vllm-project/vllm/pull/59300) MiniMax-M3 MSA 稀疏注意力支持 NVFP4 KV —— `csrc/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu`（提交时在 `csrc/libtorch_stable/`）、`vllm/models/minimax_m3/nvidia/{sparse_attention_msa,msa_cutlass_sparse_decode}.py`、`cmake/external_projects/fmha_sm100.cmake`
- [#56935](https://github.com/vllm-project/vllm/pull/56935) DeepSeek-V4.1 NVFP4 压缩 KV —— `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`、`vllm/models/deepseek_v41/common/ops/{cache_utils,fused_compress_quant_cache,fused_layout}.py`
- [#59246](https://github.com/vllm-project/vllm/pull/59246) SM90 上 NoPE-512 模型使用 `fp8_ds_mla` KV —— `vllm/v1/attention/backends/mla/flashmla_sparse.py`、`vllm/v1/attention/backend.py`
- [#58128](https://github.com/vllm-project/vllm/pull/58128) Triton DiffKV 与 MiMo-V2.6-Flash 使用 FP8 KV
- [#55557](https://github.com/vllm-project/vllm/pull/55557) Qwen4Exp QSA 路径主 KV 使用 fp8_e4m3
- [#60140](https://github.com/vllm-project/vllm/pull/60140) **GLM 系列默认改为 fp8 KV cache** —— `vllm/model_executor/models/config.py`

#### 稀疏 MLA / DSA indexer

- [#56254](https://github.com/vllm-project/vllm/pull/56254) DeepGEMM sparse MQA logits 接入 V4.1 indexer —— `vllm/model_executor/kernels/attention/dsa/sparse_mqa_logits.py`（新）、`vllm/model_executor/layers/sparse_mqa_indexer.py`（新）、`vllm/v1/attention/backends/mla/sparse_indexer.py`（新）、`vllm/v1/attention/backends/mla/indexer.py`、`vllm/config/attention.py`（`indexer_sparse_logits`）
- [#61023](https://github.com/vllm-project/vllm/pull/61023) sparse MLA prefill workspace 大小可配置（`sparse_mla_prefill_workspace_factor`）
- [#60029](https://github.com/vllm-project/vllm/pull/60029) FlashMLA sparse 的 Q heads 改为 pad 到 64（原来是 128）
- [#57885](https://github.com/vllm-project/vllm/pull/57885)、[#57458](https://github.com/vllm-project/vllm/pull/57458) 减少冗余的稀疏 attention 元数据操作
- [#58684](https://github.com/vllm-project/vllm/pull/58684) 异步调度下，FlashInfer SM90 sparse MLA plan 不再有 D2H 同步
- [#55385](https://github.com/vllm-project/vllm/pull/55385) SM90 上 GLM5Next 的 NoPE SparseMLA 接入 FA 与 FlashMLA
- [#59462](https://github.com/vllm-project/vllm/pull/59462) TokenSpeed MLA 支持 block-interleaved DCP；[#60345](https://github.com/vllm-project/vllm/pull/60345) 新增 `tokenspeed_mla_min_split_kv`（`vllm/config/attention.py`）

#### 其他 attention kernel

- [#58482](https://github.com/vllm-project/vllm/pull/58482) SM120 上 Triton split-K 段数按 occupancy 自适应 —— `vllm/v1/attention/backends/triton_attn.py`、`triton_attn_diffkv.py`
- [#59085](https://github.com/vllm-project/vllm/pull/59085) 一个 DiffKV split-KV program 覆盖整个 spec-verify
- [#58371](https://github.com/vllm-project/vllm/pull/58371) FlashInfer trtllm-gen 支持融合的多步 draft decode —— `vllm/v1/attention/backends/flashinfer.py`
- [#53175](https://github.com/vllm-project/vllm/pull/53175) Gemma4 FP8 KV 在 head dim 512 时的 FA4 后端选择（重新提交 #48666）—— `vllm/model_executor/layers/attention/attention.py` 等 40+ 文件
- [#57918](https://github.com/vllm-project/vllm/pull/57918) FlashInfer prefill 反量化 scratch 设上限
- [#58169](https://github.com/vllm-project/vllm/pull/58169) DCP 序列长度计算去掉 CPU-GPU 同步
- [#58457](https://github.com/vllm-project/vllm/pull/58457) 检查对齐后的 KV block size 是否被所有 attention 后端支持

### 2.4 MoE 算子与后端

#### 新后端 / 新结构

- [#54049](https://github.com/vllm-project/vllm/pull/54049) **FlashInfer CuteDSL MegaMoE** —— `vllm/model_executor/layers/fused_moe/flashinfer_moe_ep.py`（新）、`fused_moe/experts/flashinfer_moe_ep.py`（新）、`fused_moe/prepare_finalize/passthrough.py`（新）、`fused_moe/oracle/{mxfp4,nvfp4}.py`、`vllm/distributed/device_communicators/all2all.py`；删除了 `vllm/utils/flashinfer_moe_ep.py`
- [#58053](https://github.com/vllm-project/vllm/pull/58053) **DeepGEMM MegaMoE 后端抽象** —— `vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe.py`（新），被 `vllm/models/deepseek_v4/nvidia/model.py` 和 `vllm/models/kimi_k3/nvidia/model.py` 使用
- [#52101](https://github.com/vllm-project/vllm/pull/52101) **MoonEP**（BF16 均衡 EP）—— `fused_moe/experts/moonep_experts.py`（新）、`fused_moe/prepare_finalize/moonep.py`（新）、`fused_moe/oracle/unquantized.py`、`vllm/config/parallel.py`
- [#60079](https://github.com/vllm-project/vllm/pull/60079) MXFP4 oracle 新增 **CUTLASS W4A4** 后端和 BF16 激活回退 —— `fused_moe/oracle/mxfp4.py`
- [#50030](https://github.com/vllm-project/vllm/pull/50030) **per-token NVFP4 CuTe-DSL MoE** —— `fused_moe/experts/flashinfer_cutedsl_moe.py`、`fused_moe/oracle/nvfp4.py`
- [#57176](https://github.com/vllm-project/vllm/pull/57176) per-token NVFP4 MoE 后端改为显式选择；[#56740](https://github.com/vllm-project/vllm/pull/56740) per-token NVFP4 MoE 支持 ReLU2
- [#58877](https://github.com/vllm-project/vllm/pull/58877) OAI Triton MXFP4 MoE 支持 SM12x —— `fused_moe/experts/gpt_oss_triton_kernels_moe.py`
- [#56685](https://github.com/vllm-project/vllm/pull/56685) Humming MoE —— `fused_moe/experts/fused_humming_moe.py`、`vllm/model_executor/layers/quantization/utils/humming/*`（新目录）
- [#54024](https://github.com/vllm-project/vllm/pull/54024) zentorch MoE（CPU）—— `fused_moe/experts/zentorch_moe.py`（新）
- [#52668](https://github.com/vllm-project/vllm/pull/52668) ROCm bf16x3 router GEMM —— `fused_moe/router/bf16x3_router_gemm_rocm.py`（新）
- [#58234](https://github.com/vllm-project/vllm/pull/58234) **所有 MoE 模型统一改用 `GateLinear`** —— `fused_moe/router/gate_linear.py`，涉及 30+ 个模型文件

#### MoE 性能 / 通信

- [#49194](https://github.com/vllm-project/vllm/pull/49194) NCCL symmetric reduce-scatter 不再需要 staging —— `fused_moe/modular_kernel.py`、`prepare_finalize/naive_dp_ep.py`、`vllm/distributed/device_communicators/*`
- [#57995](https://github.com/vllm-project/vllm/pull/57995) FlashInfer one-sided all2all 支持 fp8 combine（环境变量 `VLLM_FLASHINFER_MOE_A2A_LOW_PRECISION_COMBINE`）
- [#58635](https://github.com/vllm-project/vllm/pull/58635) modular 路径上推迟 TRTLLM-Gen top-k finalize
- [#57405](https://github.com/vllm-project/vllm/pull/57405) 封装 TRT-LLM BF16 权重布局处理
- [#59731](https://github.com/vllm-project/vllm/pull/59731) 调优 MoE weighted-sum kernel 的 launch 配置 —— `fused_moe/moe_fused_mul_sum.py`
- [#58051](https://github.com/vllm-project/vllm/pull/58051) TritonExperts 跳过路由到非本地专家的 top-k 槽位
- [#59128](https://github.com/vllm-project/vllm/pull/59128) block-FP8 DeepGEMM experts 跳过 padding 部分的计算
- [#58880](https://github.com/vllm-project/vllm/pull/58880) 非 1 routed scaling 时使用融合的 MiniMax2 routing
- [#58720](https://github.com/vllm-project/vllm/pull/58720) `RoutedExperts.load_weights` 中专家映射改为索引查找
- [#57268](https://github.com/vllm-project/vllm/pull/57268) DeepGEMM warmup 复用 MoE workspace；[#53914](https://github.com/vllm-project/vllm/pull/53914) 修复 DeepGEMM FP8 workspace 过度分配
- [#56079](https://github.com/vllm-project/vllm/pull/56079) 分组 MoE routing 跳过 SP padding 行
- [#59340](https://github.com/vllm-project/vllm/pull/59340) Humming MoE 跳过 w13 GEMM 之前多余的输入拷贝

#### EP / EPLB / DeepEP

- [#52641](https://github.com/vllm-project/vllm/pull/52641) EPLB 考虑竞争的专家迁移批处理 —— `vllm/distributed/eplb/migration_scheduler.py`（新），配置项 `enable_migration_batching`
- [#57991](https://github.com/vllm-project/vllm/pull/57991) **DeepEPv2 默认自动选择 hybrid 模式**
- [#57236](https://github.com/vllm-project/vllm/pull/57236) EPLB、SharedExpert Overlap、DeepEPv2 可以同时使用；[#57210](https://github.com/vllm-project/vllm/pull/57210) DeepEPv2 加入 SP 支持列表
- [#57549](https://github.com/vllm-project/vllm/pull/57549) dspark draft 模型（DeepSeekV4）支持 EPLB —— `vllm/models/deepseek_v4/common/eplb_util.py`（新）
- [#58473](https://github.com/vllm-project/vllm/pull/58473) 修复 Elastic EP 扩缩容时的 EPLB 统计

### 2.5 量化 GEMM 与量化格式

#### Humming（新 kernel 库，`humming-kernels` 从 0.1.12 升到 0.1.17）

- [#56685](https://github.com/vllm-project/vllm/pull/56685) 集成 —— `vllm/model_executor/kernels/linear/mixed_precision/humming.py`、`kernels/linear/mxfp4/humming.py`、`kernels/linear/mxfp6/humming.py`（新）、`layers/quantization/utils/humming/{__init__,activation,linear,priority,schema}.py`（新）
- [#56997](https://github.com/vllm-project/vllm/pull/56997) **SM90 上 Humming 优先于 Marlin** —— `kernels/linear/__init__.py`、`fused_moe/oracle/{fp8,int_wna16,mxfp4,mxfp8,nvfp4}.py`、`humming/priority.py`
- [#46528](https://github.com/vllm-project/vllm/pull/46528) compressed-tensors 的 wNaM 非对称量化（zero_point）走 Humming
- [#57421](https://github.com/vllm-project/vllm/pull/57421) Marlin 与 Humming 共享持久 workspace

#### 新格式 / 新方案

- [#59604](https://github.com/vllm-project/vllm/pull/59604) ModelOpt 支持 IQ2_XS、IQ2_XXS、Q8_0（通过 b12x）—— `layers/quantization/modelopt_block_quant.py`（新）、`kernels/linear/nvfp4/b12x.py`、`fused_moe/b12x.py`、`vllm/utils/b12x.py`
- [#59815](https://github.com/vllm-project/vllm/pull/59815) compressed-tensors MXFP4 MoE + FP8 激活 —— `compressed_tensors_moe/compressed_tensors_moe_w4a8_mxfp4.py`、`schemes/compressed_tensors_w4a8_mxfp4.py`（新）
- [#48606](https://github.com/vllm-project/vllm/pull/48606) Quark 原生 W4A16 INT4/UINT4 —— `layers/quantization/quark/schemes/quark_w4a16_int4.py`（新）
- [#55684](https://github.com/vllm-project/vllm/pull/55684) online quantization API 支持对 linear 层重新量化（示例：MXFP8 → FP8 PTPC）—— `layers/quantization/online/{base,fp8,mxfp8,mxfp4,moe_base}.py`
- [#55686](https://github.com/vllm-project/vllm/pull/55686) shared expert 融合与 online `shared_expert` 量化可以同时使用
- [#52798](https://github.com/vllm-project/vllm/pull/52798) CT WNA16 MoE 统一用 N-first 权重格式
- [#57430](https://github.com/vllm-project/vllm/pull/57430) Kimi-K3 routed expert 量化；[#59431](https://github.com/vllm-project/vllm/pull/59431) CT 格式支持未量化 ngram
- [#57316](https://github.com/vllm-project/vllm/pull/57316) ModelOpt MXFP8 层可以加载预处理过的权重
- [#50814](https://github.com/vllm-project/vllm/pull/50814) 可选在加载时对 MXFP4 反量化（`VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD`）—— `kernels/linear/mxfp4/emulation.py`

#### 移除

- [#58001](https://github.com/vllm-project/vllm/pull/58001) AllSpark W8A16 —— 删除 `kernels/linear/mixed_precision/allspark.py`、`layers/quantization/utils/allspark_utils.py` 及 csrc 实现
- [#53585](https://github.com/vllm-project/vllm/pull/53585) 移除 `fp8.py` 里的 online 量化（改用 online shorthands）—— `layers/quantization/fp8.py`
- [#51800](https://github.com/vllm-project/vllm/pull/51800) 移除 Quark 特有的静默 online 量化 —— `layers/quantization/quark/quark.py`

#### 量化相关 kernel 优化

- [#55330](https://github.com/vllm-project/vllm/pull/55330) per-token-group 8-bit quant 的寄存器常驻路径 —— `csrc/quantization/w8a8/fp8/per_token_group_quant.cu`
- [#58194](https://github.com/vllm-project/vllm/pull/58194) 动态 per-tensor FP8 量化改用向量化的 flat abs-max —— `csrc/quantization/w8a8/fp8/common.cu`
- [#59800](https://github.com/vllm-project/vllm/pull/59800) 原生 per-token FP8 量化只做 value reduction —— `layers/quantization/input_quant_fp8.py`
- [#57377](https://github.com/vllm-project/vllm/pull/57377) C3x SM100 FP8 blockwise 的激活 scale pad 到 4 的倍数
- [#60434](https://github.com/vllm-project/vllm/pull/60434) 移除 LL CuTeDSL BF16 GEMM 的 workaround
- [#56063](https://github.com/vllm-project/vllm/pull/56063) Intel B70 上的 Triton W8A8 block-FP8 调优表 —— `layers/quantization/utils/configs/*Intel(R)_Arc(TM)_Pro_B70*`（新增 25 个 json）

### 2.6 通用算子：采样、Top-K、Norm/RoPE、融合

- **采样 / Top-K**
    - [#56346](https://github.com/vllm-project/vllm/pull/56346) persistent top-k 支持 sampled filtering —— `csrc/sampled_topk.cuh`（新）、`csrc/persistent_topk.cuh`、`csrc/topk.cu`
    - [#51367](https://github.com/vllm-project/vllm/pull/51367) Gumbel 采样改用 Murmur3 RNG —— `vllm/v1/worker/gpu/sample/gumbel.py`
    - [#58605](https://github.com/vllm-project/vllm/pull/58605) 减少 Triton sampler warmup 的冗余特化
    - [#57327](https://github.com/vllm-project/vllm/pull/57327) GLM 小 batch decode 使用 cooperative top-k
- **Triton 基础设施**
    - [#43048](https://github.com/vllm-project/vllm/pull/43048) **Triton kernel dispatcher** —— `vllm/triton_utils/dispatcher.py`（新）：非 CUDA 平台可以用 `register_kernels({...})` 替换 vLLM core 中的 Triton kernel
    - [#60502](https://github.com/vllm-project/vllm/pull/60502)、[#58769](https://github.com/vllm-project/vllm/pull/58769)、[#59285](https://github.com/vllm-project/vllm/pull/59285) 把 `tl.make_block_ptr` 迁移到 tensor descriptor（兼容 triton-main）—— `vllm/third_party/flash_linear_attention/ops/*`、`vllm/models/{kimi_k3,glm5next,minimax_m3,inkling}/**/ops/*`
    - [#58582](https://github.com/vllm-project/vllm/pull/58582) 启动时并行 warmup 已注册的 CUDA Triton kernel（`VLLM_TRITON_JIT_WARMUP_NUM_THREADS`）
    - [#58197](https://github.com/vllm-project/vllm/pull/58197)、[#58590](https://github.com/vllm-project/vllm/pull/58590) eager 模式下关闭 JIT warmup 和 JIT monitor；[#58685](https://github.com/vllm-project/vllm/pull/58685) JIT monitor 改为 hook TileLang 的 `JITImpl.compile`
- **Norm / RoPE**
    - [#54461](https://github.com/vllm-project/vllm/pull/54461) RMSNorm 匹配另一种 rsqrt 写法（compile pass）
    - `vllm/model_executor/layers/rotary_embedding/packed_qk_rope.py`（新）、`mrope_vit_setup.py`（新）
    - [#58651](https://github.com/vllm-project/vllm/pull/58651) KimiViT 每层 QK RoPE 融合成一个 in-place kernel
    - [#58526](https://github.com/vllm-project/vllm/pull/58526) MiniMax-M3 视觉塔使用 `triton_mrope`，并修复 int64 offset
    - [#60246](https://github.com/vllm-project/vllm/pull/60246) 所有 RoPE 持有者都已融合时跳过 M-RoPE warmup
- **多模态输入归一化**
    - [#56798](https://github.com/vllm-project/vllm/pull/56798) `mm_input_normal` 的 Triton kernel —— `vllm/model_executor/layers/fusion/mm_input_norm.py`（新）
    - 设备端 mm 归一化已覆盖 Qwen3VL/Qwen3.5/Qwen4Next（[#51289](https://github.com/vllm-project/vllm/pull/51289)）、GLM4V/GLM5Next（[#55389](https://github.com/vllm-project/vllm/pull/55389)）、Kimi K2.5/K3（[#59278](https://github.com/vllm-project/vllm/pull/59278)）、Llama Nemotron VL（[#57928](https://github.com/vllm-project/vllm/pull/57928)）
- **编译期融合**
    - [#59612](https://github.com/vllm-project/vllm/pull/59612) 3D 激活量化融合（用于 Pixtral）
    - [#57299](https://github.com/vllm-project/vllm/pull/57299) functionalized split slice 规范化
    - [#54857](https://github.com/vllm-project/vllm/pull/54857) ROCm MLA 双 RMSNorm + FP8 group quant 融合 —— `vllm/compilation/passes/fusion/rocm_aiter_fusion.py`
    - [#58099](https://github.com/vllm-project/vllm/pull/58099) ROCm AITER 静态 FP8 attention 输出融合
    - [#58098](https://github.com/vllm-project/vllm/pull/58098) ROCm BF16 AsyncTP 融合（后来被 [#60517](https://github.com/vllm-project/vllm/pull/60517) 移除）
- **其他**
    - [#55960](https://github.com/vllm-project/vllm/pull/55960) DFlash2 融合分组卷积
    - [#55190](https://github.com/vllm-project/vllm/pull/55190) CohereASR 相对注意力分数累加融合
    - [#58226](https://github.com/vllm-project/vllm/pull/58226) DiffusionGemma 单遍 sampler 统计 kernel
    - [#40337](https://github.com/vllm-project/vllm/pull/40337) late-interaction 评分用 flash-maxsim Triton kernel —— `vllm/v1/pool/flash_maxsim/*`（新），配置 `enable_flash_late_interaction=True`

### 2.7 通信算子 / 分布式 kernel

- [#55072](https://github.com/vllm-project/vllm/pull/55072) SM100/SM103 低 SM 占用的 multimem reduce-scatter（见 §2.2）—— `vllm/distributed/device_communicators/custom_all_reduce.py`
- [#57643](https://github.com/vllm-project/vllm/pull/57643) DSv4.1 TP all-reduce 与 mHC 输入准备融合，先用 CUDA 实现 `csrc/libtorch_stable/all_reduce_mhc.cu`；之后 [#58586](https://github.com/vllm-project/vllm/pull/58586) 把 MoE finalize 也融合进来，改为 CuTe-DSL 实现 `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/{all_reduce_mhc,primitives}.py`，**删除了该 `.cu` 文件**（v0.32.0 中已不存在）
- [#57428](https://github.com/vllm-project/vllm/pull/57428) MXFP8 wo_b GEMM 与 SP reduce-scatter 融合 —— `vllm/model_executor/kernels/linear/cute_dsl/gemm_rs_ar.py`（新，从 `vllm/models/kimi_k3/nvidia/ops/cute_dsl/gemm_rs_ar.py` 移过来），环境变量从 `VLLM_KIMI_K3_GEMM_RS` 改名为 `VLLM_ENABLE_GEMM_RS`
- [#58623](https://github.com/vllm-project/vllm/pull/58623) `VLLM_BATCH_INVARIANT` 下启用 custom all-reduce
- [#61989](https://github.com/vllm-project/vllm/pull/61989) 单机 FlashInfer allreduce 重新优先使用 MNNVL —— `vllm/distributed/device_communicators/flashinfer_all_reduce.py`
- [#59360](https://github.com/vllm-project/vllm/pull/59360) `--enable-nccl-comm-suspend` 时释放 FlashInfer allreduce workspace
- [#51207](https://github.com/vllm-project/vllm/pull/51207) **shm tensor arena**（CPU→GPU worker 广播）—— `vllm/distributed/device_communicators/shm_tensor_arena.py`（新），参数 `--enable-shm-tensor-arena`

### 2.8 Mamba / 线性注意力 / GDN / KDA

- [#57329](https://github.com/vllm-project/vllm/pull/57329) Mamba2 内部 prefill checkpoint —— `vllm/model_executor/layers/mamba/checkpoint.py`（新）、`mamba/ops/ssd_checkpoint.py`（新）
- [#57783](https://github.com/vllm-project/vllm/pull/57783) 通用的 prefill checkpoint builder/exporter，扩展到 KDA —— `mamba/kda_checkpoint.py`（新）；[#56960](https://github.com/vllm-project/vllm/pull/56960) GLM-5.3-Flash 启用 KDA prefill checkpoint
- [#52928](https://github.com/vllm-project/vllm/pull/52928) MTP 支持 FlashInfer ReplaySSM —— `mamba/ops/ssu_dispatch.py`、`vllm/model_executor/warmup/replayssm_warmup.py`
- [#49371](https://github.com/vllm-project/vllm/pull/49371) Mamba2 prefill SSM state 批量保存，去掉 GPU↔CPU 同步
- [#57881](https://github.com/vllm-project/vllm/pull/57881) BreakableCG：Mamba 系列算子加入 eager break 点
- [#57140](https://github.com/vllm-project/vllm/pull/57140)、[#58763](https://github.com/vllm-project/vllm/pull/58763)、[#59735](https://github.com/vllm-project/vllm/pull/59735) GDN 投机输出写入调用方 buffer；纯 spec-decode 行改为切片；非投机 decode 保持标准路径
- [#58846](https://github.com/vllm-project/vllm/pull/58846) FlashKDA 递归状态保持 fp32
- [#54255](https://github.com/vllm-project/vllm/pull/54255) Kimi-K3 FlashInfer 投机 KDA 后端 —— `vllm/models/kimi_k3/nvidia/kda.py`
- [#58997](https://github.com/vllm-project/vllm/pull/58997) **移除 `mamba_cache_mode="all"`**
- [#57382](https://github.com/vllm-project/vllm/pull/57382) `--enable-mamba-fine-grained-prefix-cache` 改名为 `--enable-mamba-shared-prefix-checkpoint`
- XPU / Intel：新增 Mamba SSU 调优配置 `mamba/ops/configs/selective_state_update/*Intel(R)_Arc(TM)_Pro_B60*|B70*`（[#56765](https://github.com/vllm-project/vllm/pull/56765)、[#57565](https://github.com/vllm-project/vllm/pull/57565)）

### 2.9 Batch Invariance（确定性推理）

- [#52231](https://github.com/vllm-project/vllm/pull/52231) **ROCm 支持 batch invariance** —— `vllm/model_executor/determinism/batch_invariant.py`、`kernels/linear/scaled_mm/triton.py` 等
- [#55881](https://github.com/vllm-project/vllm/pull/55881) XPU 支持 `VLLM_BATCH_INVARIANT`（Dense/MoE）
- [#57456](https://github.com/vllm-project/vllm/pull/57456)、[#58495](https://github.com/vllm-project/vllm/pull/58495) sm_120 的 batch-invariant persistent matmul 调优表（含 TP=2/4/8）—— `vllm/model_executor/determinism/batch_invariant_configs.py`
- [#57586](https://github.com/vllm-project/vllm/pull/57586) `VLLM_BATCH_INVARIANT` 下默认使用 breakable CUDA graph，不走 torch.compile
- [#59377](https://github.com/vllm-project/vllm/pull/59377) LoRA shrink 使用确定性 split-K=8 —— `vllm/lora/ops/triton_ops/{kernel_utils,lora_shrink_op}.py`

---

## 3. 模型层面的变化

> 大模型的专用实现在 `vllm/models/<model>/{common,nvidia,amd}/`，通用模型在 `vllm/model_executor/models/`。

### 3.1 DeepSeek-V4 / V4.1（改动最多）

- **注意力 / KV**：[#56935](https://github.com/vllm-project/vllm/pull/56935) FlashMLA mega attention 与 NVFP4 压缩 KV（见 §2.3）；[#60064](https://github.com/vllm-project/vllm/pull/60064) 加载时 permute `wq_b`/`wo_a`
- **SWA bounded replay**（新功能，默认开启 `swa_bounded_replay=True`，在 `vllm/config/cache.py`）
    - encoder 侧：[#56227](https://github.com/vllm-project/vllm/pull/56227) —— `vllm/models/deepseek_v41/sparse_mla.py`、`vllm/v1/attention/backends/mla/{compressor_utils,indexer}.py`
    - decoder 侧：[#58132](https://github.com/vllm-project/vllm/pull/58132) —— `vllm/models/deepseek_v41/decoder_replay_layers.py`（新）
    - CUDA graph 化：[#59532](https://github.com/vllm-project/vllm/pull/59532) —— `vllm/models/deepseek_v41/nvidia/decoder_replay_cudagraph.py`（新）；`vllm/config/compilation.py` 新增 `cudagraph_decoder_replay`、`decoder_replay_cudagraph_capture_sizes`、`decoder_replay_trim_threshold`；[#59894](https://github.com/vllm-project/vllm/pull/59894) trim 最后一个 KV-source 层
- **MoE**：[#56266](https://github.com/vllm-project/vllm/pull/56266) 接入 DeepGEMM Mega-Gate（`vllm/utils/deep_gemm.py`）；[#56568](https://github.com/vllm-project/vllm/pull/56568) 先为 MegaMoE 融合 pad shared experts，后由 [#57204](https://github.com/vllm-project/vllm/pull/57204) 移除这层 padding；[#57604](https://github.com/vllm-project/vllm/pull/57604) MegaMoE staging 优化 —— `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`；[#59927](https://github.com/vllm-project/vllm/pull/59927) shared-expert finalize 不再依赖 linear post-load 的顺序；[#57465](https://github.com/vllm-project/vllm/pull/57465) 修复专家分布；删除 `vllm/models/deepseek_v4/nvidia/fi_moe.py`
- **mHC / TP 融合**：[#57603](https://github.com/vllm-project/vllm/pull/57603)、[#57643](https://github.com/vllm-project/vllm/pull/57643)、[#58586](https://github.com/vllm-project/vllm/pull/58586) —— `vllm/models/deepseek_v41/nvidia/ops/{mhc,mega_mhc}.py`
- **WO-A / O-proj**：[#58634](https://github.com/vllm-project/vllm/pull/58634) SM100/SM103 小 batch 下 WO-A + inverse RoPE + MXFP8 quant 融合 —— `vllm/models/deepseek_v41/nvidia/ops/fused_wo_a.py`（新）、`ops/o_proj.py`（新）；[#57679](https://github.com/vllm-project/vllm/pull/57679) 恢复融合的 query RMSNorm + MXFP8 quant；[#58621](https://github.com/vllm-project/vllm/pull/58621) FlashInfer sparse MLA 中融合 inverse RoPE + FP8 quant
- **Engram**（host 表查找）：[#56926](https://github.com/vllm-project/vllm/pull/56926) huge pages；[#59327](https://github.com/vllm-project/vllm/pull/59327) 行排序；[#58678](https://github.com/vllm-project/vllm/pull/58678) wkv 按 TP 切分；[#59923](https://github.com/vllm-project/vllm/pull/59923) DP 间交换 packed FP8 行 —— `vllm/models/deepseek_v41/common/engram.py`、`nvidia/ops/engram_peer.py`（新）；删除 `nvidia/engram.py`；[#57651](https://github.com/vllm-project/vllm/pull/57651) 同机 DP 副本默认共享 host 表（`vllm/config/engram.py`，`cpu_offload` 默认 True）；删除 `VLLM_PLE_CPU_OFFLOAD`（[#57937](https://github.com/vllm-project/vllm/pull/57937)）
- **视觉**：[#56625](https://github.com/vllm-project/vllm/pull/56625) encoder CUDA graph —— `vllm/models/deepseek_v41/common/vl_cudagraph.py`（新）
- **DSA indexer**：[#56254](https://github.com/vllm-project/vllm/pull/56254)（见 §2.3）
- **ROCm 专用**（`vllm/models/deepseek_v4{,1}/amd/`）
    - [#60397](https://github.com/vllm-project/vllm/pull/60397) **MI355X Mono decode layer** —— `vllm/models/deepseek_v41/amd/mono/**`（新目录，约 25 个文件）、`amd/mono_decode.py`，开关 `VLLM_ROCM_MONO_DECODE`
    - [#59685](https://github.com/vllm-project/vllm/pull/59685) AITER MegaMoEV2 —— `vllm/models/deepseek_v4/amd/mega_moe.py`（新）；[#61004](https://github.com/vllm-project/vllm/pull/61004) shared expert 融合进 MegaMoE（`VLLM_ROCM_AITER_MEGA_MOE_FUSE_SHARED_EXPERTS`）
    - [#58819](https://github.com/vllm-project/vllm/pull/58819) 可选的 a4w4 MoE（`VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4`）
    - [#58671](https://github.com/vllm-project/vllm/pull/58671) paged MXFP4 sparse indexer —— `vllm/v1/attention/ops/rocm_paged_mxfp4_indexer.py`、`vllm/v1/attention/backends/mla/rocm_paged_mxfp4_indexer.py`、`vllm/model_executor/layers/rocm_paged_mxfp4_indexer.py`（均为新文件）
    - [#56853](https://github.com/vllm-project/vllm/pull/56853)、[#57407](https://github.com/vllm-project/vllm/pull/57407) 多流 overlap（[#58983](https://github.com/vllm-project/vllm/pull/58983) 把开关移到 ROCm platform）
    - [#57435](https://github.com/vllm-project/vllm/pull/57435)、[#57451](https://github.com/vllm-project/vllm/pull/57451)、[#58456](https://github.com/vllm-project/vllm/pull/58456) inverse RoPE 融合进 sparse decode reduce，并输出 MXFP8
    - [#53492](https://github.com/vllm-project/vllm/pull/53492) AITER sparse MLA Gluon kernel；[#57132](https://github.com/vllm-project/vllm/pull/57132) 回滚 #56433 和 #51692，修复精度问题

### 3.2 GLM-5.x（GLM-5.3-Flash / GLM5Next）

- [#55358](https://github.com/vllm-project/vllm/pull/55358) **目录重构**：`vllm/models/glm5next/nvidia/{attention,kda,model,mtp,multimodal}.py` 移到 `vllm/models/glm5next/common/`；新增 `glm5next/{amd,nvidia,common}/sparse_indexer.py`；删除 `vllm/model_executor/layers/sparse_attn_indexer_kpool.py`
- [#59211](https://github.com/vllm-project/vllm/pull/59211) kpool sparse indexer 支持 DCP —— `vllm/model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py`
- 性能
    - [#57443](https://github.com/vllm-project/vllm/pull/57443) 融合多步 decode（并发 1 时 E2E +13.3%）
    - [#59464](https://github.com/vllm-project/vllm/pull/59464) 跨层复用 sparse MLA 索引转换
    - [#58450](https://github.com/vllm-project/vllm/pull/58450) metadata op 优化
    - [#58845](https://github.com/vllm-project/vllm/pull/58845) MHA 跳过 qlnorm
    - [#57701](https://github.com/vllm-project/vllm/pull/57701) decode workspace 节省 3 GiB
    - [#54951](https://github.com/vllm-project/vllm/pull/54951) 长上下文 indexer prefill 按 TP 切分
    - [#57534](https://github.com/vllm-project/vllm/pull/57534) kpool tail slot mapping 融合成单个 Triton kernel
    - [#60140](https://github.com/vllm-project/vllm/pull/60140) 默认使用 fp8 KV
- [#56983](https://github.com/vllm-project/vllm/pull/56983) 支持 DFlash2 草稿模型；[#57169](https://github.com/vllm-project/vllm/pull/57169) 使用通用 packed KV 布局 + `CircularBufferSpec`
- ROCm：[#58167](https://github.com/vllm-project/vllm/pull/58167) AITER topk；[#58584](https://github.com/vllm-project/vllm/pull/58584) BF16 splitk sparse MLA decode；[#57979](https://github.com/vllm-project/vllm/pull/57979) stride-aware KDA；[#59221](https://github.com/vllm-project/vllm/pull/59221) fused shared experts（gfx950）；[#51915](https://github.com/vllm-project/vllm/pull/51915)、[#58904](https://github.com/vllm-project/vllm/pull/58904) GLM-5.2 MXFP4 / MTP 修复

### 3.3 Kimi-K3

- [#54255](https://github.com/vllm-project/vllm/pull/54255) FlashInfer 投机 KDA；[#52988](https://github.com/vllm-project/vllm/pull/52988) adaptive verification 的变长 decode —— `vllm/models/kimi_k3/nvidia/{dspark_mla,kda_metadata}.py`、`vllm/v1/attention/backends/mla/*.py`
- [#56875](https://github.com/vllm-project/vllm/pull/56875) DeepEP v2 下 shared expert 分片；[#56881](https://github.com/vllm-project/vllm/pull/56881) DCP prefill 使用融合的 K/V pack kernel
- [#58527](https://github.com/vllm-project/vllm/pull/58527) 视觉 patch embedder 走 GEMM
- ROCm：[#53940](https://github.com/vllm-project/vllm/pull/53940) a4w4 flydsl；[#54956](https://github.com/vllm-project/vllm/pull/54956)、[#59591](https://github.com/vllm-project/vllm/pull/59591) latent MoE 上投影分片；[#57640](https://github.com/vllm-project/vllm/pull/57640) MLA decode KV 写入 + Q-prep 融合；[#58344](https://github.com/vllm-project/vllm/pull/58344) prefill checkpoint（`vllm/models/kimi_k3/amd/ops/kda_checkpoint.py`）；[#59069](https://github.com/vllm-project/vllm/pull/59069) AttnRes 输出 + FP8 量化融合；[#51274](https://github.com/vllm-project/vllm/pull/51274) gfx942 上 MXFP4 → int4；[#58201](https://github.com/vllm-project/vllm/pull/58201) `VLLM_ROCM_USE_AITER_MOE_SITUV2` 用来选择 a4w4/a8w4/a16w4（替代 `..._SITUV2_A8W4`）

### 3.4 Qwen4Exp / Qwen3.8-Flash-Next

- [#57097](https://github.com/vllm-project/vllm/pull/57097) QK-norm/RoPE/gate 与 KV 写入融合进 QSA pre-indexer —— `vllm/models/qwen4_exp/nvidia/ops/qsa_prepare.py`（原名 `qsa_pre_indexer.py`）；AMD 版本见 [#59732](https://github.com/vllm-project/vllm/pull/59732) 和 `vllm/models/qwen4_exp/amd/ops/qsa_prepare.py`（新）
- [#59533](https://github.com/vllm-project/vllm/pull/59533) 合并 QSA QKVG 与 indexer QK 投影；[#58957](https://github.com/vllm-project/vllm/pull/58957) HC down 投影 + SiLU 融合 —— `vllm/models/qwen4_exp/nvidia/ops/cute_dsl/hc_down_silu*.py`（新）
- low-latency / skinny GEMM 方案：SM90（[#57273](https://github.com/vllm-project/vllm/pull/57273)）、SM100（[#59214](https://github.com/vllm-project/vllm/pull/59214)）、SM121（[#59753](https://github.com/vllm-project/vllm/pull/59753)、[#59632](https://github.com/vllm-project/vllm/pull/59632)）、H200（[#60776](https://github.com/vllm-project/vllm/pull/60776)）—— `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py`
- PLE：`vllm/models/qwen4_exp/nvidia/ops/ple.py` 移到 `common/ops/ple.py`；新增 `common/ngram_embedding.py`；[#57497](https://github.com/vllm-project/vllm/pull/57497) PLE n-gram 表 CPU offload；删除 `vllm/models/qwen4_exp/config.py` 与 `vllm/transformers_utils/configs/qwen4_exp.py`（改用上游 config，[#57387](https://github.com/vllm-project/vllm/pull/57387)）
- [#56322](https://github.com/vllm-project/vllm/pull/56322) 序列并行（`--enable-hc-sp`）；[#55867](https://github.com/vllm-project/vllm/pull/55867) FP8 TP + FlashInfer TRTLLM MoE

### 3.5 MiniMax-M3

- [#59300](https://github.com/vllm-project/vllm/pull/59300) NVFP4 KV；[#60684](https://github.com/vllm-project/vllm/pull/60684) MSA prefill 使用 E4M3 query
- [#59081](https://github.com/vllm-project/vllm/pull/59081) 非 SM100 上 Triton indexer 使用 fp8 cache；[#56151](https://github.com/vllm-project/vllm/pull/56151) 调整 decode grid，加入 SM12.0 split-K；[#59481](https://github.com/vllm-project/vllm/pull/59481) 保留原生 FP8 MMA
- [#58673](https://github.com/vllm-project/vllm/pull/58673) encoder CUDA graph —— `vllm/models/minimax_m3/common/encoder_cudagraph.py`（新）；[#58512](https://github.com/vllm-project/vllm/pull/58512) patch embedding 改用 Conv3dLayer
- ROCm：[#54535](https://github.com/vllm-project/vllm/pull/54535) packed LBHNC QK-norm 融合；[#56849](https://github.com/vllm-project/vllm/pull/56849) sparse-PA K/V 插入不再需要 contiguous copy

### 3.6 新增模型 / 架构（`vllm/model_executor/models/registry.py`）

| 架构 | 文件 | PR |
| --- | --- | --- |
| `MiniCPMV4_7ForConditionalGeneration` | `vllm/model_executor/models/minicpmv4_7.py`，以及 `vllm/parser/minicpmv.py`、`vllm/reasoning/minicpmv_reasoning_parser.py` | [#58674](https://github.com/vllm-project/vllm/pull/58674) |
| `EmbeddingGemma2Model` | `vllm/model_executor/models/embedding_gemma2.py` | [#60254](https://github.com/vllm-project/vllm/pull/60254)、[#60289](https://github.com/vllm-project/vllm/pull/60289) |
| `ClefOmniForJointSchema`（Cloudflare clef-omni，用于 `/v1/systemone`） | `vllm/model_executor/models/clef_omni.py` | [#61064](https://github.com/vllm-project/vllm/pull/61064) |
| `LiLiCorrDraftModel`（投机草稿） | `vllm/model_executor/models/lilicorr.py` | [#57934](https://github.com/vllm-project/vllm/pull/57934) |
| `ExaoneMoEForCausalLM`（只是别名：让 K-EXAONE 2.0 之前的 config 仍能加载，实现复用 `ExaoneMoeForCausalLM`） | `vllm/model_executor/models/exaone_moe.py` | [#60187](https://github.com/vllm-project/vllm/pull/60187) |
| `Nemotron3_5AsrForRNNT`（外部插件） | registry 中指向外部插件仓库 | [#59827](https://github.com/vllm-project/vllm/pull/59827) |
| zerank-2 reranker | 仅新增测试和文档（复用已有架构） | [#60415](https://github.com/vllm-project/vllm/pull/60415) |
| Hy4 的 ROCm 路径 | `vllm/models/hy_v4/amd/model.py` | [#57526](https://github.com/vllm-project/vllm/pull/57526) |
| DiffusionGemma 的结构化生成 / 调度 | `vllm/model_executor/models/diffusion_gemma_sampler.py`、`vllm/v1/core/sched/diffusion_scheduler.py`、`vllm/utils/diffusion.py` | [#57250](https://github.com/vllm-project/vllm/pull/57250) 等 |

**迁移到 Transformers modeling backend（删除了 vLLM 自己的实现）**：[#59679](https://github.com/vllm-project/vllm/pull/59679)、[#59701](https://github.com/vllm-project/vllm/pull/59701)

- `GlmForCausalLM`、`ArceeForCausalLM`、`CwmForCausalLM`、`MellumForCausalLM`、`GPTNeoXForCausalLM`、`PhiForCausalLM`、`SeedOssForCausalLM`、`Jais2ForCausalLM` 现在都映射到 `transformers` 的 `TransformersForCausalLM` / `TransformersMoEForCausalLM`
- 删除：`vllm/model_executor/models/{glm,arcee,mellum,gpt_neox,phi,seed_oss,jais2}.py`

**改用上游 Transformers 的 config / processor**（[#57387](https://github.com/vllm-project/vllm/pull/57387)、[#60623](https://github.com/vllm-project/vllm/pull/60623)、[#59762](https://github.com/vllm-project/vllm/pull/59762)）

- 删除 `vllm/transformers_utils/configs/`：`AXK1, afmoe, cosmos3, diffusion_gemma, gemma4, glm5_next, granite4_vision, hy_v3, hyperclovax, kimi_k25, laguna, lfm2_moe, mellum, nemotron, nemotron_h, olmo_hybrid, qwen3_5, qwen3_5_moe, qwen3_next, qwen4_exp`
- 删除 `vllm/transformers_utils/processors/{glm5next,granite4_vision,qwen3_asr}.py`
- Transformers 后端支持视频（[#57441](https://github.com/vllm-project/vllm/pull/57441)）—— `vllm/model_executor/models/transformers/multimodal.py`

**LoRA 新增支持**：Nemotron VL（[#56231](https://github.com/vllm-project/vllm/pull/56231)）、ModernBert（[#57148](https://github.com/vllm-project/vllm/pull/57148)）、VoyageQwen3 Bidirectional Embed（[#57708](https://github.com/vllm-project/vllm/pull/57708)）、RobertaForSequenceClassification（[#58884](https://github.com/vllm-project/vllm/pull/58884)）、BERT/RoBERTa embedding（[#61033](https://github.com/vllm-project/vllm/pull/61033)）；序列分类支持 `modules_to_save`（[#53555](https://github.com/vllm-project/vllm/pull/53555)）和可变 `num_labels`（[#57766](https://github.com/vllm-project/vllm/pull/57766)，`--max-lora-cls-labels`）—— `vllm/lora/layers/classifier.py`（新）

**其他**：Eagle3 / DFlash 的 aux hidden states 支持扩展到 Cohere2MoE（[#49819](https://github.com/vllm-project/vllm/pull/49819)）、Granite（[#60855](https://github.com/vllm-project/vllm/pull/60855)）、Qwen3ASR（[#52824](https://github.com/vllm-project/vllm/pull/52824)）；Sarvam MLA 支持 EAGLE3/DSpark PP（[#55902](https://github.com/vllm-project/vllm/pull/55902)）；MiMo V2 支持 bf16 router 和 mxfp4 MoE（[#57784](https://github.com/vllm-project/vllm/pull/57784)）；Gemma4 改用 AutoWeightsLoader（[#55911](https://github.com/vllm-project/vllm/pull/55911)）。

---

## 4. 功能层面的变化

### 4.1 投机解码（Spec Decode）

- **MRV2 草稿模型投机**：[#43091](https://github.com/vllm-project/vllm/pull/43091) 新增 `spec_decode/draft_model/speculator.py`，后来被 [#60335](https://github.com/vllm-project/vllm/pull/60335) 改名为 **`vllm/v1/worker/gpu/spec_decode/standalone_ar/speculator.py`**（v0.32.0 中的位置）
- **MRV2 GPU N-gram**：[#40704](https://github.com/vllm-project/vllm/pull/40704) —— `vllm/v1/worker/gpu/spec_decode/ngram/speculator.py`（新）
- **LiLiCorr drafter**：[#57934](https://github.com/vllm-project/vllm/pull/57934) —— `vllm/v1/worker/gpu/spec_decode/lilicorr/speculator.py`（新）、`vllm/model_executor/models/lilicorr.py`
- **重命名**：[#60335](https://github.com/vllm-project/vllm/pull/60335) `spec_decode/autoregressive/` 改为 `spec_decode/target_dependent_ar/`；`spec_decode/draft_model/` 改为 `spec_decode/standalone_ar/`
- DFlash：[#58065](https://github.com/vllm-project/vllm/pull/58065) 支持异步调度（`vllm/config/vllm.py`）；[#57632](https://github.com/vllm-project/vllm/pull/57632) 上下文 K/V precompute 纳入 draft CUDA graph；[#60692](https://github.com/vllm-project/vllm/pull/60692) 所有后端都支持时，dflash/dspark 跳过 metadata 构建
- MTP：[#59359](https://github.com/vllm-project/vllm/pull/59359) MRV2 MTP 支持 sampling mask replay；[#58463](https://github.com/vllm-project/vllm/pull/58463) 融合多步 decode 时去掉 eager metadata 重建
- DSpark：[#52362](https://github.com/vllm-project/vllm/pull/52362) ROCm adaptive verification；[#57263](https://github.com/vllm-project/vllm/pull/57263) Gemma4 + FlashInfer；[#56723](https://github.com/vllm-project/vllm/pull/56723) DCP 目标模型 + 非 DCP DSpark；[#56956](https://github.com/vllm-project/vllm/pull/56956) PP 目标模型支持在 [#58484](https://github.com/vllm-project/vllm/pull/58484) 被**回滚**
- 其他：[#57396](https://github.com/vllm-project/vllm/pull/57396) 异构词表投机解码去掉 CPU-GPU 同步；[#57107](https://github.com/vllm-project/vllm/pull/57107) 接受率估计器避免 triton 重新编译；[#61240](https://github.com/vllm-project/vllm/pull/61240) 正确使用 draft model config；[#56807](https://github.com/vllm-project/vllm/pull/56807) 水印的上下文去重支持投机解码

### 4.2 Model Runner V2（`vllm/v1/worker/gpu/`）

- [#56497](https://github.com/vllm-project/vllm/pull/56497) **自定义 logits processor** —— `vllm/v1/worker/gpu/sample/logits_processor/{interface,loader}.py`（新）；[#57728](https://github.com/vllm-project/vllm/pull/57728) 入口校验；[#50584](https://github.com/vllm-project/vllm/pull/50584) 新增 DRY 示例
- [#59079](https://github.com/vllm-project/vllm/pull/59079) 支持 stock torch.compile 模式 —— `vllm/compilation/wrapper.py`
- [#53934](https://github.com/vllm-project/vllm/pull/53934) Elastic EP 支持 MRV2；[#59139](https://github.com/vllm-project/vllm/pull/59139) MRV2 支持 PP + PCP —— `vllm/v1/worker/gpu/pcp_manager.py`
- [#58411](https://github.com/vllm-project/vllm/pull/58411) 随机化 dummy 输入；[#58400](https://github.com/vllm-project/vllm/pull/58400) 单 token prompt 尾部也可以用 FULL decode graph
- [#57102](https://github.com/vllm-project/vllm/pull/57102)、[#58762](https://github.com/vllm-project/vllm/pull/58762) 多个 KV cache group 共享 token→request 映射和 Mamba/GDN metadata
- [#60215](https://github.com/vllm-project/vllm/pull/60215) `NonUvaBuffer` H2D 拷贝不再 stream sync；[#57416](https://github.com/vllm-project/vllm/pull/57416) prefill-only batch 的 logits 行数优化
- [#60937](https://github.com/vllm-project/vllm/pull/60937) 权重 reload 逻辑与 MRV1 解耦；[#57834](https://github.com/vllm-project/vllm/pull/57834) shutdown 时释放 weight offloader
- [#55145](https://github.com/vllm-project/vllm/pull/55145) XPU 上 MRV2+PP 的 microbatch 开关（`VLLM_XPU_PP_MICROBATCH`）；[#57277](https://github.com/vllm-project/vllm/pull/57277) XPU 采样 kernel
- [#59200](https://github.com/vllm-project/vllm/pull/59200) GPU 与 XPU worker 共享 workspace 和 model runner 初始化
- [#61014](https://github.com/vllm-project/vllm/pull/61014) 原本还跑在 MRV1 上的 CI 测试切到 MRV2；[#56360](https://github.com/vllm-project/vllm/pull/56360) 移除 ROCm 的 MRV1 模型例外

### 4.3 Core / 调度器 / 引擎

- [#56758](https://github.com/vllm-project/vllm/pull/56758) **`--max-num-active-seqs`**（限制进入 RUNNING 的请求数）—— `vllm/config/scheduler.py`、`vllm/v1/core/sched/scheduler.py`
- [#57951](https://github.com/vllm-project/vllm/pull/57951)、[#58459](https://github.com/vllm-project/vllm/pull/58459) **`--long-prefill-token-threshold-adaptive`**
- [#58947](https://github.com/vllm-project/vllm/pull/58947) 重写 `skipped_waiting` 队列 —— `vllm/v1/core/sched/request_queue.py`
- [#53936](https://github.com/vllm-project/vllm/pull/53936) n>1 并行采样请求的准入变为原子操作
- [#60269](https://github.com/vllm-project/vllm/pull/60269) 降低空闲 KV cache 队列的开销；[#60497](https://github.com/vllm-project/vllm/pull/60497) fallback decode 在 32K 下 CPU 开销降低 69%
- prefix cache：[#59335](https://github.com/vllm-project/vllm/pull/59335) block hash 包含 LoRA path；[#51899](https://github.com/vllm-project/vllm/pull/51899) extra keys 按来源打标签
- [#53423](https://github.com/vllm-project/vllm/pull/53423) **KV hints 请求封装**（程序化 KV 管理）—— `vllm/v1/kv_hints/{__init__,protocol}.py`（新），以及 Rust 侧 `rust/src/engine-core-client/src/protocol/kv_hints.rs`
- [#45635](https://github.com/vllm-project/vllm/pull/45635) **AuxOutput**（routed-experts 输出按 block 存储）—— `vllm/config/aux_output.py`、`vllm/distributed/aux_output_connector/{connector,routed_experts,store,worker}.py`（新），参数 `--aux-output-config`；`enable_return_routed_experts` 从 ModelConfig 移到了新 config
- [#57226](https://github.com/vllm-project/vllm/pull/57226) 引擎握手期间不再发阻塞的 collective RPC；[#58779](https://github.com/vllm-project/vllm/pull/58779)、[#60398](https://github.com/vllm-project/vllm/pull/60398) draft-token 和 dummy-batch 的 RPC 等待受 execute-model 超时约束
- [#58946](https://github.com/vllm-project/vllm/pull/58946) UniProc EngineCore 启动线程数受可用 CPU 限制；[#59240](https://github.com/vllm-project/vllm/pull/59240) 按 Rust client 的方式合并多 engine 的 utility 结果
- [#57250](https://github.com/vllm-project/vllm/pull/57250) DiffusionGemma 结构化生成模式 —— `vllm/v1/core/sched/diffusion_scheduler.py`

### 4.4 KV Connector / P/D 分离 / KV Offload / EC

- NIXL：[#50494](https://github.com/vllm-project/vllm/pull/50494)、[#50499](https://github.com/vllm-project/vllm/pull/50499) PP push prefill 支持 attention-HMA 和 packed MLA 布局；[#54483](https://github.com/vllm-project/vllm/pull/54483) host buffer KV 拷贝跨 cache group 合并；[#55854](https://github.com/vllm-project/vllm/pull/55854)、[#58875](https://github.com/vllm-project/vllm/pull/58875) 传输失败与 KV 过期分开统计；[#60161](https://github.com/vllm-project/vllm/pull/60161) HiSparse host pool 只在单个 TP rank 注册；[#56907](https://github.com/vllm-project/vllm/pull/56907) NIXL 1.5.0
- Mooncake：[#57952](https://github.com/vllm-project/vllm/pull/57952) hybrid/MLA KV 打包成合并后的传输区域；[#49300](https://github.com/vllm-project/vllm/pull/49300) `CUSTOM_MEM_POOL` —— `vllm/distributed/mooncake_store.py`（新）；`VLLM_MOONCAKE_CONNECTOR_TIMEOUT`
- MoRIIO：[#51052](https://github.com/vllm-project/vllm/pull/51052) READ 模式传输 hybrid mamba/KDA 递归状态；[#57700](https://github.com/vllm-project/vllm/pull/57700) K3 DSpark hybrid READ；新增 `moriio/moriio_heartbeat.py`
- KV Offload：[#50045](https://github.com/vllm-project/vllm/pull/50045) back-pressure 检测与处理 —— `vllm/v1/kv_offload/tiering/backpressure.py`（新）；[#55885](https://github.com/vllm-project/vllm/pull/55885) 每请求的 `max_load_tokens`；[#57145](https://github.com/vllm-project/vllm/pull/57145) 跳过 scratch group；[#57652](https://github.com/vllm-project/vllm/pull/57652) 多 group MLA 的 replicated_layout 检测；[#59899](https://github.com/vllm-project/vllm/pull/59899) KVCR adapter 改用 pool-and-index API；`vllm/v1/kv_offload/cpu/host_register.py`（新）
- EC（encoder cache）/ EPD：[#54176](https://github.com/vllm-project/vllm/pull/54176) 动态 EPD 代理；[#56242](https://github.com/vllm-project/vllm/pull/56242) Mooncake P2P cross-encoder 缓存；[#54960](https://github.com/vllm-project/vllm/pull/54960) EC metrics —— `vllm/distributed/ec_transfer/ec_connector/{metrics,encoder_output_width}.py`、`mooncake_store_embedding/*`（新）；[#58086](https://github.com/vllm-project/vllm/pull/58086) `--mm-encoder-only` 时跳过 LM 权重分片；[#57887](https://github.com/vllm-project/vllm/pull/57887) 只含 metadata 的音频输入；[#57095](https://github.com/vllm-project/vllm/pull/57095) 每个 encoder 批量处理图像请求
- P/D：[#54222](https://github.com/vllm-project/vllm/pull/54222) 在 `prompt_tokens_details` 中报告 prefill worker 的 cache 命中；[#58874](https://github.com/vllm-project/vllm/pull/58874) KV-fetch 各阶段 gauge；[#52245](https://github.com/vllm-project/vllm/pull/52245) PushConnector 记录 D 侧的最近活动时间
- HiSparse：[#60647](https://github.com/vllm-project/vllm/pull/60647) 自己管理 GPU 驻留，不再修改 core block-table 行；[#57930](https://github.com/vllm-project/vllm/pull/57930)、[#60083](https://github.com/vllm-project/vllm/pull/60083) 驻留状态缓存；指标见 [#58949](https://github.com/vllm-project/vllm/pull/58949)、[#59485](https://github.com/vllm-project/vllm/pull/59485)、[#60285](https://github.com/vllm-project/vllm/pull/60285)
- [#55844](https://github.com/vllm-project/vllm/pull/55844) KV-event publisher 绑定端口 0，并暴露实际绑定的端点

### 4.5 分布式并行（TP/PP/DP/CP）

- [#60517](https://github.com/vllm-project/vllm/pull/60517) **移除 torch.compile 的 SP / AsyncTP**：删除 `vllm/compilation/passes/fusion/{collective_fusion,sequence_parallelism}.py`；`PassConfig` 去掉 `enable_sp`、`fuse_gemm_comms`、`sp_min_token_num`（`vllm/config/compilation.py`）
- PCP / DCP：[#57075](https://github.com/vllm-project/vllm/pull/57075) PCP + DP；[#52162](https://github.com/vllm-project/vllm/pull/52162) decode 请求在 PCP rank 间分片 —— `vllm/v1/attention/ops/pcp.py`、`vllm/v1/worker/gpu/pcp_manager.py`；[#56538](https://github.com/vllm-project/vllm/pull/56538) 暴露 DCP 的有效 attention block size
- PP：[#58542](https://github.com/vllm-project/vllm/pull/58542) 请求已离开 engine 时跳过 sampled-token 广播；[#52110](https://github.com/vllm-project/vllm/pull/52110) sharded state loader 支持 PP
- [#56015](https://github.com/vllm-project/vllm/pull/56015) XPU 按 `device_ids` 放置 worker；[#53636](https://github.com/vllm-project/vllm/pull/53636) 调度器和 NIXL 的 CPU affinity 按 local rank 对齐

### 4.6 编译 / CUDA Graph

- [#59160](https://github.com/vllm-project/vllm/pull/59160) CUDA graph 池可以在 sleep 时释放 —— `vllm/compilation/cudagraph_pool.py`（新），参数 `--sleep-mode-offload-cudagraph`；[#59523](https://github.com/vllm-project/vllm/pull/59523) ROCm 上同样支持
- [#56922](https://github.com/vllm-project/vllm/pull/56922) MRV2 中仅 encoder 的 ViT 可以 capture CUDA graph；[#58925](https://github.com/vllm-project/vllm/pull/58925)、[#56711](https://github.com/vllm-project/vllm/pull/56711) encoder cudagraph padding 和 d2d 优化
- [#57168](https://github.com/vllm-project/vllm/pull/57168) Pixtral 视觉 encoder 支持 compile；[#58182](https://github.com/vllm-project/vllm/pull/58182) 修复编译后 ViT 的 attention 输出布局
- [#48133](https://github.com/vllm-project/vllm/pull/48133) torch.compile 日志带上组件标签；[#58474](https://github.com/vllm-project/vllm/pull/58474) 遵守 Triton cache 覆盖设置
- [#57745](https://github.com/vllm-project/vllm/pull/57745) capture CUDA graph 时把 `intermediate_tensors` 传给模型

### 4.7 多模态

- 多模态缓存重构：`vllm/multimodal/cache.py` 拆分为 `vllm/multimodal/cache/{base,factories,lru,shm}.py`（[#57674](https://github.com/vllm-project/vllm/pull/57674)、[#57913](https://github.com/vllm-project/vllm/pull/57913)）
- [#53610](https://github.com/vllm-project/vllm/pull/53610)、[#57967](https://github.com/vllm-project/vllm/pull/57967)、[#57576](https://github.com/vllm-project/vllm/pull/57576) processor 清理，dummy options 按模态分类型
- [#56872](https://github.com/vllm-project/vllm/pull/56872) dp-sharded ViT 路径保留原始像素；[#60196](https://github.com/vllm-project/vllm/pull/60196) encoder 输入按模态排序；[#57531](https://github.com/vllm-project/vllm/pull/57531) Mistral3 图像预处理优化
- [#58693](https://github.com/vllm-project/vllm/pull/58693) s390x 用 torchcodec 做视频推理

### 4.8 Pooling / Embedding / Rerank / Scoring

- [#40337](https://github.com/vllm-project/vllm/pull/40337) flash-maxsim late-interaction（见 §2.6）
- [#56505](https://github.com/vllm-project/vllm/pull/56505)、[#57498](https://github.com/vllm-project/vllm/pull/57498) 分块 embedding 的 padding 上限和归一化修复
- [#57666](https://github.com/vllm-project/vllm/pull/57666) reranker 在文档长度限制下保持 tokenization 正确；[#57664](https://github.com/vllm-project/vllm/pull/57664) 原始 logits 保留 BERT 系列的 head
- [#54335](https://github.com/vllm-project/vllm/pull/54335) **固定 token 的 prefill 打分**；[#56984](https://github.com/vllm-project/vllm/pull/56984) 每行候选 ID —— `vllm/sampling_params.py`、`vllm/outputs.py`
- [#60841](https://github.com/vllm-project/vllm/pull/60841) generative scoring 支持 cache salt

### 4.9 RL / Sleep / 权重传输 / 快速启动

- 权重传输：[#51520](https://github.com/vllm-project/vllm/pull/51520) **NCCL M2N**（按分片感知）—— `vllm/distributed/weight_transfer/m2n_{common,engine,layout,source,trainer}.py`（新），新 backend 为 `nccl_m2n`（`vllm/config/weight_transfer.py`）；[#58399](https://github.com/vllm-project/vllm/pull/58399) **ModelExpress** —— `weight_transfer/modelexpress_engine.py`；[#61060](https://github.com/vllm-project/vllm/pull/61060) 不停推理也能释放 communicator
- [#57849](https://github.com/vllm-project/vllm/pull/57849) RL 入口合并：`vllm/entrypoints/serve/dev/rlhf/` 移到 `vllm/entrypoints/rl/{offline,factories}.py`、`vllm/entrypoints/rl/online/{api_router,metrics}.py`
- [#51350](https://github.com/vllm-project/vllm/pull/51350) 权重校验 —— `vllm/model_executor/model_loader/weight_checksum.py`（新）；[#59582](https://github.com/vllm-project/vllm/pull/59582) 最终输出中返回 token 级的 weight-version span
- Sleep
    - [#44890](https://github.com/vllm-project/vllm/pull/44890) `release_kv_cache_memory()` API
    - [#59156](https://github.com/vllm-project/vllm/pull/59156) 释放 WorkspaceManager scratch
    - [#59158](https://github.com/vllm-project/vllm/pull/59158) offload KV 初始化状态
    - [#57891](https://github.com/vllm-project/vllm/pull/57891) level-2 sleep 保留冻结的权重（`--sleep-preserve-parameter-names`）
    - [#61046](https://github.com/vllm-project/vllm/pull/61046) 保留 draft 私有的权重
    - [#52864](https://github.com/vllm-project/vllm/pull/52864) sleep API 响应和指标对齐
- Fast Start（权重缓存 daemon，`vllm/model_executor/model_loader/weight_cache/`）
    - [#56680](https://github.com/vllm-project/vllm/pull/56680) **`vllm preload`** —— `vllm/entrypoints/cli/preload.py`
    - [#57386](https://github.com/vllm-project/vllm/pull/57386) 支持 DP；[#55477](https://github.com/vllm-project/vllm/pull/55477) 支持 PP
    - [#57312](https://github.com/vllm-project/vllm/pull/57312) MTP 草稿模型独立缓存
    - [#58552](https://github.com/vllm-project/vllm/pull/58552) 新增 `/health`；[#58370](https://github.com/vllm-project/vllm/pull/58370) 等待 daemon 就绪
    - [#57298](https://github.com/vllm-project/vllm/pull/57298) daemon 持有的权重计入 `gpu_memory_utilization`
    - [#60085](https://github.com/vllm-project/vllm/pull/60085) 预加载 FlashInfer autotune 表
- [#51360](https://github.com/vllm-project/vllm/pull/51360) **可复用的 TP1 已初始化 engine 快照**：`vllm snapshot` —— `vllm/snapshot/{controller,manifest,runtime,server,types}.py`、`vllm/entrypoints/cli/snapshot.py`
- [#58765](https://github.com/vllm-project/vllm/pull/58765) **`vllm download-kernels`**（下载 FlashInfer 预编译 kernel）—— `vllm/entrypoints/cli/download_kernels.py`
- [#60825](https://github.com/vllm-project/vllm/pull/60825) 从本地 snapshot 加载 HF tokenizer/processor；[#60152](https://github.com/vllm-project/vllm/pull/60152) 跳过 MTP head 不加载的 checkpoint 分片；[#61167](https://github.com/vllm-project/vllm/pull/61167) InstantTensor 应用 EP 权重过滤；[#51792](https://github.com/vllm-project/vllm/pull/51792) 权重加载时校验 dtype 一致

### 4.10 Frontend / OpenAI API / 解析器

- **新端点**：[#59299](https://github.com/vllm-project/vllm/pull/59299) **`/v1/systemone`（structured decisions）**—— `vllm/entrypoints/generate/structured_decisions/*`（新），在 [#60677](https://github.com/vllm-project/vllm/pull/60677) 后无需 opt-in；后续加了 noul/score 题型（[#60696](https://github.com/vllm-project/vllm/pull/60696)）、cache salt 与 tracing（[#60786](https://github.com/vllm-project/vllm/pull/60786)）、DiffusionGemma 读策略（[#60466](https://github.com/vllm-project/vllm/pull/60466)），可服务 LiquidAI/d1-3B（[#60676](https://github.com/vllm-project/vllm/pull/60676)）
- **`/inference/v1/generate`（token-in/token-out）**：[#58588](https://github.com/vllm-project/vllm/pull/58588) `output_mode`；[#53187](https://github.com/vllm-project/vllm/pull/53187) 返回 prompt metadata；[#43310](https://github.com/vllm-project/vllm/pull/43310) 每请求的 spec decode 指标；[#58181](https://github.com/vllm-project/vllm/pull/58181) logprobs 使用整数 token id；[#60251](https://github.com/vllm-project/vllm/pull/60251) 返回 `stop_reason`；[#58488](https://github.com/vllm-project/vllm/pull/58488) 修复完整 logprobs —— `vllm/entrypoints/scale_out/token_in_token_out/*`
- **derender / render**：[#50550](https://github.com/vllm-project/vllm/pull/50550) 流式输出 reasoning 与工具调用；[#57528](https://github.com/vllm-project/vllm/pull/57528) 流式 detokenize offload；[#55029](https://github.com/vllm-project/vllm/pull/55029) 流式 chunk 附带 logprobs；[#60728](https://github.com/vllm-project/vllm/pull/60728) 加了 `return_mm_kwargs`，后在 [#61427](https://github.com/vllm-project/vllm/pull/61427) 改为 `mm_payload` 枚举；[#60073](https://github.com/vllm-project/vllm/pull/60073) 抽取 batch chat derender
- **Responses API**：[#55084](https://github.com/vllm-project/vllm/pull/55084) 每请求指标；[#48084](https://github.com/vllm-project/vllm/pull/48084) `min_p`；[#57322](https://github.com/vllm-project/vllm/pull/57322) 非 Harmony 路径支持 CustomToolCall
- **Anthropic `/v1/messages`**：[#58613](https://github.com/vllm-project/vllm/pull/58613) 支持关闭 thinking
- **解析器框架（`vllm/parser/`）**
    - [#58602](https://github.com/vllm-project/vllm/pull/58602)、[#58603](https://github.com/vllm-project/vllm/pull/58603) 从 Transformers 移植 `chat_parsing` —— `vllm/parser/chat_parsing/*`（新）
    - [#58604](https://github.com/vllm-project/vllm/pull/58604) 从 checkpoint 的 response template 解析工具调用和 reasoning —— `vllm/parser/response_template.py`、`vllm/reasoning/response_template_reasoning_parser.py`、`vllm/tool_parsers/response_template_tool_parser.py`
    - 迁移到流式 parser engine 的解析器：Step-3.5（[#59321](https://github.com/vllm-project/vllm/pull/59321)）、MiniMax-M3（[#59743](https://github.com/vllm-project/vllm/pull/59743)）、plamo3（[#56575](https://github.com/vllm-project/vllm/pull/56575)）、Granite（`granite_engine_tool_parser.py` 替换 `granite_tool_parser.py`）、MiMo（`vllm/parser/mimo.py`）
    - 新增 Granite 4.2 的 `granite_thinking` reasoning parser（[#55957](https://github.com/vllm-project/vllm/pull/55957)）
    - [#59744](https://github.com/vllm-project/vllm/pull/59744) 删除 PyO3 tool-parser bridge —— `vllm/tool_parsers/rust_tool_parser.py`
- **工具调用 / 结构化输出**
    - [#56268](https://github.com/vllm-project/vllm/pull/56268) **`--tool-strict-level`** —— `vllm/tool_parsers/tool_strict_level.py`（新）
    - [#58019](https://github.com/vllm-project/vllm/pull/58019) MiMo-V2.6 strict 工具调用；[#56403](https://github.com/vllm-project/vllm/pull/56403) GLM-4.7 非 strict 工具调用用浅层 structural tag 约束
    - [#58321](https://github.com/vllm-project/vllm/pull/58321) xgrammar 后端原生解析 Lark 语法；[#57594](https://github.com/vllm-project/vllm/pull/57594) 允许 `propertyNames` 与 `additionalProperties` 同时出现
    - [#57272](https://github.com/vllm-project/vllm/pull/57272) 升级 XGrammar（后续固定为 0.2.8）
- **其他**
    - [#44229](https://github.com/vllm-project/vllm/pull/44229) DeepSeek-V4 FIM 补全渲染
    - [#58545](https://github.com/vllm-project/vllm/pull/58545) **移除 slow tokenizer 模式**
    - [#57520](https://github.com/vllm-project/vllm/pull/57520) **移除 `assistant_token_mask`**
    - [#55128](https://github.com/vllm-project/vllm/pull/55128) Harmony 依赖换成 `oss-harmony`
    - [#60723](https://github.com/vllm-project/vllm/pull/60723) 新连接分给负载最低的 API worker
    - [#58163](https://github.com/vllm-project/vllm/pull/58163) `--enable-log-requests` 时记录请求 JSON body
    - [#57045](https://github.com/vllm-project/vllm/pull/57045) `vllm chat` 支持 thinking；[#57357](https://github.com/vllm-project/vllm/pull/57357) `--help` 只显示 config docstring 的摘要行
    - [#59889](https://github.com/vllm-project/vllm/pull/59889) 404 时列出已服务的模型名；[#61042](https://github.com/vllm-project/vllm/pull/61042) 解释被禁用的 scale-out 端点
    - 模块移动：`vllm/entrypoints/chat_utils.py` 改到 `vllm/renderers/chat_utils.py`（[#60522](https://github.com/vllm-project/vllm/pull/60522)）；`vllm/entrypoints/offline_utils.py` 改到 `vllm/entrypoints/common/offline.py`（[#58052](https://github.com/vllm-project/vllm/pull/58052)）

### 4.11 Rust Frontend（`rust/`，67 个提交）

- CLI / 服务参数：`--hf-overrides`（[#56931](https://github.com/vllm-project/vllm/pull/56931)）、`--sse-keep-alive-interval`（[#58306](https://github.com/vllm-project/vllm/pull/58306)）、`--stream-interval` 和每请求的 `stream_interval`（[#60589](https://github.com/vllm-project/vllm/pull/60589)）；[#58330](https://github.com/vllm-project/vllm/pull/58330) 识别不支持或无效的 serve 参数；[#59659](https://github.com/vllm-project/vllm/pull/59659) Python `vllm serve` 暴露 gRPC 端口
- gRPC：sampling mask（[#56777](https://github.com/vllm-project/vllm/pull/56777)）、本地 DP size（[#57116](https://github.com/vllm-project/vllm/pull/57116)）、Shutdown RPC（[#59316](https://github.com/vllm-project/vllm/pull/59316)）、禁止的 token 序列和 cache usage（[#59837](https://github.com/vllm-project/vllm/pull/59837)）、reasoning controls（[#59836](https://github.com/vllm-project/vllm/pull/59836)）、显式的 0 采样值（[#60115](https://github.com/vllm-project/vllm/pull/60115)）、客户端断开时中止请求（[#60922](https://github.com/vllm-project/vllm/pull/60922)）、ping 容忍度（[#55102](https://github.com/vllm-project/vllm/pull/55102)）；`vllm-proto` 0.3.0（[#57233](https://github.com/vllm-project/vllm/pull/57233)）
- 输出语法 / 解析器：parser 自有的 output grammar 接口（[#55269](https://github.com/vllm-project/vllm/pull/55269)、[#57340](https://github.com/vllm-project/vllm/pull/57340)、[#60781](https://github.com/vllm-project/vllm/pull/60781)）、通过 XGrammar 回放（[#59143](https://github.com/vllm-project/vllm/pull/59143)）、`SchemaRoot` 共享（[#59408](https://github.com/vllm-project/vllm/pull/59408)）、MiMo V2.5 和 Kimi K3 parser（[#57933](https://github.com/vllm-project/vllm/pull/57933)、[#58358](https://github.com/vllm-project/vllm/pull/58358)、[#59190](https://github.com/vllm-project/vllm/pull/59190)）、`hf` response-template parser（[#59005](https://github.com/vllm-project/vllm/pull/59005)）、自定义 chat role（[#58311](https://github.com/vllm-project/vllm/pull/58311)）
- OpenAI 兼容：`logprob_token_ids`（[#59549](https://github.com/vllm-project/vllm/pull/59549)）、weight-version span（[#61070](https://github.com/vllm-project/vllm/pull/61070)）
- 指标：抢占直方图（[#57033](https://github.com/vllm-project/vllm/pull/57033)）、无锁直方图（[#58574](https://github.com/vllm-project/vllm/pull/58574)）
- 视觉：通过 spec 构造模型自己的 vision processor（[#58109](https://github.com/vllm-project/vllm/pull/58109)）、Nemotron-H（[#57634](https://github.com/vllm-project/vllm/pull/57634)）

### 4.12 指标 / 日志 / Profiling

- 日志配置化：[#57205](https://github.com/vllm-project/vllm/pull/57205) 控制台日志改成 CLI 配置 —— `vllm/config/logging.py`（新），参数 `--logging-config`、`--log-level`；`--log-config-file` 已弃用，计划在 v0.33.0 移除；[#58739](https://github.com/vllm-project/vllm/pull/58739) 内置 JSON formatter —— `vllm/logging_utils/formatter.py`；`access_log_filter.py` 改名为 `uvicorn_logging.py`
- 指标：
    - [#48867](https://github.com/vllm-project/vllm/pull/48867) `--custom-histogram-buckets` —— `vllm/v1/metrics/buckets.py`
    - [#56318](https://github.com/vllm-project/vllm/pull/56318) 按 cache tier 统计命中的 prompt token —— `vllm/v1/metrics/cache_hit_source.py`（新）
    - [#55812](https://github.com/vllm-project/vllm/pull/55812) 准入拒绝指标
    - [#57251](https://github.com/vllm-project/vllm/pull/57251) SimpleCPUOffload 的 Prometheus 指标 —— `vllm/v1/simple_kv_offload/metrics.py`（新）
    - [#58715](https://github.com/vllm-project/vllm/pull/58715) 按水印状态统计
    - [#52141](https://github.com/vllm-project/vllm/pull/52141) 模型初始化耗时
- Profiling：[#57460](https://github.com/vllm-project/vllm/pull/57460) 统一的平台感知 torch profiling（`torch_profiler_activities`）；[#57875](https://github.com/vllm-project/vllm/pull/57875) 每 session 的 profiling 控制

### 4.13 安全（[Security]）

- [#50300](https://github.com/vllm-project/vllm/pull/50300) chat template 资源耗尽 DoS（GHSA-4hhp-h66f-…），对应 `VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT`
- [#51504](https://github.com/vllm-project/vllm/pull/51504) P2P ZMQ 无限会话导致 socket 耗尽，对应 `VLLM_P2P_{MAX_PEERS,IDLE_TIMEOUT_S,HANDSHAKE_TIMEOUT_S}`
- [#61009](https://github.com/vllm-project/vllm/pull/61009) **默认收紧 CORS origins** —— `vllm/entrypoints/launchers/cli_args.py`、`rust/src/server/src/middleware/cors.rs`
- [#59357](https://github.com/vllm-project/vllm/pull/59357) shm 多模态 cache handle 需要认证；[#57833](https://github.com/vllm-project/vllm/pull/57833) 优先使用新的 MM payload
- [#58830](https://github.com/vllm-project/vllm/pull/58830) 对每请求的 mm processor kwargs 设门控；[#60022](https://github.com/vllm-project/vllm/pull/60022) 不可信媒体只允许部分 Pillow 格式；[#58832](https://github.com/vllm-project/vllm/pull/58832) 加强消息清洗
- [#57731](https://github.com/vllm-project/vllm/pull/57731) 拒绝超过默认 max_tokens 的 min_tokens；[#60614](https://github.com/vllm-project/vllm/pull/60614) 不保留 Phi-4 MM 的音频 attention 矩阵；[#59417](https://github.com/vllm-project/vllm/pull/59417) DeepSeek-OCR 接受全零像素张量
- 依赖安全升级：[#59249](https://github.com/vllm-project/vllm/pull/59249)、[#59315](https://github.com/vllm-project/vllm/pull/59315)、[#59427](https://github.com/vllm-project/vllm/pull/59427)

---

## 5. 硬件平台

### 5.1 ROCm / AMD（207 个提交，最活跃的平台）

- **依赖**：AITER 依次升到 v0.1.22.post1、v0.1.23、v0.1.24.post1（[#56885](https://github.com/vllm-project/vllm/pull/56885)、[#58867](https://github.com/vllm-project/vllm/pull/58867)、[#59794](https://github.com/vllm-project/vllm/pull/59794)）；torch 2.13 + triton 3.8（[#50605](https://github.com/vllm-project/vllm/pull/50605)）；TheRock 镜像使用 Triton 3.8.x ToT（[#58006](https://github.com/vllm-project/vllm/pull/58006)）；MoRI 升级（[#56073](https://github.com/vllm-project/vllm/pull/56073)）
- **GEMM**
    - [#58510](https://github.com/vllm-project/vllm/pull/58510) gfx950 上原生 32x32 block scale 的 MXFP8 GEMM —— `vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py`（新）
    - [#55001](https://github.com/vllm-project/vllm/pull/55001) 重构 tuned gemm
    - [#53283](https://github.com/vllm-project/vllm/pull/53283) 单输出 GEMM 使用 wvSplitK
    - [#58566](https://github.com/vllm-project/vllm/pull/58566) skinny GEMM 每步减少 69 次多余拷贝 —— `vllm/model_executor/layers/utils.py`
    - [#54185](https://github.com/vllm-project/vllm/pull/54185) 融合的 shared-expert gate GEMM 走平台 dispatcher
    - W4A16：[#56301](https://github.com/vllm-project/vllm/pull/56301) gfx11 stride pad；[#52619](https://github.com/vllm-project/vllm/pull/52619) RDNAHybrid 中等 skinny；[#54706](https://github.com/vllm-project/vllm/pull/54706) RDNA3 split-K 精度修复
    - [#57055](https://github.com/vllm-project/vllm/pull/57055) 恢复 `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM`，w4a4 ASM GEMM 默认关闭
- **Attention / MLA**
    - [#59132](https://github.com/vllm-project/vllm/pull/59132) `ROCM_SEGMENTED_ATTN`；[#57057](https://github.com/vllm-project/vllm/pull/57057) `ULTRAQUANT`
    - [#53492](https://github.com/vllm-project/vllm/pull/53492) AITER sparse MLA Gluon（`VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA`）；[#49263](https://github.com/vllm-project/vllm/pull/49263) Sparse MLA 支持 dense MHA
    - [#56861](https://github.com/vllm-project/vllm/pull/56861)、[#59965](https://github.com/vllm-project/vllm/pull/59965) DCP verify 默认用 AITER ASM round-robin decode（`VLLM_ROCM_AITER_MLA_DCP_VERIFY`）
    - [#58381](https://github.com/vllm-project/vllm/pull/58381) MLA metadata 构建的 host dispatch 减少约 21 倍；[#57978](https://github.com/vllm-project/vllm/pull/57978) 并行展开 MLA page-index
    - [#60159](https://github.com/vllm-project/vllm/pull/60159) gfx942/950 使用原生 `merge_attn_states`；[#58225](https://github.com/vllm-project/vllm/pull/58225) RDNA3/4 上收窄 prefill KV tile
    - [#54805](https://github.com/vllm-project/vllm/pull/54805) 从 ROCM_AITER_FA **回滚** AITER PA gluon decode；[#59550](https://github.com/vllm-project/vllm/pull/59550) 修复 ROCM_ATTN 的 sliding-window 边界；[#57750](https://github.com/vllm-project/vllm/pull/57750) RDNA3 上关闭 query 量化
- **融合**：[#48249](https://github.com/vllm-project/vllm/pull/48249) QuickReduce + RMSNorm；[#50212](https://github.com/vllm-project/vllm/pull/50212) QK-norm/RoPE/KV-cache 融合扩展到 MRoPE；[#54857](https://github.com/vllm-project/vllm/pull/54857) MLA 双 RMSNorm + FP8 group quant；[#58099](https://github.com/vllm-project/vllm/pull/58099) 静态 FP8 attention 输出；[#47842](https://github.com/vllm-project/vllm/pull/47842) GDN 输出 norm
- **MoE**：[#55934](https://github.com/vllm-project/vllm/pull/55934) triton_kernels 3.8 的 mxfp4 MoE；[#55368](https://github.com/vllm-project/vllm/pull/55368) AITER MoE 中间维度在分配时 pad；[#55104](https://github.com/vllm-project/vllm/pull/55104) W4A4 中间维度对齐到 128；[#57426](https://github.com/vllm-project/vllm/pull/57426) MXFP8 MoE 受 AITER 开关控制；[#59253](https://github.com/vllm-project/vllm/pull/59253) CT MoE 权重在设备上转置；[#58262](https://github.com/vllm-project/vllm/pull/58262) gfx942 上的 MiMo-V2.6 MXFP4
- **GDN / KDA**：[#57560](https://github.com/vllm-project/vllm/pull/57560) AITER FlyDSL GDN prefill —— `vllm/v1/attention/backends/gdn_attn.py`；[#53623](https://github.com/vllm-project/vllm/pull/53623) GDN decode 快速路径
- **其他**：[#52231](https://github.com/vllm-project/vllm/pull/52231) batch invariance；[#59523](https://github.com/vllm-project/vllm/pull/59523) cuMem CUDA graph 池 offload；[#56343](https://github.com/vllm-project/vllm/pull/56343) 大的 pageable H2D 先 staging；[#57328](https://github.com/vllm-project/vllm/pull/57328) 修复 ROCR host segfault；[#51681](https://github.com/vllm-project/vllm/pull/51681) mori-io 多 decode P/D 的路由竞争
- 新增 env：`VLLM_ROCM_MONO_DECODE`、`VLLM_ROCM_AITER_MEGA_MOE_FUSE_SHARED_EXPERTS`、`VLLM_ROCM_AITER_MLA_DCP_VERIFY`、`VLLM_ROCM_USE_AITER_FP4_ASM_GEMM`、`VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4`、`VLLM_ROCM_USE_AITER_MOE_SITUV2`、`VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA`

### 5.2 Intel XPU（54 个提交）

- [#56013](https://github.com/vllm-project/vllm/pull/56013) 升级到 PyTorch 2.14（triton 3.8.0+xpu）
- [#51600](https://github.com/vllm-project/vllm/pull/51600) **XPU graph 默认开启**（删除 `VLLM_XPU_ENABLE_XPU_GRAPH`）
- SYCL kernel：[#57172](https://github.com/vllm-project/vllm/pull/57172) LayerNorm；[#55721](https://github.com/vllm-project/vllm/pull/55721) `apply_rotary_emb` —— `vllm/_xpu_ops.py`；[#56096](https://github.com/vllm-project/vllm/pull/56096) `fused_qk_rmsnorm_rope_gate` Triton；[#58896](https://github.com/vllm-project/vllm/pull/58896) context-key 归一化使用 rms_norm
- [#58297](https://github.com/vllm-project/vllm/pull/58297) **XPU MLA 合入 FLASH_ATTN_MLA 后端** —— `vllm/v1/attention/backends/mla/flashattn_mla.py`、`fa_utils.py`；[#57297](https://github.com/vllm-project/vllm/pull/57297) FA block size 必须是 64 的倍数
- 量化 / MoE：[#53162](https://github.com/vllm-project/vllm/pull/53162) Triton int8_w8a8 MoE；[#53065](https://github.com/vllm-project/vllm/pull/53065) fused MoE 调优；[#56063](https://github.com/vllm-project/vllm/pull/56063) B70 block-FP8 调优；[#56027](https://github.com/vllm-project/vllm/pull/56027) online fp8_per_channel 修复；[#60976](https://github.com/vllm-project/vllm/pull/60976) auto-round-lib 0.16.0
- [#44987](https://github.com/vllm-project/vllm/pull/44987) EPLB；[#55881](https://github.com/vllm-project/vllm/pull/55881) batch invariance；[#51956](https://github.com/vllm-project/vllm/pull/51956) KV offload mmap 区域注册为 pinned 内存；[#61158](https://github.com/vllm-project/vllm/pull/61158) sleep 模式内存释放；[#59320](https://github.com/vllm-project/vllm/pull/59320) EC producer 使用只含 encoder 的 runner

### 5.3 CPU / POWER / s390x / Arm（48 个提交）

- x86：[#49942](https://github.com/vllm-project/vllm/pull/49942) **FP8 W8A8 linear/MoE**（见 §2.2）；[#58515](https://github.com/vllm-project/vllm/pull/58515) wheel 在 Ubuntu 22.04 上构建，支持 AMX-FP8；[#58133](https://github.com/vllm-project/vllm/pull/58133) 只在编译器支持时启用 AVX10.2 路径；[#58140](https://github.com/vllm-project/vllm/pull/58140) 使用预编译的 triton（triton-cpu 3.8.0）
- [#53913](https://github.com/vllm-project/vllm/pull/53913) 向量化采样 kernel；[#56252](https://github.com/vllm-project/vllm/pull/56252) fp32 attention sinks；[#43048](https://github.com/vllm-project/vllm/pull/43048) CPU 通过 Triton dispatcher 覆盖 kernel
- AMD Zen（zentorch）：[#54508](https://github.com/vllm-project/vllm/pull/54508) encoder attention 使用 SDPA；[#54967](https://github.com/vllm-project/vllm/pull/54967) MLA prefill；[#54024](https://github.com/vllm-project/vllm/pull/54024) DA8W4（W4A8）dense/MoE；[#59434](https://github.com/vllm-project/vllm/pull/59434) INT8 MoE 使用 f32 scale
- Arm：[#56045](https://github.com/vllm-project/vllm/pull/56045) 重构 paged attention —— `csrc/cpu/cpu_attn_neon*.hpp`；[#54093](https://github.com/vllm-project/vllm/pull/54093) aarch64 conv1d —— `csrc/cpu/sgl-kernels/conv_arm.h`（新）
- POWER：[#55316](https://github.com/vllm-project/vllm/pull/55316) W8A8 INT8 MoE —— `csrc/cpu/micro_gemm/cpu_micro_gemm_int8_vsx.hpp`（新）；[#59149](https://github.com/vllm-project/vllm/pull/59149) POWER10 VSX 上的 W4A16（AWQ/GPTQ）；[#58528](https://github.com/vllm-project/vllm/pull/58528) auto dtype 优先 bf16
- s390x：[#58693](https://github.com/vllm-project/vllm/pull/58693) torchcodec 视频；[#54978](https://github.com/vllm-project/vllm/pull/54978) 固定 protobuf 版本
- [#58268](https://github.com/vllm-project/vllm/pull/58268) CPU WNA16 kernel 支持 W4A16 量化 Whisper；[#56547](https://github.com/vllm-project/vllm/pull/56547) **新增 `--device-memory-utilization` 作为 `--gpu-memory-utilization` 的别名**

### 5.4 TPU

- tpu-inference 升级到 v0.30.0、v0.31.0（[#59396](https://github.com/vllm-project/vllm/pull/59396)、[#60190](https://github.com/vllm-project/vllm/pull/60190)）

---

## 6. 配置、CLI 参数、环境变量的变化

### 6.1 新增 / 变化的 CLI 参数（`vllm/engine/arg_utils.py`）

| 参数 | 对应 config 字段（文件） | PR |
| --- | --- | --- |
| `--max-num-active-seqs` | `SchedulerConfig.max_num_active_seqs`（`vllm/config/scheduler.py`） | [#56758](https://github.com/vllm-project/vllm/pull/56758) |
| `--long-prefill-token-threshold-adaptive` | `SchedulerConfig.long_prefill_token_threshold_adaptive` | [#57951](https://github.com/vllm-project/vllm/pull/57951)、[#58459](https://github.com/vllm-project/vllm/pull/58459) |
| `--swa-bounded-replay` | `CacheConfig.swa_bounded_replay=True`（`vllm/config/cache.py`） | [#56227](https://github.com/vllm-project/vllm/pull/56227) |
| `--enable-mamba-shared-prefix-checkpoint`（原 `--enable-mamba-fine-grained-prefix-cache`） | `CacheConfig.enable_mamba_shared_prefix_checkpoint` | [#57382](https://github.com/vllm-project/vllm/pull/57382) |
| `--device-memory-utilization`（`--gpu-memory-utilization` 的别名） | `CacheConfig.gpu_memory_utilization` | [#56547](https://github.com/vllm-project/vllm/pull/56547) |
| `--enable-hc-sp` | `ParallelConfig.enable_hc_sp`（`vllm/config/parallel.py`） | [#56322](https://github.com/vllm-project/vllm/pull/56322) |
| `--enable-shm-tensor-arena` | `ParallelConfig.enable_shm_tensor_arena` | [#51207](https://github.com/vllm-project/vllm/pull/51207) |
| `--sleep-mode-offload-cudagraph` | `ModelConfig.sleep_mode_offload_cudagraph`（`vllm/config/model.py`） | [#59160](https://github.com/vllm-project/vllm/pull/59160) |
| `--sleep-preserve-parameter-names` | `ModelConfig.sleep_preserve_parameter_names` | [#57891](https://github.com/vllm-project/vllm/pull/57891) |
| `--max-lora-cls-labels` | `LoRAConfig.max_lora_cls_labels`（`vllm/config/lora.py`） | [#57766](https://github.com/vllm-project/vllm/pull/57766) |
| `--logging-config`、`--log-level`、`--log-config-file`（已弃用，v0.33.0 移除） | `LoggingConfig`（`vllm/config/logging.py`，新） | [#57205](https://github.com/vllm-project/vllm/pull/57205) |
| `--custom-histogram-buckets` | `ObservabilityConfig.custom_histogram_buckets`（`vllm/config/observability.py`） | [#48867](https://github.com/vllm-project/vllm/pull/48867) |
| `--aux-output-config` | `AuxOutputConfig`（`vllm/config/aux_output.py`，新） | [#45635](https://github.com/vllm-project/vllm/pull/45635) |
| `--disable-cascade-attn` | **no-op，已弃用** | [#61126](https://github.com/vllm-project/vllm/pull/61126) |
| `--tool-strict-level`（serve） | `vllm/entrypoints/launchers/cli_args.py` | [#56268](https://github.com/vllm-project/vllm/pull/56268) |

**新 CLI 子命令**：`vllm preload`（`vllm/entrypoints/cli/preload.py`）、`vllm snapshot`（`cli/snapshot.py`）、`vllm download-kernels`（`cli/download_kernels.py`）。另外删除了 `vllm/scripts.py`（原来已弃用的 `vllm.scripts.main()` 入口）。

### 6.2 其他新增 / 删除的 config 字段

- 新增
    - `AttentionConfig`（`vllm/config/attention.py`）：`tokenspeed_mla_min_split_kv`、`indexer_sparse_logits`、`sparse_mla_prefill_workspace_factor`
    - `CacheConfig`：`effective_attention_block_size`、`hash_block_size`、`cache_hit_alignment_tokens`（均为 init=False）
    - `CompilationConfig`（`vllm/config/compilation.py`）：`cudagraph_decoder_replay=True`、`decoder_replay_cudagraph_capture_sizes`、`decoder_replay_trim_threshold=768`
    - `KernelConfig`（`vllm/config/kernel.py`）：`enable_rocm_segmented_attn_autotune`
    - `PoolerConfig`（`vllm/config/pooler.py`）：`enable_flash_late_interaction=True`
    - EPLB 配置（`vllm/config/parallel.py`）：`enable_migration_batching`
    - Engram（`vllm/config/engram.py`）：`cpu_offload` 默认改为 True；`dp_shared_memory` 默认改为 None（自动）；新增 `use_thp`
    - `ProfilerConfig`：`torch_profiler_activities`、`delay_iterations`、`max_iterations`
    - `WeightTransferConfig.backend` 新增 `nccl_m2n`
- 删除
    - `PassConfig.enable_sp`、`fuse_gemm_comms`、`sp_min_token_num`（[#60517](https://github.com/vllm-project/vllm/pull/60517)）
    - KVTransferConfig 的 `kv_buffer_size`、`kv_rank`、`kv_parallel_size`、`kv_ip`，ECTransferConfig 的 `ec_rank`、`ec_parallel_size`，以及 `sd_worker_cls` 等无用字段（[#59781](https://github.com/vllm-project/vllm/pull/59781)）
    - `mamba_cache_mode="all"`（[#58997](https://github.com/vllm-project/vllm/pull/58997)）
- `SchedulerConfig.max_num_scheduled_tokens` 的约束从 `ge=0` 改为 `ge=1`

### 6.3 环境变量（`vllm/envs.py`）

| 新增 | 说明 / 来源 |
| --- | --- |
| `VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT` | chat template 渲染超时（安全修复 [#50300](https://github.com/vllm-project/vllm/pull/50300)） |
| `VLLM_ENABLE_GEMM_RS` | GEMM + reduce-scatter 融合（[#57428](https://github.com/vllm-project/vllm/pull/57428)，替代 `VLLM_KIMI_K3_GEMM_RS`） |
| `VLLM_FLASHINFER_MOE_A2A_LOW_PRECISION_COMBINE` | all2all 中的 fp8 combine（[#57995](https://github.com/vllm-project/vllm/pull/57995)） |
| `VLLM_MOONCAKE_CONNECTOR_TIMEOUT` | Mooncake 连接器超时 |
| `VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD` | 加载时对 MXFP4 反量化（[#50814](https://github.com/vllm-project/vllm/pull/50814)） |
| `VLLM_P2P_HANDSHAKE_TIMEOUT_S` / `VLLM_P2P_IDLE_TIMEOUT_S` / `VLLM_P2P_MAX_PEERS` | P2P ZMQ 防耗尽（[#51504](https://github.com/vllm-project/vllm/pull/51504)） |
| `VLLM_SKIP_VERSION_SUFFIX` | 构建版本号（[#41074](https://github.com/vllm-project/vllm/pull/41074)） |
| `VLLM_TRITON_JIT_WARMUP_NUM_THREADS` | 并行 Triton warmup（[#58582](https://github.com/vllm-project/vllm/pull/58582)） |
| `VLLM_XPU_PP_MICROBATCH` | XPU MRV2+PP microbatch（[#55145](https://github.com/vllm-project/vllm/pull/55145)） |
| `VLLM_ROCM_*`（7 个） | 见 §5.1 |

| 删除 | 说明 |
| --- | --- |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | 删除无用的 env（[#59781](https://github.com/vllm-project/vllm/pull/59781)） |
| `VLLM_KIMI_K3_GEMM_RS` | 改名为 `VLLM_ENABLE_GEMM_RS` |
| `VLLM_PLE_CPU_OFFLOAD` | 已被 Engram config 取代（[#57937](https://github.com/vllm-project/vllm/pull/57937)） |
| `VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4` | 合并进 `VLLM_ROCM_USE_AITER_MOE_SITUV2`（[#58201](https://github.com/vllm-project/vllm/pull/58201)） |
| `VLLM_XPU_ENABLE_XPU_GRAPH` | XPU graph 改为默认开启（[#51600](https://github.com/vllm-project/vllm/pull/51600)） |

---

## 7. 破坏性变更 / 移除 / 默认值变化清单（升级时重点检查）

| 类别 | 变化 | PR |
| --- | --- | --- |
| Python | **不再支持 Python 3.10**，要求 `>=3.11,<3.15`（`pyproject.toml`、`setup.py`、`CMakeLists.txt`） | [#60402](https://github.com/vllm-project/vllm/pull/60402) |
| 依赖 | Transformers `>=5.16.1,<5.20.0`，删除 <5.16.1 的兼容代码 | [#59614](https://github.com/vllm-project/vllm/pull/59614)、[#59762](https://github.com/vllm-project/vllm/pull/59762) |
| Attention | 移除 cascade attention | [#61126](https://github.com/vllm-project/vllm/pull/61126) |
| 编译 | 移除 torch.compile 的 SP / AsyncTP | [#60517](https://github.com/vllm-project/vllm/pull/60517) |
| 量化 | 移除 AllSpark；移除 fp8.py 和 Quark 中的 online 量化 | [#58001](https://github.com/vllm-project/vllm/pull/58001)、[#53585](https://github.com/vllm-project/vllm/pull/53585)、[#51800](https://github.com/vllm-project/vllm/pull/51800) |
| Tokenizer | 移除 slow tokenizer 模式 | [#58545](https://github.com/vllm-project/vllm/pull/58545) |
| API | 移除 `assistant_token_mask`；移除计划中的弃用项（`wait_for_inflight_requests`、`IOProcessor.output_to_response`、benchmark 的 `--dataset`、`vllm.scripts.main()`） | [#57520](https://github.com/vllm-project/vllm/pull/57520)、[#60923](https://github.com/vllm-project/vllm/pull/60923) |
| LoRA | 移除 tensorizer 加载 LoRA | [#60024](https://github.com/vllm-project/vllm/pull/60024) |
| Mamba | 移除 `mamba_cache_mode="all"`；参数改名（见 §6.1） | [#58997](https://github.com/vllm-project/vllm/pull/58997)、[#57382](https://github.com/vllm-project/vllm/pull/57382) |
| Benchmark | 弃用 sonnet 数据集 | [#60129](https://github.com/vllm-project/vllm/pull/60129) |
| 模型 | 8 个架构改走 Transformers 后端；删除 20 个 vendored config | §3.6 |
| 默认值 | GLM 默认 fp8 KV；XPU graph 默认开启；DeepEPv2 默认 hybrid；SM90 上 Humming 优先于 Marlin；`swa_bounded_replay`、`cudagraph_decoder_replay`、`enable_flash_late_interaction` 默认为 True；batch invariant 下默认用 breakable CG | 见各节 |
| 安全 | 默认 CORS origins 收紧 | [#61009](https://github.com/vllm-project/vllm/pull/61009) |
| 源码路径 | `csrc/libtorch_stable/**` 合并到 `csrc/**`；GLM5Next、Qwen4Exp 目录重构；`chat_utils.py`、`offline_utils.py` 移动；RL 入口移动；spec_decode `autoregressive/` 改名 | §2.1、§3、§4 |

---

## 8. 依赖版本变化

| 依赖 | v0.30.0 | v0.32.0 | 文件 |
| --- | --- | --- | --- |
| Python | `>=3.10,<3.15` | `>=3.11,<3.15` | `pyproject.toml` |
| torch（CUDA） | 2.13.0 | 2.14.0 | `requirements/cuda.txt` |
| torch / triton（XPU） | triton 3.7.2+xpu | torch 2.14、triton 3.8.0+xpu | `requirements/xpu.txt` |
| torch / triton（ROCm） | — | torch 2.13、triton 3.8 | `requirements/rocm*.txt`、`docker/` |
| transformers | `>=5.10.4` | `>=5.16.1,<5.20.0` | `requirements/common.txt` |
| tokenizers | `>=0.21.1` | `>=0.23.2` | `requirements/common.txt` |
| xgrammar | `>=0.2.1,<1.0.0` | `==0.2.8` | `requirements/common.txt` |
| openai-harmony | `>=0.0.3` | 换成 `oss-harmony>=0.0.11`，并排除 openai-harmony | `requirements/common.txt`、`pyproject.toml` |
| flashinfer-python | 0.6.18.post1（另有 flashinfer-cubin） | 0.7.0.post1（不再依赖 cubin，用 `vllm download-kernels`） | `requirements/cuda.txt` |
| nvidia-cutlass-dsl | 4.7.1 | 4.8.0 | `requirements/cuda.txt` |
| humming-kernels | 0.1.12 | 0.1.17 | `requirements/cuda.txt` |
| triton-cpu（x86 CPU） | — | 3.8.0（预编译 wheel） | `requirements/cpu.txt` |
| AITER（ROCm） | — | v0.1.24.post1 | ROCm Dockerfile |
| NIXL | — | 1.5.0 | [#56907](https://github.com/vllm-project/vllm/pull/56907) |
| tpu-inference | — | v0.31.0 | [#60190](https://github.com/vllm-project/vllm/pull/60190) |

---

## 9. 常用的进一步查询命令（给 AI）

```bash
# 某目录在两个版本间的提交
git log --oneline v0.30.0..v0.32.0 -- csrc/
git log --oneline v0.30.0..v0.32.0 -- vllm/model_executor/layers/fused_moe/

# 新增 / 删除 / 重命名的文件
git diff --name-status -M v0.30.0 v0.32.0 -- vllm/ | grep -v '^M'

# 某个函数 / 符号在哪个提交引入或删除
git log --oneline v0.30.0..v0.32.0 -S'mnnvl_multimem_reduce_scatter'

# 某个文件在两个版本间的 diff（注意 csrc 路径重组）
git diff v0.30.0 v0.32.0 -- vllm/v1/attention/backends/flashinfer.py
git diff v0.30.0:csrc/libtorch_stable/topk.cu v0.32.0:csrc/topk.cu

# 环境变量 / CLI 参数对比
diff <(git show v0.30.0:vllm/envs.py | grep -oE '"VLLM_[A-Z0-9_]+"' | sort -u) \
     <(git show v0.32.0:vllm/envs.py | grep -oE '"VLLM_[A-Z0-9_]+"' | sort -u)
```

> 说明：本文档基于 commit 标题、改动文件列表和关键 diff 整理，覆盖了主要的功能和算子变化，但不是逐行审阅。
> 未展开的 Bugfix（477 个）、CI（约 230 个）以及各平台的细碎修复，请查全量索引 [`vllm_v0.30.0_to_v0.32.0_commit_index.md`](./vllm_v0.30.0_to_v0.32.0_commit_index.md)。
