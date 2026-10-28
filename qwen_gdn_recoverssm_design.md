# Qwen GDN RecoverSSM 移植设计（上游优先版，以 Dspark 为例）

- 状态：设计草案 v2.1，替代此前的五缓存自定义方案
- 目标平台：vLLM Ascend，Model Runner V2，A3/A5
- 主要适配场景：Dspark；架构上兼容满足统一 target-verify/commit 契约的 speculative decoding 方法
- 上游基础：vLLM PR #58863、#61309
- Ascend 参考：vllm-ascend PR #19133
- 核心原则：**上游能直接复用的代码不复制；Ascend 只保留硬件适配、图执行和必要的性能优化**

## 1. 设计背景

Qwen GDN 在 speculative decoding 中，原生路径为每个请求保存 `num_spec + 1` 个 recurrent-state checkpoint，采样后按接受长度选择状态并回滚 convolution history。该方式的 recurrent-state 显存随 draft 数线性增长。本文以 Dspark 为主要示例，但 RecoverSSM 位于 target 模型的状态管理层，不依赖 MTP drafter；因此 Qwen3.6 没有 MTP 并不构成架构障碍。

RecoverSSM 改为：

1. 每个请求只保留一个 recurrent checkpoint。
2. verify 从只读 checkpoint 计算整个窗口，并记录每个 token 的 replay record。
3. 采样后仅重放接受前缀，一次性写回 recurrent state。
4. 同时压缩 convolution history。
5. `num_speculative_blocks=0`，不再为每个 draft 保留额外 Mamba page。

上游 #58863 已完成通用 RecoverSSM 框架和 Qwen 风格 GDN 算子，#61309 进一步接入 Qwen3-Next/Qwen3.5。因而 Ascend 的正确策略不是重新设计一套 GDN RecoverSSM，而是**继承上游协议和公共代码，仅适配 Ascend 后端**。Dspark 的提案生成、调度和接受采样逻辑保持原样，RecoverSSM 只替换 target verify 的状态保存与采样后恢复方式。

## 2. 上游 PR 的定位

### 2.1 vLLM PR #58863：通用框架与 GDN 实现

主要提供：

- `CacheConfig.use_recoverssm`：将 KDA 专用开关泛化为 RecoverSSM 开关。
- `recoverssm_utils.py`：统一激活判断、参数校验和 pointer table。
- `recoverssm_metadata.py`：通用 spec-row 分类和 request mapping。
- `mamba/ops/recoverssm.py`：模型无关的 commit-plan 和 convolution compaction。
- `gdn/recoverssm_gdn.py`：Qwen 风格 GDN verify、forward commit 和多层 commit context。
- `gdn_recoverssm.py`：GDN Recover metadata、builder 和 attention backend。
- GDN、PLE、align、graph padding、极端 gate 和 pointer 等测试与修复。

### 2.2 vLLM PR #61309：Qwen3-Next/Qwen3.5 接入

主要提供：

- Qwen3-Next/Qwen3.5 target 和 MTP drafter 的 `SupportsReplaySSM` 声明。
- Qwen GDN architecture allowlist 和配置校验。
- 模型级与 layer 级一致的 replay shape/dtype helper。
- Qwen GDN layer 的 Recover verify 分支。
- strided Q/K/V view 和 in-place output 优化。
- FULL graph padding、profiling teardown 和后续 bugfix。

### 2.3 vllm-ascend PR #19133：只作为 Ascend 经验参考

保留参考：

- NPU Graph staging、图填充和 oversized eager fallback。
- 多层 base-address/stride context。
- A3/A5 tile、UB 和 stride 特化经验。
- Mooncake MambaSpec 传输适配。
- 真实权重、TPOT、GSM8K 和 JIT/recompilation 验证方法。

不再以其 KDA 专用 metadata、记录布局或 commit 数学作为 Qwen GDN 的设计基线。

## 3. 上游复用策略

| 项目 | 复用上游的策略 |
|---|---|
| cache | 复用上游 conv、state、单个 FP32 replay 三类 cache |
| replay layout | 复用上游 `[HV,Q,V+K+1]` 布局，按 value head 保存 |
| verify 输入 | 与上游一致，输入 raw a/b 和 q/k，在 kernel 内 gating/L2 norm |
| commit | 复用上游 decay record 和前向重放 `h=fma(c,k,h*decay)` |
| 数值目标 | 优先与上游逐 token 实现位级一致；AscendC 对照另列 |
| metadata | 复用上游 Recover builder 流程，仅适配 Ascend 卷积描述符 |
| 算子落地 | 先直接运行上游 Triton；确认不兼容后再实现同名后端 |

上游单个 replay cache 会在 GQA 下重复保存 normalized key，但它是上游兼容性、FP32 位级重放和连续访问布局的一部分。第一版不做自定义拆分优化。

## 4. 目标与范围

### 4.1 第一阶段目标

- 采用 Qwen GDN 的目标模型和 `AscendGatedDeltaNetAttention`，不把能力绑定到特定 MTP 模型代际。
- Model Runner V2 + speculative decoding，以 Dspark 为主要适配和验收场景。
- eager 和 FULL_DECODE_ONLY NPU Graph。
- 与上游相同的三类 cache 和 replay record。
- 与上游相同的 verify/commit 公共 API。
- 优先直接复用上游 GDN Triton kernels。
- 默认关闭，原生 Ascend GDN 路径保持不变。

### 4.2 第一阶段的 Ascend 适配边界

在上游通用限制之外，Ascend 首版建议：

- `mamba_cache_mode` 和 prefix caching 复用上游能力，不要求固定为 `none`，也不默认关闭 prefix caching。
- PP=1、PCP=1、DCP=1。
- 非 interleaved GQA，`Nv % Nk == 0`，TP 后仍满足该关系。
- 首版重点验证 `Dk=Dv=128`、BF16 activation、BF16/FP32 state。
- 关闭 stochastic rounding。
- Mooncake PD 作为独立后续工作包。

RecoverSSM 不修改 prefix cache 的查找、命中、block 分配和复用逻辑。它只改变 speculative verify 后的状态提交方式，因此真正需要适配的是 align 模式下接受窗口跨 Mamba block 时的状态落页：final state 写入最后接受 token 所在页，跨边界时还要写 boundary state，并同步 postprocess 的 running state column。上述逻辑直接复用上游实现；Ascend 只需验证其 Triton/图执行和 Dspark 边界组合，不需要另写一套 prefix cache 算法。

### 4.3 “Dspark 适配层＋方法无关 Recover 层”

RecoverSSM 只处理 target 模型的 recurrent/convolution state，不参与 draft 生成和接受采样。为复用 #19133 的接口分层经验，同时避免继承其 K3 专用 builder，本设计明确拆成两层。

#### A. Dspark 适配层：保留现有 Ascend GDN builder 的方法语义

该层由现有 `AscendGDNAttentionMetadataBuilder`、runner 和 sampler 负责：

- 保留 Dspark/Dflash 的 `reorder_batch_threshold` 和 target/draft query 数差异。
- 保留 anchor、draft 生成、调度和接受采样逻辑。
- 生成 Ascend 所需的 causal-convolution descriptors、actual sequence lengths 和 output buffers。
- 保留 `num_actual_reqs` 图填充截断，以及 request/token/state 三类容量 guard。
- 将 Dspark 特有的 batch 和 draft-count 表达规范化成统一 target metadata。

该层可以存在 `method in ("dflash", "dspark")` 等已有分支，因为这些分支描述的是方法自身的调度语义，不属于 RecoverSSM。

#### B. 方法无关 Recover 层：只消费统一 target 契约

该层由上游 Recover helpers、metadata、verify/commit 和 coordinator 组成：

- 用 `recoverssm_spec_rows` 和 `recoverssm_request_indices` 生成 spec rows 与 commit mapping。
- 将 state indices 截成单 checkpoint，并设置 accepted placeholder。
- 根据 query offsets 和实际 accepted count 执行 verify、plan、conv compaction 和 forward commit。
- 不读取 speculative method，不包含 MTP/Dspark/Dflash 专用 kernel 或 cache 分支。

两层之间的兼容契约为：

1. target verify 通过 `CommonAttentionMetadata` 和 `num_decode_draft_tokens_cpu` 描述投机窗口，支持固定或 ragged query length。
2. scheduler/block table 为每个 speculative row 提供唯一 checkpoint index。
3. target sampling 后通过现有 postprocess 接口给出实际接受长度，语义与 `RecoverSSMState.commit_step()` 一致。
4. metadata builder 提供 CPU `is_prefilling`、query offsets 和必要的 request mapping。
5. 接受结果始终表示连续前缀，rejected suffix 不参与 commit。

Dspark 示例：

```text
Dspark 生成 draft、组织 target batch（适配层，保持不变）
    ↓ 统一 target metadata
Recover metadata 规范化 spec rows 和单 checkpoint view（Recover 层）
    ↓
Qwen GDN target 从单 checkpoint verify 并写 replay record
    ↓
Dspark 采样得到接受前缀（适配层，保持不变）
    ↓ 统一 actual accepted count
RecoverSSM commit recurrent state 并压缩卷积历史（Recover 层）
```

#19133 在这里只提供两项借鉴：方法无关的 target 接口，以及图填充/容量保护方式；不引入其 K3 专用请求分区。MTP、Dflash 等其他方法也通过各自的适配层满足同一契约，而不是各自实现状态恢复算法。

## 5. 总体架构

```text
上游配置与能力声明
  use_replayssm → use_recoverssm + SupportsReplaySSM
                         ↓
上游模型级 / layer 级 cache helper
  conv + checkpoint + FP32 replay
                         ↓
┌──────────────────────────────────────────────────────────┐
│ A. Dspark / Speculative Method Adapter                   │
│ 现有 Ascend GDN builder + runner + sampler               │
│ reorder / anchor / conv descriptors / num_actual_reqs    │
└──────────────────────────┬───────────────────────────────┘
                           ↓ 统一 target metadata
┌──────────────────────────────────────────────────────────┐
│ B. Method-independent RecoverSSM                         │
│ spec rows / 单 checkpoint / request mapping / commit meta │
└──────────────────────────┬───────────────────────────────┘
                           ↓
AscendGatedDeltaNetAttention speculative 分支
  causal conv → 上游 gdn_recoverssm_verify
                         ↓
          Target Sampling（以 Dspark 为例）
                         ↓
上游 RecoverSSMState + GDNRecoverSSMCommitContext
  prepare plan → compact conv → forward replay commit
                         ↓
       必要时替换为同名 Ascend kernel/context，不改变上游协议
```

## 6. 上游复用与 Ascend 适配清单

| 能力 | 上游模块 | 复用策略 | Ascend 工作 |
|---|---|---|---|
| 开关与激活判断 | `use_recoverssm`、`uses_recoverssm` | 直接复用 | 只增加 Ascend 配置约束 |
| 模型能力声明 | `SupportsReplaySSM` | 直接复用 | 确保目标 GDN 模型及实际 speculative 配置通过校验，不绑定 MTP drafter |
| Dspark 方法适配 | 现有 Ascend GDN builder，非 Recover 模块 | 保留现有实现 | 保留 reorder、anchor、target query 和采样语义 |
| cache shape/dtype | `gdn_mamba_state_*_from_config`、layer helper | 直接复用 | 保证 Ascend layer 返回相同三类 cache |
| spec-row 分类 | `recoverssm_spec_rows` | 直接复用 | 不重复实现 draftless decode 规则 |
| request mapping | `recoverssm_request_indices` | 直接复用 | 验证混合 batch 顺序 |
| Recover metadata | 上游 Recover helpers、metadata types、builder 流程 | 流程与类型复用 | 用轻量 mixin＋两条 Ascend builder 路径衔接，保留 conv descriptors |
| 图填充与容量保护 | 上游 base builder＋#19133 经验 | 接口复用 | 保留 `num_actual_reqs` 和 request/token/state capacity guards |
| postprocess | `RecoverSSMState` | 直接复用 | Ascend runner 按同一时机调用并支持 reset |
| commit plan | `prepare_commit_plan_kernel` | 先直接运行 | 仅在不兼容时做 Ascend Triton 适配 |
| conv compaction | `compact_conv_state_kernel` | 先直接运行 | 按 A3 UB/stride 调整 tile，不改语义 |
| GDN verify | `gdn_recoverssm_verify` | 先直接运行 | 保留 API；必要时替换 kernel 实现 |
| GDN commit | `GDNRecoverSSMCommitContext` | 先直接运行 | 必要时子类化并替换 commit kernel/tile |
| layer dispatch | Qwen GDN Recover 分支 | 复用控制流 | 接入 Ascend causal conv/output buffer |
| Mooncake | 上游无对应完整实现 | 参考 Ascend #19133 | 按三类 cache 更新传输测试 |

## 7. Cache 与 Replay Record

令：

- `Q = num_spec + 1`；
- `H = Nk_local`，`HV = Nv_local`；
- `K = Dk`，`V = Dv`；
- `B = Mamba block 数`。

### 7.1 三类 cache

| cache | shape | dtype | 语义 |
|---|---|---|---|
| `conv_state` | `[B, conv_dim_local, conv_kernel-1+num_spec]`，按现有 DS/SD 布局 | 现有 conv dtype | checkpoint 历史和扩展卷积窗口 |
| `checkpoint_state` | `[B, HV, V, K]` | BF16 或 FP32 | 唯一 recurrent checkpoint |
| `replay_cache` | `[B, HV, Q, V+K+1]` | FP32 | 每个 token 的 correction、normalized key、decay |

每个 replay record 的布局为：

```text
[0 : V]         correction c_t，已包含 beta
[V : V+K]       normalized key k_t
[V+K]           decay = exp(g_t)，即 verify 实际使用的 FP32 浮点值
```

### 7.2 为什么记录 decay 而不是 gate

上游最新实现记录 verify 实际参与运算的 `decay`，commit 不再重新计算 `exp(g)`。这样可以保证 verify 和 commit 使用完全相同的浮点输入，避免重复计算指数带来的差异。

### 7.3 speculative page 与显存

Recover 模式：

```text
block_table row = [checkpoint_0]
num_speculative_blocks = 0
replay_cache 与 checkpoint_0 使用同一个 block id
```

原生与 Recover 的 recurrent/replay 显存近似为：

```text
Native  = Q × HV × V × K × sizeof(state_dtype)
Recover = HV × V × K × sizeof(state_dtype)
        + Q × HV × (V + K + 1) × sizeof(fp32)
```

不包含 convolution 和其他 KV。第一版必须沿用上游 replay layout 核算收益，不能套用其他记录布局的理论值。

## 8. 上游 GDN Verify 算子

### 8.1 公共 API 保持不变

```python
gdn_recoverssm_verify(
    A_log,
    a,
    b,
    dt_bias,
    q,                      # [1,T,H,K]
    k,                      # [1,T,H,K]
    v,                      # [1,T,HV,V]
    checkpoint_state=...,
    replay_cache=...,
    query_start_loc=...,
    state_indices=...,      # 每个 speculative row 一个 checkpoint
    spec_query_len=Q,
    use_qk_l2norm_in_kernel=True,
    out=...,
)
```

Ascend 不另创 processed-gate API，避免上游模型接入代码出现两套语义。

### 8.2 数学语义

value head `hv` 对应的 key head：

```text
h = hv // (HV // H)
```

每个 token：

```text
g_t = -exp(A_log) × softplus(a_t + dt_bias)
decay_t = exp(g_t)
beta_t = sigmoid(b_t)
S̄_t = S_(t-1) × decay_t
c_t = beta_t × (v_t - S̄_t k_t)
S_t = S̄_t + c_t k_t^T
o_t = scale × S_t q_t
```

verify 将 `c_t`、normalized `k_t` 和 `decay_t` 写入 replay cache，但不写 persistent checkpoint。

### 8.3 并行方式

上游 grid：

```text
grid = (cdiv(V,BV), batch, HV)
BV 默认最高 32
```

每个 program 处理一个 value-head state tile，顺序遍历静态 `SPEC_QUERY_LEN`，使用 `valid = t < query_len` 支持 ragged window。图填充行使用 null state 和零长度窗口，不写有效状态。

### 8.4 Ascend 接入规则

1. 先原样编译运行上游 kernel。
2. 保留 raw `a/b` 和 `A_log/dt_bias` 输入，不在 Ascend 外层再次执行 gating。
3. Recover speculative 分支不再调用原生 recurrent op。
4. 非 speculative prefill/decode 仍使用现有 `fused_gdn_gating`、L2 norm 和原生 recurrent op。
5. strided Q/K/V views 和 in-place output 复用上游优化；不满足 view 条件时回退到 contiguous rearrange。
6. 只有出现明确的 Ascend Triton 不兼容、UB 溢出或性能问题时，才实现同名 Ascend verify。

## 9. 上游 GDN Commit 算子

### 9.1 前向重放

commit 从 checkpoint `s0` 开始，仅遍历接受前缀：

```text
h = s0
for t in range(commit_len):
    h = fma(c_t, k_t, h × decay_t)
store(h, destination_state)
```

这不是上一版的反向闭式重建。其目的包括：

- 与 verify 使用相同的更新顺序。
- 使用 verify 记录的同一组 FP32 correction/key/decay。
- 在上游测试中与逐 token recurrent state 做 `torch.equal` 验证。
- BF16 state 按上游语义在最终写回时 rounding once。

AscendC 与上游 Triton 的数学实现可能不同，因此验收要区分“上游 Triton 对照”和“AscendC 原生对照”，不预先承诺跨后端 bitwise 一致。

### 9.2 多层 commit context

直接复用上游 `GDNRecoverSSMCommitContext`：

- 每层绑定 conv/checkpoint/replay 三个 cache view。
- 校验各层 shape、dtype、block count 和 inner strides。
- 用 `recoverssm_ptr_table` 保存指针的 uint64 bit pattern。
- 保存每层 block strides 和卷积 dim/token strides。
- 预分配 commit length、final/boundary index 和 boundary length。
- 一次 postprocess 完成 plan、conv compaction 和所有层 state commit。

只有需要 Ascend 专用 commit kernel 时才子类化 context；metadata、cache layout 和 `commit()` 参数保持上游一致。

### 9.3 align 模式必须保留的上游修复

align/prefix caching 直接复用上游状态管理链路。为确保 Ascend 与上游行为一致，以下修复不得在后端适配中丢失：

1. final page 使用最后一个提交 token 所在列：

   ```text
   final_state_col = max(num_computed + commit_len - 1, 0) // block_size
   ```

2. final-state commit 与 boundary-state commit 分两次 launch，避免一个 program 写源页、另一个 program 同时读源页产生竞争。
3. boundary page 与 final page 相同时跳过第二次写入。
4. 每个 program 只保留一个 FP32 state accumulator tile，避免 final/boundary 双 tile 降低 occupancy。

## 10. 通用 Commit Plan 与 Convolution Compaction

### 10.1 Commit Plan

直接复用上游 `prepare_commit_plan_kernel`：

- `commit_len = clamp(actual_accepted, 0, query_len, Q)`；
- non-align 下 `final_state_idx = source_state_idx`；
- 生成 final/boundary plan；
- 支持非连续 request mapping；
- invalid source 或零接受长度时不写。

第一版不为 Ascend 复制一套简化 plan，以免重新引入 exact-boundary off-by-one。

### 10.2 Convolution Compaction

直接复用上游 `compact_conv_state_kernel`：

```text
history_len = conv_state_len - Q + 1
final_history = source[commit_len-1 : commit_len-1+history_len]
```

上游默认 `BLOCK_D=256`，与 Ascend #19133 后续 A3 修复方向一致。若实际 A3/A5 仍需调优，只修改 launch tile 或增加后端参数，不改变：

- record 与 checkpoint 的共享 block id；
- accepted-suffix 语义；
- align 下 final/boundary 双目标语义；
- 未触及区域保持不变的约束。

## 11. Metadata：Dspark 适配层与 Recover 层的衔接

### 11.1 不能直接整份替换的原因

上游 `GDNRecoverSSMMetadataBuilder` 的 base builder 是上游 `GDNAttentionMetadataBuilder`。Ascend layer 还依赖现有 builder 附加的：

- `spec_decode_metadata.spec_causal_conv1d`；
- non-spec prefill/decode convolution descriptors；
- `actual_seq_lengths`；
- host-only buffer 和 convolution indices；
- `num_actual_reqs` 图填充参数；
- Dspark/Dflash 的 target reorder 和 draft-count 语义。

因此直接用上游 builder 或 #19133 的 K3 builder 替换 Ascend builder，会同时丢失 Ascend 硬件描述符和 Dspark 方法语义。正确做法是让两层串联，而不是相互替代。

### 11.2 具体类、继承关系与方法签名

具体落地采用“轻量 Recover mixin＋两条独立 Ascend builder 路径”。mixin 只实现 Recover 协议，不负责 base metadata；`super().build()` 在两条路径中分别调用现有 Ascend builder。这样不会让两个完整 builder 相互继承，也不复制上游 `GDNRecoverSSMMetadataBuilder`。

#### 11.2.1 类与继承关系

```text
上游 helpers / types（直接复用）
  recoverssm_spec_rows
  recoverssm_request_indices
  GDNRecoverSSMCommitMetadata
  GDNRecoverSSMMetadata
  GDNRecoverSSMCommitContext
  GDNRecoverSSMAttentionBackend
                    │
                    ▼
AscendGDNRecoverSSMMetadataMixin
  Recover 输入规范化、commit metadata、context、动态字段保留
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
AscendGDNRecoverSSMMetadataBuilder
  mixin + AscendGDNAttentionMetadataBuilder
        │                        AscendGDNHostRecoverSSMMetadataBuilder
        │                          mixin + AscendGDNHostMetadataBuilder
        └───────────┬────────────┘
                    ▼
AscendGDNRecoverSSMAttentionBackend
  按现有 host-prefill 条件选择其中一个 builder
```

对应声明：

```python
class AscendGDNRecoverSSMMetadataMixin:
    """只实现 RecoverSSM 协议；base build 由后继 Ascend builder 完成。"""


class AscendGDNRecoverSSMMetadataBuilder(
    AscendGDNRecoverSSMMetadataMixin,
    AscendGDNAttentionMetadataBuilder,
):
    """全量 Ascend chunk/device metadata 路径。"""


class AscendGDNHostRecoverSSMMetadataBuilder(
    AscendGDNRecoverSSMMetadataMixin,
    AscendGDNHostMetadataBuilder,
):
    """Ascend host-only prefill/conv-buffer 路径。"""


class AscendGDNRecoverSSMAttentionBackend(GDNRecoverSSMAttentionBackend):
    """复用上游 Recover metadata 类型，只选择 Ascend builder。"""
```

这两条继承链的 MRO 都是线性的：mixin 的 `super().build()` 只会进入其后继的 Ascend base builder，不会进入另一个具体 Recover builder。

#### 11.2.2 Mixin 初始化与 context

```python
class AscendGDNRecoverSSMMetadataMixin:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        capacity = max(
            self.vllm_config.scheduler_config.max_num_seqs,
            self.decode_cudagraph_max_bs,
        )
        device = self.spec_state_indices_tensor.device
        self._recoverssm_ones = torch.ones(
            capacity, dtype=torch.int32, device=device
        )
        self._recoverssm_context: GDNRecoverSSMCommitContext | None = None
        self._recoverssm_capturing = False

    def _get_recoverssm_context(self) -> GDNRecoverSSMCommitContext:
        if self._recoverssm_context is None:
            fc = self.vllm_config.compilation_config.static_forward_context
            layers = [fc[name] for name in self.layer_names]
            self._recoverssm_context = GDNRecoverSSMCommitContext.create(
                layers,
                spec_query_len=self.num_spec + 1,
                max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
            )
        return self._recoverssm_context
```

默认直接创建上游 context。只有阶段 0 证明上游 kernel 不能运行时，才将该返回类型替换为 Ascend context 子类；metadata 和 cache 协议不变。

#### 11.2.3 输入规范化

```python
def _prepare_recoverssm_inputs(
    self,
    m: CommonAttentionMetadata,
    draft_counts: torch.Tensor | None,
    num_actual_reqs: int | None,
) -> tuple[
    CommonAttentionMetadata,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """返回 (m, normalized_draft_counts, spec_rows)。"""
    rows = None
    if self.use_spec_decode and draft_counts is not None:
        if self._recoverssm_capturing and m.is_prefilling is None:
            m = m.replace(
                is_prefilling=torch.zeros(m.num_reqs, dtype=torch.bool)
            )
        if m.is_prefilling is None or m.is_prefilling.device.type != "cpu":
            raise ValueError(
                "GDN RecoverSSM requires a CPU is_prefilling mask"
            )

        # 只让真实请求参与 spec-row 分类；图填充行由 Ascend base builder 继续裁剪。
        real_reqs = m.num_reqs if num_actual_reqs is None else num_actual_reqs
        draft_counts, rows = recoverssm_spec_rows(
            m.is_prefilling,
            m.query_start_loc_cpu,
            draft_counts,
            real_reqs,
        )
    return m, draft_counts, rows
```

这里不单独执行 `m.unpadded()`，避免与现有 Ascend base builder 的图填充处理重复。Dspark 的 reorder/anchor 语义由 base builder 保留，mixin 只将 draft counts 规范化成 Recover 所需的 spec-row 表达。

#### 11.2.4 统一 build 签名

```python
def build(
    self,
    common_prefix_len: int,
    common_attn_metadata: CommonAttentionMetadata,
    num_accepted_tokens: torch.Tensor | None = None,
    num_decode_draft_tokens_cpu: torch.Tensor | None = None,
    fast_build: bool = False,
    num_actual_reqs: int | None = None,
) -> GDNRecoverSSMMetadata:
    m, draft_counts, rows = self._prepare_recoverssm_inputs(
        common_attn_metadata,
        num_decode_draft_tokens_cpu,
        num_actual_reqs,
    )

    # 全量路径进入 AscendGDNAttentionMetadataBuilder；
    # host 路径进入 AscendGDNHostMetadataBuilder。
    meta = super().build(
        common_prefix_len,
        m,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=draft_counts,
        fast_build=fast_build,
        num_actual_reqs=num_actual_reqs,
    )
    return self._make_recoverssm_metadata(meta, rows, m)
```

实际调用顺序因此固定为：

```text
AscendGDNRecoverSSMMetadataMixin.build
  1. _prepare_recoverssm_inputs
     ├─ Dspark/图真实请求边界
     └─ recoverssm_spec_rows
  2. super().build
     ├─ 现有 Dspark reorder/anchor 语义
     ├─ num_actual_reqs 与图容量 guard
     └─ Ascend convolution descriptors
  3. _make_recoverssm_metadata
     ├─ 单 checkpoint 和 accepted placeholder
     ├─ recoverssm_request_indices
     ├─ GDNRecoverSSMCommitMetadata
     ├─ _get_recoverssm_context
     └─ 保留 Ascend 动态字段
```

#### 11.2.5 Recover metadata 构造

```python
def _make_recoverssm_metadata(
    self,
    meta: GDNAttentionMetadata,
    rows: torch.Tensor | None,
    m: CommonAttentionMetadata,
) -> GDNRecoverSSMMetadata:
    base = {f.name: getattr(meta, f.name) for f in fields(meta)}
    commit = None
    context = None

    if meta.num_spec_decodes > 0:
        if rows is None:
            raise ValueError("RecoverSSM spec rows require draft counts")
        n = meta.num_spec_decodes
        assert meta.spec_state_indices_tensor is not None
        assert meta.spec_query_start_loc is not None

        state_indices = meta.spec_state_indices_tensor[:, :1]
        base["spec_state_indices_tensor"] = state_indices
        base["num_accepted_tokens"] = self._recoverssm_ones[
            : state_indices.shape[0]
        ]
        request_indices = recoverssm_request_indices(
            rows, n, m.query_start_loc.device
        )

        align = self.vllm_config.cache_config.mamba_cache_mode == "align"
        commit = GDNRecoverSSMCommitMetadata(
            state_indices=state_indices,
            query_start_loc=meta.spec_query_start_loc,
            request_indices=request_indices,
            block_table=m.block_table_tensor if align else None,
            num_computed_tokens=(
                m.compute_num_computed_tokens() if align else None
            ),
            block_size=self.kv_cache_spec.block_size if align else None,
        )
        context = self._get_recoverssm_context()

    recover = GDNRecoverSSMMetadata(
        **base,
        recoverssm_commit=commit,
        recoverssm_context=context,
    )

    # GDNRecoverSSMMetadata 非 slots dataclass。
    # dataclass fields 已用于构造；Ascend 动态字段在这里逐一保留。
    for name, value in vars(meta).items():
        if name not in base:
            setattr(recover, name, value)
    return recover
```

动态字段包括但不限于：

- `spec_decode_metadata`；
- `non_spec_prefill_metadata`；
- `non_spec_decode_metadata`；
- `pcp_context`；
- host-only 后续新增字段。

因此不需要再声明 `AscendGDNRecoverSSMMetadata` 子类，也不会因上游 `fields(meta)` 只覆盖 dataclass 字段而丢失 Ascend 信息。

#### 11.2.6 Graph capture 与 backend 选择

```python
def build_for_cudagraph_capture(
    self,
    common_attn_metadata: CommonAttentionMetadata,
) -> GDNRecoverSSMMetadata:
    self._recoverssm_capturing = True
    try:
        # base capture 内部会回调本 mixin 的 build()。
        return super().build_for_cudagraph_capture(common_attn_metadata)
    finally:
        self._recoverssm_capturing = False
```

```python
class AscendGDNRecoverSSMAttentionBackend(GDNRecoverSSMAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[GDNAttentionMetadataBuilder]:
        if (
            get_pcp_group().world_size == 1
            and AscendGatedDeltaNetAttention._supports_host_metadata_prefill()
        ):
            return AscendGDNHostRecoverSSMMetadataBuilder
        return AscendGDNRecoverSSMMetadataBuilder
```

`AscendGatedDeltaNetAttention.get_attn_backend()` 仅在 `uses_recoverssm()` 为真时返回该 backend，否则继续返回 `AscendGDNFusedAttentionBackend`。

#### 11.2.7 采样后调用

Recover metadata 的 commit 方法直接复用上游实现：

```python
# Dspark 或其他 speculative sampler 之后，由通用 coordinator 调用：
postprocess = recover_metadata.commit_recoverssm_state(
    actual_accepted_count
)

# 上游 GDNRecoverSSMMetadata 内部继续调用：
context.commit(
    actual_accepted_count,
    commit.state_indices[:num_spec_decodes, 0],
    commit.query_start_loc[: num_spec_decodes + 1],
    request_indices=commit.request_indices,
    block_table=commit.block_table,
    num_computed_tokens=commit.num_computed_tokens,
    mamba_block_size=commit.block_size,
)
```

至此，Dspark 适配层只存在于 `_prepare_recoverssm_inputs()` 之前的输入形成和 base builder 中；Recover mixin、metadata、context 和 commit 均不读取 speculative method。若上游后续抽出等价的 Recover builder 模板，可进一步删除 Ascend mixin 中的通用后处理代码，只保留这些 hook。

### 11.3 Metadata 类型

优先复用：

- `GDNRecoverSSMCommitMetadata`；
- `GDNRecoverSSMMetadata`；
- `GDNRecoverSSMAttentionBackend`。

Ascend backend 只负责返回衔接两层后的 builder。现有 prefill/decode metadata 和 Dspark 特有 metadata 仍由原 Ascend builder 生成；Recover metadata 只增加状态恢复所需信息。

### 11.4 两种 accepted count

- `accepted_placeholder`：verify 前全 1，供 checkpoint/conv 窗口协议使用，不是真实接受长度，也不由 Dspark 的采样结果决定。
- `actual_accepted_count`：Dspark 或其他方法采样后，由统一 `RecoverSSMState.commit_step()` 接口传入，是 plan 和 commit 的唯一长度依据。

方法适配层可以决定 accepted count 的计算方式，但 Recover 层只接受规范化后的连续前缀长度，不再读取 method。

## 12. Ascend Layer 接入（以 Dspark 为例）

`AscendGatedDeltaNetAttention` 只做四类改动：

1. 使用上游 `uses_recoverssm()` 判断能力，不判断 speculative method 是否为 MTP。
2. `get_state_shape/get_state_dtype` 返回上游一致的三类 cache。
3. Recover 时选择 Ascend Recover backend，否则保持 `AscendGDNFusedAttentionBackend`。
4. speculative recurrent 分支调用上游 `gdn_recoverssm_verify`；其他路径不变。

Dspark 场景下，Recover 分支的控制流与上游一致：

```text
Dspark target metadata 规范化（沿用现有 builder）
    ↓
causal_conv1d_update（扩展窗口、单 checkpoint、placeholder=1）
    ↓
构造 q/k/v views，必要时 contiguous fallback
    ↓
gdn_recoverssm_verify(raw a/b, q/k/v)
    ↓
output 直接写入或合并到 core_attn_out
    ↓
Dspark 采样后由通用 coordinator 触发 commit
```

保持不变：

- Dspark 的 draft 生成、调度、anchor 和接受采样逻辑；
- prefill chunk/fused AscendC 选择；
- 普通 decode recurrent；
- 非 speculative gating/L2 norm；
- 输出 RMSNorm/gate 和 output projection；
- KV connector 的 layer wait/save hooks。

其他 speculative 方法复用完全相同的 layer 分支；方法差异只能存在于现有 metadata 和 sampler 中，不能进入 GDN Recover kernel。

## 13. NPU Graph 与生命周期

### 13.1 图协议

复用上游最新的 `AttentionCGSupport.UNIFORM_BATCH` 方向：

- speculative rows 使用固定 staging buffer。
- padding rows 使用 null state 和零 query length。
- verify 在图内执行，commit 在 sampling 后、图外执行。
- commit 使用 staging 后的 state indices/query offsets。
- oversized eager fallback 不得写超 capture/token/state capacity。

Ascend adapter 必须额外保留 `num_actual_reqs` 的截断逻辑，不能因调用上游 build signature 而丢失图填充处理。

### 13.2 Hot path 禁止项

- `.item()` 或 device→host 同步；
- commit 时重新查询 `data_ptr()`；
- 按 batch 分配新 tensor；
- 阻塞 H2D copy；
- 依赖 host accepted count 的 Python 分支。

### 13.3 Profiling 与 teardown

同步上游 `RecoverSSMState.reset()` 语义：

- profiling 后清除已记录 step；
- 释放 metadata→context→profiling KV cache 的引用链；
- 防止正式 KV cache 分配前仍占用 profiling cache。

Ascend worker 若使用独立 teardown，也必须增加对应 reset 测试。

## 14. PD 分离（后续工作包）

沿用 Ascend #19133 的通用 MambaSpec 思路：

- `num_speculative_blocks=0` 时保留唯一 checkpoint block。
- Mooncake transfer planner 不再裁剪 speculative tail。
- 传输 conv/checkpoint/replay 三类 cache 的 shape/dtype/stride/address。
- P/D 两侧模型、TP、Q 和 Recover 开关一致。

Replay 是 scratch record，首版可随整页传输以保持通用实现简单；后续若跳过 replay 传输以节约带宽，必须独立设计 page accounting 和地址规则。

## 15. 建议代码落点

### 15.1 上游代码，保持上游维护

- `vllm/model_executor/layers/mamba/ops/recoverssm.py`
- `vllm/model_executor/layers/mamba/recoverssm_utils.py`
- `vllm/model_executor/layers/mamba/gdn/recoverssm_gdn.py`
- `vllm/v1/attention/backends/gdn_recoverssm.py`
- `vllm/v1/attention/backends/recoverssm_metadata.py`
- `vllm/v1/worker/gpu/model_states/recoverssm.py`
- Qwen 模型级 shape/dtype helper 和 `SupportsReplaySSM` 声明

### 15.2 Ascend 只保留适配代码

| 建议位置 | 职责 |
|---|---|
| `vllm_ascend/ops/gdn.py` | Ascend layer 的 backend 选择和 Recover 分支 |
| `vllm_ascend/ops/gdn_attn_builder.py` | 薄 metadata adapter，保留 Ascend conv descriptors |
| `vllm_ascend/ops/triton/gdn/recoverssm.py` | 可选；仅在上游 kernel 不兼容或需要调优时存在 |
| `vllm_ascend/patch/platform/patch_replayssm_config.py` | Ascend 特有配置限制，不复制上游通用开关 |
| Ascend worker/model state | 调用 coordinator、profiling reset 和图生命周期适配 |
| Mooncake connector | 后续三类 cache 的 PD 适配 |

如果上游 Triton 可以直接运行，Ascend 不应新增第三份 GDN RecoverSSM 算子文件。

## 16. 分阶段实施计划

### 阶段 0：锁定上游基线并做零改动验证

- 以 #58863 + #61309 的同一组提交为基线，避免混用不同 replay/commit 版本。
- 在 A3/A5 直接 import 上游 GDN RecoverSSM。
- 原样运行 verify、commit、plan、compaction smoke tests。
- 记录不支持的 Triton 语法、dtype、stride、UB 和性能问题。
- 只有此阶段证明必须修改时，才建立 Ascend kernel 文件。

### 阶段 1：上游协议下的 Eager MVP

- 接通 `use_recoverssm`、三类 cache 和模型级 helper。
- 实现 Ascend metadata 薄 adapter。
- 接入 layer Recover 分支。
- 直接复用上游 verify/commit/context。
- 先通过 eager 的上游算子测试和 Dspark 场景下的 Qwen GDN 功能测试。

### 阶段 2：NPU Graph 与服务集成

- 保留 `num_actual_reqs`、graph staging 和 oversized fallback。
- 验证 mixed batch、draftless decode、null padding 和连续多轮 commit。
- 复用上游 align/prefix caching，并验证 Dspark 接受窗口跨 block、exact-boundary、prefix 命中后继续投机解码的组合场景。
- 增加 profiling teardown reset。
- 进行 Dspark 真实权重 Recover on/off 回归，并用其他可用 speculative 方法做统一契约 smoke test。

### 阶段 3：按需 Ascend 优化

- 仅在 profile 证明必要时调整 BV、num_warps、静态 Q、conv tile 和 stride 特化。
- 优化不得改变 replay record、公共 API 和 forward commit 语义。
- Mooncake 三类 cache 适配。
- 测试实际 HBM、最大并发、verify/commit latency 和 TPOT。

### 阶段 4：开放更多上游能力

- interleaved GQA。
- 更多并行、head dim 和 dtype 组合。

## 17. 测试与验收

### 17.1 直接移植上游 GDN 测试

尽量保留上游测试原意和严格程度：

- verify output 与原生 recurrent 对照，checkpoint 逐位不变。
- FP32 commit 与逐 token state `torch.equal`。
- accepted length 1/2/3/4、极端 gate、多个 seed。
- malformed input、短 metadata、不一致 layer。
- mixed batch spec rows 和非连续 request mapping。
- align exact-boundary、cross-boundary、BF16 state。
- strided Q/K/V views、in-place output、variable query length。
- graph padding rows 不写 output/replay/checkpoint 的有效区域。
- pointer table 保留最高位。
- 目标 Qwen GDN 模型及实际 speculative 配置的能力声明与三类 cache shape/dtype 一致，不要求存在 MTP drafter。

### 17.2 Ascend 额外测试

- 上游 kernel 在 A3/A5 的编译和功能 smoke。
- 验证“方法/图输入适配 → Recover spec-row 规范化 → Ascend base build → Recover commit metadata”的调用顺序。
- 分别验证全量和 host-only 两条继承链的 `super().build()` 路由、`build_for_cudagraph_capture()` 回调及 `_recoverssm_capturing` 的异常安全复位。
- 验证 `GDNRecoverSSMMetadata` 构造后完整保留 `vars(meta)` 中的 Ascend 动态字段，且无需新增 Ascend metadata 子类。
- Ascend conv descriptors 和 Dspark reorder/draft-count 语义在 Recover metadata 转换后不丢失。
- `num_actual_reqs` 图填充与 oversized eager fallback。
- eager/NPU Graph 的 output、replay 和 committed state 一致。
- Dspark 动态 draft count、无 draft 轮次、ragged window 和接受前缀边界的回归。
- 至少一种其他可用 speculative 方法的统一接口 smoke，验证不存在 Dspark 专用 kernel 分支。
- profiling cache teardown 后无遗留引用。
- Mooncake 三类 cache 地址、长度和 block 裁剪。
- 与 AscendC 原生 recurrent 对照时使用明确的后端容差，不混同上游 Triton bitwise 测试。

### 17.3 服务与性能验收

至少报告：

- GDN cache 实际 HBM 与理论公式差异。
- 最大 decode 并发。
- verify、commit、conv compaction 分项延迟。
- 不同并发、接受率、speculative window 长度下的 TPOT，以 Dspark 数据为主。
- acceptance rate/length 和模型精度。
- A3/A5 分开结果以及 JIT/recompilation 情况。

## 18. 风险与控制

| 风险 | 控制方式 |
|---|---|
| 上游 PR 尚未稳定或合入 | 固定同一上游提交系列；下游只保留薄适配，避免形成长期 fork |
| 将 RecoverSSM 绑定到 MTP 或 Dspark 专用路径 | 只消费统一 target metadata/accepted count；在 kernel 和 cache 中禁止 method 分支 |
| 把“方法无关设计”等同于“所有方法已验证” | Dspark 做完整验收，其他方法按统一契约逐项补 smoke 和回归 |
| 直接套用 #19133 的 K3 builder | 只借鉴统一接口和图容量保护；保留 Ascend GDN builder 的 Dspark 与卷积逻辑 |
| 两层重复做请求分区或顺序错误 | 固定适配层→Recover 规范化→base build→commit metadata 顺序，并做字段/行数守恒测试 |
| Ascend 复制整套上游实现 | 代码审查要求说明每一处不能复用的硬件原因 |
| metadata 转换丢失 Ascend conv 字段 | 显式 hook 和字段保留测试，不依赖 `fields(meta)` 猜测完整状态 |
| 混用旧闭式 commit 与新 replay | 基线必须包含 decay-record + forward-commit，禁止跨版本拼接 |
| align 源页读写竞争 | 保留 final/boundary 两次 launch 和 exact-boundary 修复 |
| profiling cache 泄漏 | 同步 `RecoverSSMState.reset()` 和 teardown 测试 |
| 为省显存改变 replay layout | 首版禁止；先与上游一致，再作为独立上游优化讨论 |
| AscendC 与 Triton 数值混淆 | 分开定义 bitwise 后端对照和跨后端容差 |

## 19. 结论

Qwen GDN RecoverSSM 的最终架构应是：

- **Dspark/方法适配层负责形成标准输入**：保留 draft 生成、target reorder、anchor、采样、Ascend 卷积描述符和图容量处理。
- **方法无关 Recover 层负责状态恢复**：三类 cache、FP32 replay、单 checkpoint、GDN verify、forward commit、通用 plan/compaction 和 coordinator。
- **上游拥有 Recover 协议和主要实现**，Ascend 只保留两层衔接、NPU Graph 生命周期、必要的 Triton/tile 适配和 Mooncake。
- **Ascend #19133 提供统一接口与图处理经验，而不是提供需要照搬的 K3 metadata**。

最重要的实施纪律是：先在 A3/A5 直接运行上游代码，再决定需要哪些 Ascend 修改。以 Dspark 完成端到端验收，同时保持 kernel、cache 和 commit 的方法无关性，可以最大限度降低重复代码、上游同步成本和数值语义分叉。

## 参考资料

- [vLLM PR #58863：通用 RecoverSSM 与 GDN 实现](https://github.com/vllm-project/vllm/pull/58863)
- [vLLM PR #61309：Qwen3.5 / Qwen3-Next GDN RecoverSSM](https://github.com/vllm-project/vllm/pull/61309)
- [vllm-ascend PR #19133：GLM KDA RecoverSSM](https://github.com/vllm-project/vllm-ascend/pull/19133)
- [当前 Ascend GDN 实现](https://github.com/vllm-project/vllm-ascend/blob/e6ac48c006b1c9c17379b44c1f2bc05fa3c09eab/vllm_ascend/ops/gdn.py)
- [当前 Ascend GDN Metadata Builder](https://github.com/vllm-project/vllm-ascend/blob/e6ac48c006b1c9c17379b44c1f2bc05fa3c09eab/vllm_ascend/ops/gdn_attn_builder.py)
