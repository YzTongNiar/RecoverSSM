# Qwen3.6 前向传播流程

## 1. 核心架构概览

Qwen3.6 并不是“每一层都执行标准自注意力”的传统 Transformer，而是采用了 **Gated DeltaNet 线性注意力与 Gated Attention 全注意力交替排列的混合架构**。在此基础上，不同版本分别使用稠密前馈网络或混合专家网络。

目前官方开源的两个主要版本如下：

| 模型 | 层数与排列方式 | 前馈网络 | 激活方式 |
| --- | --- | --- | --- |
| Qwen3.6-27B | 64 层；16 × [3 ×（Gated DeltaNet → FFN）→ 1 ×（Gated Attention → FFN）] | 稠密 SwiGLU FFN | 不使用专家路由 |
| Qwen3.6-35B-A3B | 40 层；10 × [3 ×（Gated DeltaNet → MoE）→ 1 ×（Gated Attention → MoE）] | 稀疏 MoE | 每个 token 选择 8 个路由专家，并经过 1 个共享专家；总参数 35B，激活约 3B |

因此，Qwen3.6 的前向传播可以概括为：

> **输入嵌入 → 混合 Gated DeltaNet／全注意力层 → 稠密 FFN 或 MoE → 最终归一化 → LM Head → 下一 token 的 logits。**

---

## 2. 整体数据流

以文本生成为例，主干前向传播如下：

```text
输入文本
   ↓
Tokenizer：文本 → token IDs
   ↓
Embedding：token IDs → 隐状态 X⁰
   ↓
重复执行若干 Decoder Layer：
   RMSNorm
      ↓
   Gated DeltaNet 或 Gated Attention
      ↓
   残差连接
      ↓
   RMSNorm
      ↓
   FFN（27B）或 MoE（35B-A3B）
      ↓
   残差连接
   ↓
最终 RMSNorm
   ↓
LM Head
   ↓
词表 logits
   ↓
Softmax / 采样，得到下一个 token
```

如果输入包含图像或视频，视觉编码器会先将视觉输入转换成视觉 token 表征，再将这些表征填入多模态输入序列，与文本一起进入语言主干。

---

## 3. 输入嵌入

设 batch 大小为 $B$，序列长度为 $T$，隐藏维度为 $d$。Tokenizer 首先将文本转换为 token IDs：

$$
I\in\mathbb{Z}^{B\times T}
$$

Embedding 层再将离散 ID 映射为连续隐状态：

$$
X^0=\operatorname{Embedding}(I)\in\mathbb{R}^{B\times T\times d}
$$

Qwen3.6 不像早期 Transformer 那样，在输入端统一叠加绝对位置向量。其顺序与位置信息主要由以下机制处理：

- **全注意力层**：通过 RoPE／多模态 RoPE 注入位置信息。
- **Gated DeltaNet 层**：通过因果卷积和递归状态更新感知 token 顺序。

---

## 4. Decoder Layer 的通用结构

每一层都由两个 Pre-Norm 残差子层组成。第一个子层负责跨 token 交换信息，第二个子层负责对每个 token 的表示进行非线性变换：

$$
H=X+\operatorname{TokenMixer}(\operatorname{RMSNorm}(X))
$$

$$
X'=H+\operatorname{FFN/MoE}(\operatorname{RMSNorm}(H))
$$

其中：

- **TokenMixer**：Gated DeltaNet 或 Gated Attention，负责混合不同位置的信息。
- **FFN／MoE**：对每个 token 的隐藏表示进行进一步变换。
- **RMSNorm**：稳定各层数值尺度。
- **残差连接**：保留原始信息，并改善深层网络的梯度传播。

对应的简化伪代码为：

```python
def decoder_layer(x, layer_type):
    residual = x
    u = rms_norm(x)

    if layer_type == "linear_attention":
        mixed = gated_delta_net(u)
    else:
        mixed = gated_attention(u)

    h = residual + mixed

    residual = h
    v = rms_norm(h)
    transformed = ffn(v)       # Qwen3.6-27B
    # transformed = moe(v)     # Qwen3.6-35B-A3B

    return residual + transformed
```

---

## 5. Gated DeltaNet：高效压缩和传递历史

Gated DeltaNet（GDN）是一种带门控和记忆修正机制的线性注意力。它不让当前 token 与所有历史 token 两两计算注意力，而是将历史信息压缩到固定大小的递归状态中。

### 5.1 生成 Q、K、V 与控制门

归一化后的隐状态经过线性投影，得到：

- $Q,K,V$：分别用于读取、寻址和内容表示；
- $z$：输出门；
- $\beta$：控制新信息的写入强度；
- $g$：控制旧状态的衰减程度。

$Q,K,V$ 会先经过**深度因果 Conv1D 和 SiLU 激活**，融合局部上下文；随后对 $Q,K$ 进行 L2 归一化。

### 5.2 更新递归记忆

每个头维护一个状态矩阵 $S_t$。省略头维度后，单 token 的更新过程可简化表示为：

$$
\bar S_t=e^{g_t}S_{t-1}
$$

$$
S_t=\bar S_t+k_t\left[\beta_t\left(v_t-\bar S_t^{T}k_t\right)\right]^T
$$

$$
y_t=S_t^Tq_t
$$

可以将其直观理解为：

1. $e^{g_t}S_{t-1}$：对旧记忆进行衰减，决定保留多少历史信息。
2. $\bar S_t^Tk_t$：根据当前 key 从旧记忆中读取已有内容。
3. $v_t-\bar S_t^Tk_t$：计算当前内容与旧记忆之间的预测误差。
4. 按 $\beta_t$ 将这份误差写回状态，这就是 **delta rule**。
5. 最后使用 $q_t$ 从更新后的状态中读取输出。

输出还会经过带 $z$ 门的 RMSNorm 和输出投影，随后写回残差流。

### 5.3 Prefill 与 Decode 的实现差异

- **Prefill 阶段**：通常使用分块并行算法处理整段输入，避免完全逐 token 执行。
- **Decode 阶段**：每次只处理新 token，并更新递归状态。
- 除递归状态外，还需要保存短因果卷积所需的卷积状态。

GDN 层的状态大小不随上下文长度线性增长，这是 Qwen3.6 能降低长上下文推理成本的重要原因。

---

## 6. Gated Attention：精确检索历史 token

每第 4 层使用标准因果全注意力，并加入 Q/K 归一化、GQA 和输出门控等设计。

### 6.1 投影、归一化与位置编码

Q 投影会同时生成 query 和输出门 $g$。Q、K 按注意力头分别进行 RMSNorm，然后对部分维度应用 RoPE。多模态输入使用相应的多模态位置编码机制。

### 6.2 GQA 因果注意力

注意力计算为：

$$
A=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_h}}+M\right)V
$$

其中 $M$ 是因果掩码，保证位置 $t$ 不能读取未来 token。

模型采用 **GQA（Grouped-Query Attention，分组查询注意力）**：多个 Q 头共享较少的 K/V 头，从而减少 KV cache 和相关计算开销。

注意力输出还会经过门控和输出投影：

$$
Y=\left(A\odot\sigma(g)\right)W_o
$$

其中 $\sigma(g)$ 控制当前注意力结果写回残差流的强度。

混合架构中的两类 token mixer 可以概括为：

- **GDN 层**：以较低成本压缩、保存和传递较长历史。
- **全注意力层**：保留对历史 token 的直接、精确检索能力。

---

## 7. FFN 与 MoE：两个版本的主要差异

### 7.1 Qwen3.6-27B：稠密 SwiGLU FFN

每个 token 都经过同一个前馈网络：

$$
\operatorname{FFN}(x)=W_{\text{down}}\left[\operatorname{SiLU}(W_{\text{gate}}x)\odot W_{\text{up}}x\right]
$$

这里的“稠密”是指 **FFN 不使用 MoE 专家路由**，并不表示模型所有层都采用传统全注意力。

### 7.2 Qwen3.6-35B-A3B：稀疏 MoE

路由器首先为每个 token 计算 256 个专家的概率：

$$
p=\operatorname{softmax}(W_rx)
$$

然后选出概率最高的 8 个专家，并对选中的路由权重重新归一化：

$$
Y_{\text{routed}}=\sum_{i\in\operatorname{TopK}(p)}p_iE_i(x)
$$

每个专家都是一个小型 SwiGLU FFN。此外，还有一个始终参与计算的共享专家，其输出经过 sigmoid 门控：

$$
Y_{\text{MoE}}=Y_{\text{routed}}+\sigma(g_s(x))E_{\text{shared}}(x)
$$

两类组件的作用分别是：

- **路由专家**：根据 token 内容选择不同的专门计算路径。
- **共享专家**：提供所有 token 都可以利用的公共变换。

MoE 因而能够扩大模型总参数量，同时只激活其中一部分参数参与每个 token 的计算。

---

## 8. 输出层与下一 token 预测

所有 Decoder Layer 计算完成后，隐状态先经过最终 RMSNorm，再由 LM Head 映射到词表空间：

$$
Z=W_{\text{lm}}\operatorname{RMSNorm}(H^L)
$$

其中 $Z$ 是 **logits**，还不是概率。自回归生成时通常取最后一个有效位置的 logits：

$$
P(x_{T+1}\mid x_{1:T})=\operatorname{softmax}(Z_T)
$$

之后通过 greedy、top-p、top-k 等策略选择下一个 token。严格来说，**采样属于前向传播之后的解码步骤，不属于模型内部的前向计算**。

训练时则通常对多个位置的 logits 计算下一 token 交叉熵损失，而不是只使用最后一个位置。

---

## 9. 连续生成时的缓存机制

Qwen3.6 的混合架构决定了两类缓存并存：

```text
Gated DeltaNet 层：
    Conv1D 状态＋固定大小的递归状态

Gated Attention 层：
    随上下文长度增长的 K/V Cache
```

完整生成过程分为：

1. **Prefill**：处理完整提示词，得到第一个输出分布，同时建立各层缓存。
2. **Decode**：输入刚生成的 token，复用历史状态，不必重新计算整段提示词。
3. 得到下一个 token 后，将其作为下一轮输入，重复 Decode。

因此，并非所有层都需要保存不断增长的 KV cache：只有全注意力层使用标准 KV cache，GDN 层使用固定大小的递归状态和短卷积状态。

模型附带的 **MTP（Multi-Token Prediction，多 token 预测）模块**可用于训练或投机解码加速，但不属于理解基本 next-token 前向流程所必需的主干路径。

---

## 10. 总结

Qwen3.6 的前向传播可以归纳为三个关键设计：

1. **混合 token mixer**：Gated DeltaNet 负责高效压缩历史，全注意力负责精确回看。
2. **两种前馈路径**：27B 使用稠密 FFN，35B-A3B 使用带共享专家的稀疏 MoE。
3. **混合状态缓存**：GDN 层维护递归与卷积状态，全注意力层维护 KV cache。

最终的数据流为：

```text
Token IDs
  → Embedding
  → [RMSNorm → GDN / Gated Attention → 残差
     → RMSNorm → FFN / MoE → 残差] × N
  → Final RMSNorm
  → LM Head
  → Logits
  → Softmax / 采样
  → 下一个 token
```

---

## 参考资料

1. [Qwen3.6-27B 官方模型卡](https://huggingface.co/Qwen/Qwen3.6-27B)
2. [Qwen3.6-35B-A3B 官方模型卡](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
3. [Hugging Face Transformers：Qwen3.5／Qwen3.6 稠密架构实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
4. [Hugging Face Transformers：Qwen3.5／Qwen3.6 MoE 架构实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)
