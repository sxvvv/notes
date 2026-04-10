# 10 TinyML: MCUNet

> 📺 [Lecture 10 - MCUNet](https://hanlab.mit.edu/courses/2024-fall-65940)
> 📄 [Slides](https://hanlab.mit.edu/courses/2024-fall-65940)

---

## 10.1 问题: 在微控制器上跑深度学习

**微控制器 (MCU) 的极限约束**:
- SRAM: 通常只有 **256KB** (比 GPU 显存小 100万倍)
- Flash: 通常 1-2MB
- 算力: 几百 MHz，无 GPU
- 功耗: 几十 mW

> 对比: 一个 ResNet50 模型大小约 25MB，是 MCU Flash 的 25 倍。怎么可能在 MCU 上跑深度学习？

---

## 10.2 MCUNetV1: TinyNAS + TinyEngine

### 10.2.1 Joint Optimization

核心创新: **同时搜索模型架构和推理引擎**，而不是分别优化。

```
传统: NAS 找架构 → 部署到引擎 (可能放不下)
MCUNet: TinyNAS + TinyEngine 联合搜索 (保证能放下)
```

### 10.2.2 TinyNAS

在 MCU 内存约束下的 NAS:
- 搜索空间: MobileNet-like (inverted bottleneck blocks)
- 约束: 峰值 SRAM < 256KB, 模型大小 < 1MB
- 方法: 在约束内搜索最优精度

关键技巧: **搜索空间的约束感知设计**
- 不是搜"最好的模型"然后看放不放得下
- 而是只在"放得下"的模型中搜最好的

### 10.2.3 TinyEngine

推理引擎的编译时优化:

| 优化 | 描述 | 效果 |
|------|------|------|
| 内存规划 | 编译时确定每层 tensor 生命周期 → 内存复用 | SRAM 减少 2-4x |
| 算子融合 | 多个小算子合成一个 kernel | 减少中间结果存储 |
| 循环展开 | 消除运行时循环 overhead | 加速 1.5-2x |
| In-place | 直接覆盖不再需要的 tensor | 减少峰值内存 |

```python
# 内存规划示意: tensor 生命周期分析
# 假设3个中间tensor, 分析哪些可以复用同一块内存
tensors = {
    'conv1_out': {'start': 0, 'end': 3, 'size': 1024},
    'conv2_out': {'start': 1, 'end': 4, 'size': 2048},
    'conv3_out': {'start': 2, 'end': 5, 'size': 1024},
}
# conv1_out 在 step 3 结束, conv3_out 在 step 2 开始 → 不能复用
# conv1_out 结束后空间可以给 conv3_out → 可以复用!
# 峰值内存: max(同时活跃tensor大小之和) 而非 所有tensor大小之和

def memory_plan(tensors):
    """贪心内存规划: 尽量复用已释放的内存块"""
    allocated = {}  # tensor_name -> memory_offset
    free_blocks = []  # [(offset, size)]
    next_offset = 0

    for name, info in sorted(tensors.items(), key=lambda x: x[1]['start']):
        # 先回收已结束的 tensor
        for alloc_name in list(allocated):
            if tensors[alloc_name]['end'] <= info['start']:
                offset = allocated.pop(alloc_name)
                free_blocks.append((offset, tensors[alloc_name]['size']))

        # 尝试从 free_blocks 中找合适的
        placed = False
        for i, (offset, size) in enumerate(free_blocks):
            if size >= info['size']:
                allocated[name] = offset
                free_blocks.pop(i)
                placed = True
                break

        if not placed:
            allocated[name] = next_offset
            next_offset += info['size']

    return next_offset  # 总内存需求
```

### 10.2.4 成果

- 在 **256KB SRAM** 的 MCU 上跑 ImageNet 分类
- 精度: 70%+ (远超之前 <50% 的方法)
- 推理时间: < 1s

---

## 10.3 MCUNetV2

### 10.3.1 问题: 语义分割需要更大的特征图

MCUNetV1 做分类只需要最后一层特征。但语义分割需要每个像素的输出 → 中间特征图太大放不下。

### 10.3.2 Patch-based Inference

把大图切成小块 (patch)，逐块推理:
- 每次只需要存一个 patch 的特征图
- 峰值内存从 O(H×W) 降到 O(patch_size²)

```
全图推理: 需要 H×W×C 的特征图 → 放不下
Patch推理: 每次 patch_size²×C → 放得下!
代价: patch 边界有重叠 → 计算量增加
```

### 10.3.3 联合搜索

同时搜索: 架构 + patch 大小 + 重叠率
- Patch 大 → 单次推理内存大，但重叠少
- Patch 小 → 内存小，但重叠多（效率低）

---

## 10.4 现实意义

### 与 LLM 推理的共性

| | MCU 推理 | LLM 推理 |
|---|---------|---------|
| 瓶颈 | SRAM 太小 | GPU 显存不够 |
| 解决 | 内存规划 + patch推理 | KV Cache管理 + PagedAttention |
| 核心 | 用计算换内存 | 用计算换内存 / 用内存换计算 |

> **洞察**: MCUNet 的内存规划和 vLLM 的 PagedAttention 本质上是同一个问题 — 在有限内存中高效管理 tensor 的生命周期。

---

## Infra 实战映射

### 对 LLM Infra 的启示
- MCUNet 的内存规划思路可以直接用在 LLM 推理中
- vLLM 的 PagedAttention 和 MCUNet 的内存复用是同一个思想
- Patch-based inference 和 pipeline parallelism 都是"切分处理"的思路

---

## 跨 Lecture 关联

- **前置 ←** [Lec07: NAS](../lec07-nas-I/README.md) — TinyNAS 是 hardware-aware NAS
- **前置 ←** [Lec08: 高效NAS](../lec08-nas-II/README.md) — Once-for-All 思路
- **延伸 →** [Lec11: 推理引擎](../lec11-tiny-engine/README.md) — TinyEngine 的编译优化细节

---

## 面试高频题

**Q1: MCUNet 的核心创新是什么？**
> A: Joint optimization — 同时搜索模型架构(TinyNAS)和推理引擎(TinyEngine)，保证在硬件约束内找到最优方案。传统方法是分开优化，可能找到的架构放不下。

**Q2: MCU 推理和 GPU 推理的最大区别？**
> A: MCU 是内存受限（SRAM 只有几百 KB），GPU 是计算受限或带宽受限。MCU 上优化的核心是内存规划，GPU 上的核心是计算效率和带宽利用率。
