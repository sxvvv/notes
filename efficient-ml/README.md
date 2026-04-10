# EfficientML Notes | 高效深度学习笔记

> MIT 6.S965 / 6.5940 — TinyML and Efficient Deep Learning Computing
> Instructor: [Song Han](https://hanlab.mit.edu) (MIT EECS)
> Course: [efficientml.ai](https://efficientml.ai) | [YouTube Playlist](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB)

Inspired by [erectbranch/MIT-Efficient-AI](https://github.com/erectbranch/MIT-Efficient-AI) (Korean notes), with added:
- **Infra 实战映射** — 每个技术在 vLLM / TensorRT-LLM / 沐曦 MACA 中怎么用
- **可运行代码** — 不只有公式，还有 Python / PyTorch 代码片段
- **跨 Lecture 关联** — 技术树而非孤立笔记
- **面试高频题** — AI Infra 工程师面试视角
- **双语关键术语** — 中文直觉 + 英文术语

---

## 技术关系图

```
EfficientML
├── Efficient Inference (推理优化)
│   ├── Quantization (量化)
│   │   ├── Basics: 数值类型 / 线性量化 / 对称非对称 ──── [Lec05]
│   │   ├── Advanced: PTQ / QAT / STE / 低比特 ─────────── [Lec06]
│   │   └── LLM: SmoothQuant / AWQ / GPTQ / QServe ────── [Lec13]
│   │
│   ├── Pruning (剪枝)
│   │   ├── Basics: 粒度 / 准则 / Magnitude / Saliency ── [Lec03]
│   │   ├── Advanced: Lottery Ticket / 自动剪枝 / 硬件支持 ─ [Lec04]
│   │   └── LLM: SparseGPT / Wanda / Mixture-of-Experts ── [Lec13]
│   │
│   ├── NAS (神经架构搜索)
│   │   ├── Basics: 搜索空间 / 搜索策略 ────────────────── [Lec07]
│   │   └── Advanced: Weight Sharing / Hardware-Aware ───── [Lec08]
│   │
│   └── Knowledge Distillation (知识蒸馏)
│       └── Teacher-Student / Soft Labels / Feature Match ── [Lec09]
│
├── Domain-Specific (领域优化)
│   ├── Transformer & LLM ────────────────────────────────── [Lec12]
│   ├── LLM Post-Training (PEFT / LoRA / Prompt) ────────── [Lec14]
│   ├── Long Context & Efficient Attention ───────────────── [Lec15]
│   ├── Vision Transformer ──────────────────────────────── [Lec16]
│   └── TinyML & MCUNet ─────────────────────────────────── [Lec10]
│
└── System Support (系统支持)
    └── TinyEngine / Compiler / Kernel Fusion / SIMD ────── [Lec11]
```

---

## 4-Day 学习路径

### Day 1: 量化专题 (Quantization)
| Lecture | Topic | Notes |
|---------|-------|-------|
| Lec02 | DL Basics + 效率指标 | [basics](lec02-basics/README.md) |
| Lec05 | 量化基础：数值类型、线性量化 | [quantization-I](lec05-quantization-I/README.md) |
| Lec06 | 进阶量化：PTQ、QAT、低比特 | [quantization-II](lec06-quantization-II/README.md) |
| Lec13 | LLM部署：量化 + 推理系统 | [llm-deploy](lec13-llm-deploy/README.md) |
| Lec14 | LLM后训练：PEFT / LoRA | [llm-post-training](lec14-llm-post-training/README.md) |

### Day 2: 剪枝 + NAS + 蒸馏
| Lecture | Topic | Notes |
|---------|-------|-------|
| Lec03 | 剪枝基础：粒度、准则 | [pruning-I](lec03-pruning-I/README.md) |
| Lec04 | 进阶剪枝：Lottery Ticket、硬件支持 | [pruning-II](lec04-pruning-II/README.md) |
| Lec07 | NAS基础：搜索空间、搜索策略 | [nas-I](lec07-nas-I/README.md) |
| Lec08 | 高效NAS：Weight Sharing | [nas-II](lec08-nas-II/README.md) |
| Lec09 | 知识蒸馏 | [distillation](lec09-distillation/README.md) |
| Lec10 | TinyML: MCUNet | [mcunet](lec10-mcunet/README.md) |

### Day 3: LLM + 多模态高效推理
| Lecture | Topic | Notes |
|---------|-------|-------|
| Lec12 | Transformer架构 | [transformer](lec12-transformer/README.md) |
| Lec15 | 长上下文 + 高效注意力 | [long-context](lec15-long-context/README.md) |
| Lec16 | Vision Transformer | [vit](lec16-vit/README.md) |
| Lec11 | 推理引擎 + 编译优化 | [tiny-engine](lec11-tiny-engine/README.md) |

### Day 4: 高效训练 + 总结
分布式训练（DP / TP / PP / ZeRO）+ 4天总复盘

---

## Lecture Notes Index

| # | Title | Key Topics |
|---|-------|------------|
| 02 | [DL Basics & Efficiency Metrics](lec02-basics/README.md) | 参数量、FLOPs、内存占用、数值精度 |
| 03 | [Pruning I](lec03-pruning-I/README.md) | 结构化/非结构化剪枝、Magnitude、Saliency |
| 04 | [Pruning II](lec04-pruning-II/README.md) | Lottery Ticket、自动剪枝率、稀疏硬件 |
| 05 | [Quantization I](lec05-quantization-I/README.md) | 数值类型、线性量化、对称/非对称 |
| 06 | [Quantization II](lec06-quantization-II/README.md) | PTQ、QAT、STE、Binary/Ternary |
| 07 | [NAS I](lec07-nas-I/README.md) | 搜索空间、Chain/Cell-based、DARTS |
| 08 | [NAS II](lec08-nas-II/README.md) | Weight Sharing、One-Shot、ProxylessNAS |
| 09 | [Knowledge Distillation](lec09-distillation/README.md) | Soft Labels、Temperature、Feature蒸馏 |
| 10 | [MCUNet](lec10-mcunet/README.md) | TinyNAS、TinyEngine、MCU上的DL |
| 11 | [TinyEngine](lec11-tiny-engine/README.md) | 编译优化、算子融合、SIMD、内存规划 |
| 12 | [Transformer](lec12-transformer/README.md) | MHA、FFN、位置编码、KV Cache |
| 13 | [LLM Deployment](lec13-llm-deploy/README.md) | SmoothQuant、AWQ、KV Cache、Batching、投机解码 |
| 14 | [LLM Post-Training](lec14-llm-post-training/README.md) | SFT、RLHF、LoRA、QLoRA、Prompt Engineering |
| 15 | [Long Context](lec15-long-context/README.md) | StreamingLLM、DuoAttention、Mamba |
| 16 | [Vision Transformer](lec16-vit/README.md) | ViT、Swin、EfficientViT、SAM |

---

## References

- [MIT 6.5940 Course Page](https://efficientml.ai)
- [Song Han Lab](https://hanlab.mit.edu)
- [erectbranch/MIT-Efficient-AI](https://github.com/erectbranch/MIT-Efficient-AI) (Korean reference notes)
- [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295)
