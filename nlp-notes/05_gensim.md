# 基于 Gensim 的词向量实战

> 用 Gensim 跑通 TF-IDF、LDA、Word2Vec 三条路线，所有输出均为实际运行结果。
>
> 完整代码见 [04_gensim.ipynb](https://github.com/datawhalechina/base-llm/blob/main/code/C2/04_gensim.ipynb)

## 一、Gensim 概述

Gensim（Generate Similar）是 Python 生态中主流的文本语义分析库，内置算法覆盖了从经典统计到神经网络词向量的主要方法：

| 类别 | 算法 | 对应 API |
|------|------|----------|
| 权重计算 | TF-IDF | `models.TfidfModel` |
| 主题模型 | LSA / LDA / NMF | `models.LsiModel` / `LdaModel` / `Nmf` |
| 神经网络词向量 | Word2Vec / FastText / Doc2Vec | `models.Word2Vec` / `FastText` / `Doc2Vec` |

安装：

```bash
pip install gensim
```

### 1.1 核心概念

- **语料库（Corpus）**：训练数据，分词后为 `list[list[str]]`，每个子列表是一篇文档。
- **词典（Dictionary）**：词→整数 ID 的映射表，根据语料构建。
- **向量（Vector）**：文档的数学表示。词袋模型下，`["我","爱","我"]` → `[(0,2),(1,1)]`。
- **稀疏向量**：仅记录非零项 `(index, value)`，跳过全部 0 值，节省存储。
- **模型（Model）**：向量转换算法，如 `TfidfModel` 将词频向量转为 TF-IDF 权重向量。

## 二、标准工作流：文本 → 词袋向量

TF-IDF、LDA 等基于词袋的模型共用一套预处理流程：

```
原始文本 ──分词──▶ list[list[str]] ──建词典──▶ Dictionary ──词袋化──▶ BoW 语料库
```

> **注意**：Word2Vec / FastText / Doc2Vec 直接接收分词后的句子列表，不需要词袋化。

```python
import jieba
from gensim import corpora

# Step 1: 分词
raw_headlines = [
    "央行降息，刺激股市反弹",
    "球队赢得总决赛冠军，球员表现出色"
]
tokenized_headlines = [jieba.lcut(doc) for doc in raw_headlines]

# Step 2: 建词典
dictionary = corpora.Dictionary(tokenized_headlines)

# Step 3: 词袋化
corpus_bow = [dictionary.doc2bow(doc) for doc in tokenized_headlines]
```

**输出：**

```
分词结果:
  [['央行', '降息', '，', '刺激', '股市', '反弹'],
   ['球队', '赢得', '总决赛', '冠军', '，', '球员', '表现出色']]

词典映射:
  {'刺激': 0, '反弹': 1, '央行': 2, '股市': 3, '降息': 4,
   '，': 5, '冠军': 6, '总决赛': 7, '球员': 8, '球队': 9, '表现出色': 10, '赢得': 11}

BoW 语料:
  [[(0,1), (1,1), (2,1), (3,1), (4,1), (5,1)],
   [(5,1), (6,1), (7,1), (8,1), (9,1), (10,1), (11,1)]]
```

每个元组 `(token_id, frequency)` 表示该词在本文档中的出现次数。这就是后续模型的标准输入格式。

## 三、TF-IDF：衡量词的区分度

词袋只记录词频，无法区分"的"和"降息"的重要性。TF-IDF 的逻辑是：一个词在本文档中频次高（TF↑），同时在其他文档中很少出现（IDF↑），则权重大。

```python
from gensim import models

# 语料：6 条新闻标题（财经 + 体育混合）
headlines = [
    "央行降息，刺激股市反弹",
    "球队赢得总决赛冠军，球员表现出色",
    "国家队公布最新一期足球集训名单",
    "A股市场持续震荡，投资者需谨慎",
    "篮球巨星刷新历史得分记录",
    "理财产品收益率创下新高"
]
tokenized_headlines = [jieba.lcut(title) for title in headlines]

dictionary = corpora.Dictionary(tokenized_headlines)
corpus_bow = [dictionary.doc2bow(doc) for doc in tokenized_headlines]

# 训练 TF-IDF 模型
tfidf_model = models.TfidfModel(corpus_bow)
corpus_tfidf = tfidf_model[corpus_bow]
```

**第一篇标题的 TF-IDF 向量（带词）：**

```
[('刺激', 0.4407), ('反弹', 0.4407), ('央行', 0.4407),
 ('股市', 0.4407), ('降息', 0.4407), ('，',   0.1705)]
```

"降息""股市"等财经专属词权重 0.44，跨文档的标点"，"仅 0.17——TF-IDF 自动压低了通用词。

**对新文本打分：**

```python
new_bow = dictionary.doc2bow(jieba.lcut("股市大涨，牛市来了"))
print(tfidf_model[new_bow])
# → [(3, 0.9326), (5, 0.3608)]
```

仅"股市"和"，"命中词典，其余新词为 OOV（Out-of-Vocabulary）被忽略。实际使用时需确保语料词汇覆盖面足够。

## 四、LDA：无监督主题发现

LDA（Latent Dirichlet Allocation）不需要标签，只需指定主题数 K，即可从语料中归纳出每个主题的关键词分布，并为每篇文档输出主题概率。

```python
lda_model = models.LdaModel(
    corpus=corpus_bow,
    id2word=dictionary,
    num_topics=2,
    random_state=100
)

for topic in lda_model.print_topics():
    print(topic)
```

**模型归纳的 2 个主题：**

```
Topic 0: "公布" + "一期" + "名单" + "足球" + "集训" + "国家队" + "A股" + "市场" ...
Topic 1: "篮球" + "刷新" + "历史" + "记录" + "得分" + "巨星" + "降息" + "反弹" ...
```

**推断新文档主题：**

```python
new_bow = dictionary.doc2bow(jieba.lcut("巨星詹姆斯获得常规赛MVP"))
print(lda_model[new_bow])
```

```
Topic 0: 0.2724
Topic 1: 0.7276   ← 73% 偏向主题 1（含体育关键词）
```

> 若新文档在词典中几乎无重叠词，`doc2bow` 返回空列表，主题分布退化为接近均匀（如 0.5/0.5），此时无参考价值。

## 五、Word2Vec：学习词的稠密语义向量

Word2Vec 的目标与前两者不同——它为**每个词**训练一个低维稠密向量，使语义相近的词在向量空间中距离相近。训练完成后神经网络本身被丢弃，只保留词向量查询表 `model.wv`。

Word2Vec 直接接收分词后的句子列表，**不需要建词典和词袋化**。

### 5.1 训练

```python
from gensim.models import Word2Vec

# 16 条新闻标题（财经 8 条 + 体育 8 条）
tokenized = [jieba.lcut(title) for title in headlines_16]

model = Word2Vec(
    tokenized,
    vector_size=50,    # 向量维度
    window=3,          # 上下文窗口
    min_count=1,       # 最低词频阈值
    sg=1               # 1=Skip-gram, 0=CBOW
)
```

### 5.2 核心参数

| 参数 | 作用 | 典型值 |
|------|------|--------|
| `vector_size` | 词向量维度。越高表达力越强，计算开销也越大 | 50–300 |
| `window` | 上下文窗口大小（中心词前后各取几个词） | 3–10 |
| `min_count` | 词频低于此值的词直接丢弃（过滤噪音和罕见词） | 5–10 |
| `sg` | 训练算法。0 = CBOW（上下文→中心词）；1 = Skip-gram（中心词→上下文） | 小语料推荐 1 |
| `hs` | 输出层优化。0 = 负采样；1 = 层次 Softmax | 0 |
| `negative` | 负采样个数（`hs=0` 时生效） | 5–20 |
| `sample` | 高频词下采样阈值，降低"的""是"等词的过度影响 | 1e-3–1e-5 |

### 5.3 使用词向量

```python
# 最相似词
model.wv.most_similar('股市', topn=5)
```

```
足球: 0.2796
吸引: 0.2658
冠军: 0.2289
引援: 0.2250
动荡: 0.1866
```

```python
# 余弦相似度
model.wv.similarity('球队', '球员')  # → 0.2203

# 获取向量
vec = model.wv['市场']  # shape: (50,)
# [0.0104, -0.0122, -0.0055, -0.0014, -0.0001, -0.0182, 0.0018, -0.0138, ...]
```

> 小语料（16 条标题）下相似度偏低且不稳定是正常现象。Word2Vec 效果高度依赖数据规模，通常需要百万级句子才能产出高质量词向量。

### 5.4 模型持久化

如果不再做增量训练，只保存 `model.wv`（`KeyedVectors`）即可。完整模型包含哈夫曼树、梯度累积量等训练中间状态，体积更大、加载更慢。

```python
from gensim.models import KeyedVectors

# 保存
model.wv.save("news_vectors.kv")

# 加载（不需要原始模型对象）
loaded_wv = KeyedVectors.load("news_vectors.kv")
loaded_wv.similarity('球队', '球员')  # → 0.2203
```

## 小结

| 模型 | 输入格式 | 输出 | 适用场景 |
|------|----------|------|----------|
| TF-IDF | BoW 语料 | 词权重向量 | 关键词提取、文档相似度、特征工程 |
| LDA | BoW 语料 | 主题概率分布 | 主题发现、文档聚类 |
| Word2Vec | 分词句子列表 | 词级稠密向量 | 语义相似度、下游 NLP 任务的预训练表示 |

三者共同点：都是将文本映射到数值向量空间。区别在于粒度（文档级 vs 词级）和方法论（统计频次 vs 矩阵分解 vs 神经网络）。
