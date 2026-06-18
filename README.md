# LDPJoinSketch

**ICDE 2024 论文**《Sketches-Based Join Size Estimation Under Local Differential Privacy》实验源代码。

> Meifan Zhang, Xin Liu, Lihua Yin.
> *Sketches-Based Join Size Estimation Under Local Differential Privacy.*
> IEEE ICDE 2024, pp. 1726–1738.
> [[IEEE Xplore]](https://ieeexplore.ieee.org/document/10598055)

---

## 研究背景

对敏感数据进行连接大小估计（join size estimation）存在隐私泄露风险。本地差分隐私（LDP）可以在数据收集阶段为用户提供隐私保护，但当连接属性值域较大时，直接对原始值施加 LDP 扰动会引入过大的噪声误差。采用概率数据结构（如 sketch）可以压缩大值域，但又会带来哈希碰撞误差。本文需要同时克服这两类误差。

## 方法

本文提出两个方法：

**LDPJoinSketch（基础方法）**

将 fast-AGMS sketch 改造为 LDP 版本。每个用户在本地对敏感连接属性值进行编码和扰动后发送给服务端；服务端聚合扰动值，分别为属性 A、B 构建 sketch $M_A$、$M_B$，再通过内积 $M_A \times M_B$ 估计连接大小。方法满足 LDP，估计误差有理论界。

**LDPJoinSketch+（增强方法）**

在 LDPJoinSketch 基础上引入两阶段频率感知扰动（FAP）机制，专门降低哈希碰撞误差：

- **第一阶段**：利用采样用户的数据构建初始 sketch，服务端从中计算频繁项集合（FI），并将 FI 下发给剩余用户。
- **第二阶段**：剩余用户按频繁/非频繁项分为两组，分别采用不同编码方式对私有值进行 FAP 扰动。服务端分别构建高频 sketch 和低频 sketch，合并后得到更精确的估计结果。

两种方法均满足 LDP，估计误差有理论界。

## 目录结构

```
LDPJoinSketch/
├── example/                  # 各方法的示例实现代码
├── data/                     # 实验数据集
├── LDPJoinSketch.pdf         # 论文 PDF
└── README.md
```

## 环境依赖

Python 3.8+，主要依赖：

```
numpy
```

安装：

```bash
pip install numpy
```

## 数据集

实验使用真实数据集与合成数据集：

- **合成数据集**：Zipf 分布，参数 α ∈ {1.1, 1.5, 2.0}，数据量 n = 1M
- **真实数据集**：Facebook、Twitter 等社交网络数据（可从 [SNAP](https://snap.stanford.edu/data/) 获取）

数据文件放置于 `data/` 目录下，每个文件为单列 CSV，每行一个整数属性值，无表头。

## 方法对比

| 方法 | 描述 | 核心挑战 |
|---|---|---|
| **k-RR** | 直接对原始值用 k-元随机响应扰动 | 大值域下噪声过大 |
| **FLH** | 快速局部哈希扰动 | 大值域下噪声过大 |
| **FAGMS** | 无隐私 AMS sketch（精度上界） | — |
| **LDPJoinSketch** | 本文提出，LDP 版 fast-AGMS sketch | 解决大值域噪声问题 |
| **LDPJoinSketch+** | 本文提出，两阶段频率感知扰动 | 同时降低噪声误差和哈希碰撞误差 |

## 评估指标

- **AE（绝对误差）**：|真实连接大小 − 估计值|
- **RE（相对误差）**：AE / 真实连接大小

## 引用

如果本代码对您的研究有帮助，请引用：

```bibtex
@inproceedings{zhang2024ldpjoinsketch,
  author    = {Meifan Zhang and Xin Liu and Lihua Yin},
  title     = {Sketches-Based Join Size Estimation Under Local Differential Privacy},
  booktitle = {Proceedings of the 40th IEEE International Conference on Data Engineering (ICDE)},
  pages     = {1726--1738},
  year      = {2024}
}
```

## 相关工作

本工作的 Shuffle DP 扩展版本见 [SDPJoinSketch](https://github.com/Triumphhh/SDPJoinSketch)，在相同中心隐私预算下通过 shuffle 隐私放大进一步提升了估计精度。
