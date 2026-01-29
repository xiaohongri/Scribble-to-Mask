# Point Sampling Methodology for Scribble-to-Mask

## 概述 (Overview)

这个文档详细解释了Scribble-to-Mask项目中如何从线条（scribbles）采样提示点。
This document explains in detail how prompt points are sampled from lines/scribbles in the Scribble-to-Mask project.

## 问题：线条是怎么采样提示点的呢？

在训练和推理过程中，Scribble-to-Mask使用多种策略从分割错误区域生成训练用的scribble（线条标注）。主要有以下几种采样方法：

## 1. 基于机器人的自动采样 (Robot-based Automated Sampling)

**文件**: `dataset/tamed_robot.py`

这是最核心的点采样算法，改编自DAVIS交互式分割的robot。采样流程如下：

### Step 1: 骨架提取 (Skeleton Extraction)
```python
_generate_scribble_mask(mask)
```
- **输入**: 错误区域的二值mask
- **处理**: 
  1. 根据区域面积计算自适应的形态学核大小
  2. 使用形态学操作（erosion + dilation）平滑区域
  3. 计算中轴线（medial axis）得到骨架
- **输出**: 骨架mask，其中的像素点是初始的候选采样点
- **作用**: 将不规则的错误区域简化为一条或多条骨架线，这些骨架点代表了区域的中心线特征

### Step 2: 图构建 (Graph Construction)
```python
_mask2graph(skeleton_mask)
```
- **输入**: 骨架mask
- **处理**:
  1. 提取所有骨架像素的(x,y)坐标作为节点
  2. 使用半径邻居图连接距离≤√2的像素（支持8连通）
  3. 边的权重为像素间的欧式距离
  4. 转换为NetworkX图结构
- **输出**: 图G和节点坐标数组P
- **作用**: 将离散的骨架像素组织成连通图结构，为路径查找做准备

### Step 3: 路径选择 (Path Selection)
```python
_acyclics_subgraphs(G)  # 去除环
_longest_path_in_tree(G)  # 找最长路径
```
- **处理**:
  1. 将图分解为连通分量
  2. 对每个分量，移除环中权重最大的边，形成树状结构
  3. 使用两次最短路径遍历找到每棵树的最长路径
     - 从任意节点v出发，找最远点v'
     - 从v'出发，找到的最远路径即为直径（最长路径）
- **输出**: 每个连通分量的最长路径节点索引
- **作用**: 选择最能代表错误区域形状的路径，这些路径上的点将作为Bezier曲线的控制点

### Step 4: Bezier曲线采样 (Bezier Curve Sampling)
```python
bezier_curve(points, nb_points=1000)
```
- **输入**: 最长路径上的控制点
- **处理**:
  1. 使用Bernstein多项式拟合Bezier曲线
  2. 在参数t ∈ [0,1]上均匀采样nb_points个点（默认1000个）
  3. 每个点计算为: P(t) = Σ C(n,i) * t^(n-i) * (1-t)^i * P_i
- **输出**: 1000个平滑的曲线采样点
- **作用**: 将离散的控制点转换为密集、平滑的点集，形成自然的线条

### Step 5: 线条渲染 (Line Rendering)
```python
bresenham(points)
```
- **输入**: Bezier曲线采样的浮点坐标点
- **处理**:
  1. 对连续点对应用Bresenham直线算法
  2. 插值出所有中间像素，保证8连通性
  3. 将浮点坐标转换为整数像素坐标
- **输出**: 连续的像素坐标数组
- **作用**: 将采样点转换为可绘制的连续像素线条，无间隙

### 完整流程示意
```
错误区域 (H×W) 
    ↓ 2×下采样
    ↓ 骨架提取 → 候选点集
    ↓ 图构建 → 连通图
    ↓ 路径查找 → 控制点 (3-10个)
    ↓ Bezier采样 → 密集点 (1000个)
    ↓ Bresenham → 连续像素线
    ↓ 2×上采样
Scribble Mask (H×W)
```

## 2. 曲线scribble采样 (Curve Scribble Sampling)

**文件**: `dataset/gen_scribble.py` - `get_curve_scribble()`

### 采样策略
- 从错误区域随机采样3个控制点
- 拟合二次Bezier曲线
- 在曲线上均匀采样**1024个点**
- 用OpenCV的polylines绘制连续线条（thickness=3）

### 关键参数
```python
eval_pts = np.linspace(0.0, 1.0, 1024)  # 1024个均匀参数点
```

### 特点
- 每个区域生成2-4条曲线
- 选择最长的曲线保留（更明显）
- 允许曲线略微超出区域边界（误差容忍）

## 3. 边界scribble采样 (Boundary Scribble Sampling)

**文件**: `dataset/gen_scribble.py` - `get_boundary_scribble()`

### 采样策略
- 腐蚀错误区域（随机核大小3-50）
- 计算形态学梯度，得到边界
- 在边界上采样点

### 特点
- 沿着对象边缘采样
- 适合边界清晰的区域

## 4. 细化scribble采样 (Thinned Scribble Sampling)

**文件**: `dataset/gen_scribble.py` - `get_thinned_scribble()`

### 采样策略
- 使用Zhang-Suen细化算法骨架化区域
- 在骨架上采样点
- 扩张骨架（3×3核）

### 特点
- 沿着区域的中轴线采样
- 类似于机器人方法，但更简单

## 5. 训练时的采样混合策略

在训练数据生成时（`get_scribble()`函数）：
- **75%概率**: 使用TamedRobot的图采样方法
- **25%概率**: 随机组合边界/曲线/细化方法
  - 每个区域随机选择1-2种方法
  - 33%概率选边界scribble
  - 33%概率选细化scribble  
  - 33%概率选曲线scribble

这种多样化的采样策略使模型能够适应不同风格的人工标注。

## 6. 交互式GUI中的采样

**文件**: `interactive.py`

在交互式工具中，用户手绘的scribble采样方式不同：
- 鼠标移动时，每个点到上一个点之间用`cv2.line()`连接
- thickness=3，确保线条可见
- 不需要复杂的采样算法，直接记录鼠标轨迹

```python
# 点采样：鼠标按下
cv2.circle(self.p_srb, (ex, ey), radius=3, color=(1), thickness=-1)

# 线段采样：鼠标移动
cv2.line(self.p_srb, (last_ex, last_ey), (ex, ey), (1), thickness=3)
```

## 总结：关键采样参数

| 方法 | 采样点数 | 参数化方式 | 用途 |
|-----|---------|----------|-----|
| TamedRobot | 1000/curve | Bezier参数t∈[0,1] | 自动训练数据生成 |
| Curve Scribble | 1024/curve | Bezier参数t∈[0,1] | 训练数据增强 |
| Boundary | 可变 | 边界梯度 | 训练数据增强 |
| Thinned | 可变 | 骨架像素 | 训练数据增强 |
| Interactive | 鼠标轨迹 | 直接坐标 | 人工交互 |

## 为什么这样采样？

1. **密集采样（1000-1024点）**: 确保曲线平滑，没有间隙
2. **Bezier曲线**: 产生自然、平滑的人类标注风格
3. **最长路径选择**: 选择最显著的特征，避免琐碎的小线段
4. **多策略组合**: 增加训练数据的多样性，提高泛化能力
5. **图算法**: 处理复杂拓扑结构，避免重复路径

## 代码示例

完整的点采样流程：
```python
# 1. 创建机器人
robot = TamedRobot(nb_points=1000)

# 2. 对错误区域采样
scribble_mask = robot.interact(error_mask)
# 内部流程：
#   skeleton -> graph -> longest_paths -> bezier(1000 points) -> bresenham -> mask

# 3. 或者使用曲线采样
curve_scribble = get_curve_scribble(region)
# 内部流程：
#   sample 3 points -> bezier curve -> evaluate at 1024 points -> polylines
```

## 相关文件

- `dataset/tamed_robot.py`: 核心图采样算法
- `dataset/gen_scribble.py`: 多种scribble生成方法
- `interactive.py`: 交互式GUI的直接采样
- `dataset/static_dataset.py`: 静态数据集的scribble生成调用
- `dataset/lvis_dataset.py`: LVIS数据集的scribble生成调用
