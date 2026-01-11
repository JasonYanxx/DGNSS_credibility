# 设计文档：面向低成本 DGNSS 调试期的可信度诊断协议

| 属性 | 内容 |
| :--- | :--- |
| **作者** | AI Assistant (基于 Research Plan) |
| **状态** | 已实现 (Implemented) |
| **最后更新** | 2026-01-10 |
| **代码路径** | `/Users/jacobianyan/Documents/TmpFolder/DGNSS credibility code/` |

---

## 1. 背景与动机 (Context & Motivation)

在低成本 DGNSS 监测网络（如滑坡、大坝监测）的**调试期（Commissioning Phase）**，运维人员面临一个核心痛点：无法区分观测数据的异常波动是由**环境多径（Multipath）** 引起的，还是由**系统故障（System Fault/Bias）** 引起的。

*   **环境多径**：表现为高强度的有色噪声（Colored Noise），在 DGNSS 中通常可通过长时间平均消除，属于“健康”状态。
*   **系统故障**：如基准站坐标错误、天线相位中心偏差或严重的大气模型失配，表现为刚性的系统偏差（SMM），无法通过平均消除，必须人工干预。

现有的单指标诊断方法（如 NEES/Chi-square）在面对强多径时往往会误报，因为有色噪声破坏了统计假设。

## 2. 目标 (Goals)

开发一套自动化的“分诊协议（Triage Protocol）”，利用 **15分钟块平均（Block Averaging）** 实现时域解耦，结合 **TIM 诊断框架**，准确区分以下场景：

1.  **Scenario A (Multipath/Healthy)**: 能够识别出多径效应已被块平均“白化”，诊断为 **Calibrated (Pass)**。
2.  **Scenario B (System Fault)**: 能够穿透噪声识别出刚性偏差，诊断为 **Bias Detected (Fail)**。
3.  **Scenario C (Mixed)**: 在多径和偏差共存时，仍能识别出偏差，诊断为 **Bias Detected (Fail)**。
4.  **Scenario D (Optimism)**: 识别出随机模型过于自信（协方差低估），诊断为 **Optimistic (Fail)**。

## 3. 系统架构 (System Architecture)

系统由四个主要阶段组成的流水线构成：

```mermaid
graph LR
    A[Stage 1: Geometry Engine] --> B[Stage 2: Error Injection]
    B --> C[Stage 3: DGNSS Solver]
    C --> D[Stage 4: Diagnostic System]
    D --> E[Visualization & Reporting]
```

### 3.1 模块职责

*   **Geometry Engine (`geometry_engine.py`)**: 生成真实的卫星几何分布（星历、视线向量），提供 Ground Truth。
*   **Error Injection (`error_injection.py`)**: 根据预设场景（A/B/C/D）注入物理意义明确的误差（Gauss-Markov 多径、常数 Bias 等）。
*   **DGNSS Solver (`dgnss_solver.py`)**: 执行双差加权最小二乘解算，输出 1Hz 的原始残差（Residuals）和协方差矩阵（Covariance）。
*   **Diagnostic System (`diagnostic_system.py`)**: 核心逻辑层。执行块平均和 TIM 框架诊断。
*   **Core Algorithm (`lib/algorithm_comparison.py`)**: 具体的数学实现，包括 NCI、NLL、ES 计算及 ELT 检验。

---

## 4. 详细设计 (Detailed Design)

### 4.1 误差注入模型 (Error Injection Model)

为了验证协议，我们在 `error_injection.py` 中实现了四种场景的物理模型：

*   **多径模型 (Scenario A/C)**: 采用混合模型模拟静态多径。
    *   **Specular（镜面反射）**: $A \cdot \sin(2\pi f t)$，其中 $A \approx 0.1\sim0.2m, f \approx 0.001\sim 0.002 Hz$。
    *   **Diffuse（漫反射）**: 一阶高斯-马尔可夫过程 (AR(1))，$\eta_k = \phi \eta_{k-1} + w_k$，其中 $\phi=0.99$（强相关）。
*   **系统偏差 (Scenario B/C)**: 注入常数偏差 $b = 0.5m$（模拟坐标系误差）。
*   **模型乐观 (Scenario D)**: 注入白噪声，但将求解器的输入协方差缩放为 $0.5 \times \Sigma_{true}$。

### 4.2 诊断核心逻辑 (`diagnostic_system.py`)

这是本项目的核心创新点，严格遵循 `research plan.md` 的 **"Block Averaging -> TIM Diagnosis"** 路径。

#### 步骤 1: 15-Minute Block Averaging (时域解耦)
*   **输入**: 1Hz 原始残差序列 $e_{1:N}$，原始协方差 $\Sigma_{1:N}$。
*   **操作**:
    *   将数据划分为长度为 $M$ 的块（$M = 15 \text{min} \times 60 \text{sec} = 900$）。
    *   计算块均值：$\bar{y}_k = \frac{1}{M} \sum_{i \in \text{block}_k} e_i$。
    *   计算块协方差（假设白化）：$R_k = \frac{1}{M} \bar{\Sigma}_k$。
*   **物理依据**: 静态多径的去相关周期通常小于 15 分钟，块平均能有效抑制多径并恢复统计独立性（白化效应）。

#### 步骤 2: TIM 框架诊断 (基于 `lib/algorithm_comparison.py`)
利用块均值 $\bar{y}_k$ 和缩放后的协方差 $R_k$ 作为输入，运行 `NCI_NLL_ES_Algorithm`：

1.  **ELT (Energy Location Test / Bias Check)**:
    *   检测是否存在系统偏差（SMM）。
    *   方法：基于能量距离（Energy Distance）的非参数检验，通过符号翻转（Sign-flip randomization）计算 p-value。
    *   判据：若 $p < \alpha$，判定为 **Bias**。

2.  **Directional Probing (NCI/NLL/ES)**:
    *   若通过 ELT（无 Bias），则检查随机模型的尺度一致性。
    *   **NCI (Noncredibility Index)**: 衡量整体非可信度。
    *   **Directional Probing**: 计算 NLL 和 ES 对协方差缩放的敏感度（Slope Relative Difference, SRD）。
    *   判据：
        *   若 NCI > 0.5 (dB)，判定为 **Optimistic**。
        *   若 NCI < -0.5 (dB)，判定为 **Pessimistic**。
        *   否则，判定为 **Calibrated**。

### 4.3 求解器 (`dgnss_solver.py`)

实现标准的双差（Double-Difference）加权最小二乘算法：
*   **观测值**: 伪距（Pseudorange）。
*   ==**权重模型**: 支持高度角定权 (`sin^2(el)`)==。
*   **输出**: 每个历元的定位误差向量 $(dx, dy, dz)$ 和对应的后验协方差矩阵 $Q_{xyz}$。

---

## 5. 接口设计 (Interfaces)

### 5.1 主程序入口 (`main.py`)

```python
def run_commissioning_verification(
    baseline_km=1.0, 
    duration_hours=2.0, 
    block_window_s=900.0
) -> dict
```
*   **功能**: 自动运行 A/B/C/D 四个场景，生成对比数据。
*   **参数**:
    *   `baseline_km`: 基线长度（调试期通常较短，设为 1km）。
    *   `duration_hours`: 模拟时长（需足以生成多个 Block）。
    *   `block_window_s`: 块大小（默认为 900秒 = 15分钟）。

### 5.2 诊断系统 (`diagnostic_system.py`)

```python
class DiagnosticSystem:
    def run(self, error_vecs: List[np.ndarray], cov_matrices: List[np.ndarray]) -> dict
```
*   **输入**: 原始 1Hz 的误差序列和协方差序列。
*   **输出**: 包含分类结果（String）、统计量（NCI, p-value）和块均值序列的字典。

---

## 6. 数据验证与可视化 (Validation & Visualization)

为了证明方法的有效性，系统生成以下关键图表（对应论文图表）：

1.  **白化效应验证 (Whitening Effect / ACF Plot)**:
    *   **内容**: 对比 Scenario A 中“原始 1Hz 残差”与“15-min 块均值”的自相关函数 (ACF)。
    *   **预期**: 原始数据 ACF 拖尾长（有色），块均值 ACF 在 Lag > 0 处迅速归零（白化）。

2.  **分诊逻辑演示 (Triage Summary)**:
    *   **内容**: 表格化展示四个场景在“Raw 1Hz”和“15-min Block”下的诊断结果变化。
    *   **关键转变**:
        *   Scenario A (Multipath): Raw (Optimistic/Bias) $\to$ Block (**Calibrated**).
        *   Scenario B (Fault): Raw (Bias) $\to$ Block (**Bias**).

3.  **指标追踪 (Metric Tracking)**:
    *   展示 Bias Estimate (ELT结果) 和 NCI 指标在不同场景下的表现。

---

## 7. 局限性与讨论 (Limitations)

1.  **Block 数量依赖**: 诊断的统计显著性依赖于 Block 的数量。若总时长过短（如仅 1 个 Block），ELT 和 NCI 的计算可能不稳定或退化。建议最少时长为 2 小时（8 个 Blocks）。
2.  **偏差检出下限**: 对于极微小的系统偏差（小于热噪声水平），块平均后的信噪比提升可能仍不足以通过 ELT 检出。
3.  **多径去相关假设**: 假设环境多径在 15 分钟内是均值为零的波动。若存在极低频的多径（如周期 > 1小时），该方法可能会将其误判为 Bias。