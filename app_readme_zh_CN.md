# PFAS CCS Screening App 中文说明

该应用读取 Excel 或 CSV 特征表，完成质量校正、CCS 校正、第一同位素峰比值计算以及 PFAS 分级预测。

## 第一同位素峰比值

应用支持三种输入方式：

- 使用已有的 `first_isotopic_peak_ratio`；旧字段 `riMp1` 仍可作为输入别名。
- 使用单同位素峰和第一同位素峰强度计算：

```text
first_isotopic_peak_ratio =
    100 * I(第一同位素峰) / I(单同位素峰)
```

- 根据 m/z、保留时间、`3D_TC` 和强度自动建立同位素包络。CCS 不参与配对。

自动配对假定离子为单电荷，参数为：

```text
名义质量偏移：+1、+2、+3、+4 Da
质量窗口：+/-0.020 Da
保留时间容差：+/-0.020 min
3D_TC 容差：+/-0.060
```

每个峰只能属于一个同位素包络，已经分配为 M+n 的峰不能再作为新的 M。候选冲突依次按照保留时间误差、`3D_TC` 误差和名义质量误差处理。

### M+1 至 M+4 实验同位素分布

自动配对模式提取 M+1、M+2、M+3 和 M+4 名义同位素簇。各比值均相对于单同位素峰 M：

```text
rMp1 = 100 * sum(I(M+1 候选峰)) / I(M)
rMp2 = 100 * sum(I(M+2 候选峰)) / I(M)
rMp3 = 100 * sum(I(M+3 候选峰)) / I(M)
rMp4 = 100 * sum(I(M+4 候选峰)) / I(M)
```

同一名义同位素簇内满足条件的精细同位素候选峰强度会被求和，同时输出候选数、m/z 范围、质量偏移范围、RT 误差范围、`3D_TC` 误差范围和歧义标记。

当前 RF-MC 模型使用完整 M+1 簇比值 `rMp1`。另外输出 `first_13C_peak_ratio`，它是在已分配 M+1 簇中选择最接近 `+1.00335483507 Da` 的单峰，仅用于审计，不作为模型输入。M+2 至 M+4 用于初步同位素模式或元素组成类别判断，不能单独作为结构鉴定结果。

## 强度筛选与PFAS数据库匹配

应用使用 `Minimum M intensity` 和 `Top N intense PFAS peaks` 筛选浓度较高的候选峰。只有 `isotope_role=M` 且 `Level >=1` 的峰参与筛选，并按照 M 峰强度和PFAS概率排序。

默认数据库为 `Chemical List PFAS.xlsx`，也可以选择具有名称、分子式、单同位素质量和SMILES字段的其他Excel或CSV数据库。默认质量容差为 `+/-5 ppm`。

数据库匹配同时考虑：

```text
POS：观测 m/z 匹配 [M+H]+ 或 M+
NEG：观测 m/z 匹配 [M-H]- 或 M-
质子精确质量：1.007276466621 Da
```

每个候选会输出名称、DTXSID、CASRN、分子式、中性单同位素质量、SMILES、离子解释和质量误差。应用根据分子式计算理论 `rMp1` 至 `rMp4`，并与实验值并列比较。

SMILES表示数据库中的可能结构。同一质量或分子式可能对应多个结构，因此应用保留容差内的全部候选，不自动指定唯一结构。

## CCS 校正

第一阶段可以使用标准物的校正到达时间、m/z 和参考 CCS 建立 cIMS 校正，也可以直接使用已有实验 CCS。

固定二级转换为：

- POS：`CCS_model = 0.8289 * CCS_stage1 + 38.258`
- NEG：`CCS_model = 0.9585 * CCS_stage1 + 9.3196`

这些公式用于 AllCCS2 预测值。对于实验校正 CCS，建议将二级转换设为 `None`。

## 输出

- `Labeled data`：原始字段、质量和 CCS 校正结果、第一同位素峰配对信息、预测概率及分级标签。
- `Run summary`：运行参数、校正公式、拟合结果和各 Level 数量。
- `Calibration curve`：使用 cIMS 标准物校正时生成。
- `Prioritized PFAS peaks`：达到强度阈值并位于Top N内的Level >=1单同位素M峰。
- `PFAS database matches`：全部质量候选、离子解释、分子式、SMILES及实验/理论同位素分布。

模型内部仍将该比值映射到训练时使用的历史特征名 `riMp1`，但新版对外字段统一使用 `first_isotopic_peak_ratio`。

## 运行

```powershell
python launch_pfas_app.py
```

便携版位于：

```text
dist\PFAS_CCS_Screening\PFAS_CCS_Screening.exe
```
