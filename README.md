# Data-Thinking

本项目基于 Kaggle 发布的中国就业市场招聘数据，对中国城市人才结构、薪资决定机制及高薪岗位特征进行了系统的数据分析与建模。  
项目结合统计建模与机器学习方法，完成解释性分析、预测性建模与城市画像分析.

---

## ⚙️ 环境配置

本项目使用 **Conda** 进行环境管理。

### 1️⃣ 创建并激活环境

```bash
conda env create -f environment.yml
conda activate china-job-analysis
```
### 若环境已存在，可使用：
```bash
conda env update -f environment.yml
```

---

## 🚀 使用方法

### 模型一：薪资决定机制分析
```bash
python src/model_1.py
```
输出：

薪资决定因素的重要性

模型拟合指标（R²、RMSE）
### 模型二：高薪岗位分类建模
```bash
python src/model_2.py
```
输出：

Logistic Regression 与 XGBoost 的分类性能指标

ROC-AUC、F1-score

SHAP 特征重要性解释图
### 模型三：城市人才吸引力聚类分析
#### Step 1：生成所有行业列表来计算shannon指数
```bash
python src/model_3_pre.py
```
#### Step 2：聚类与可视化
```bash
python src/model_3.py
```
输出：

城市聚类结果（CSV）

PCA 降维聚类图

各类城市画像特征均值表

---

## 📊 数据来源

本项目使用的数据来源于 Kaggle 平台发布的公开数据集：

China Job Market 2025 Dataset

Author: Kundan Bedmutha

URL: https://www.kaggle.com/datasets/kundanbedmutha/china-job-market-2025-dataset


## 📌 备注

本项目为课程《数据思维》期末研究项目.

代码结构与建模流程遵循可复现研究（Reproducible Research）原则,

欢迎交流与改进建议.