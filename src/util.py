import pandas as pd
import numpy as np
from scipy.stats import entropy


def extract_ai_skills(skills_str):
    """
    判断岗位是否包含 Python + ML + Statistics
    """
    if pd.isna(skills_str):
        return 0
    skills_str = skills_str.lower()
    required = ["python", "ml", "statistics"]
    return int(all(skill in skills_str for skill in required))


def create_high_salary_label(df, percentile=0.75):
    """
    在 city 维度计算薪资分位数，生成高薪岗位 binary 标签
    """
    df = df.copy()
    df["high_salary"] = 0
    for city in df["city"].unique():
        threshold = df.loc[df["city"] == city, "salary_median_cny"].quantile(percentile)
        df.loc[(df["city"] == city) & (df["salary_median_cny"] > threshold), "high_salary"] = 1
    return df


def create_city_tier(df):
    """
    构造 city_tier 列
    使用 metro_class 或 city 名单划分 T1/T2/T3
    """
    df = df.copy()
    # 优先使用 metro_class
    if "metro_class" in df.columns:
        df["city_tier"] = df["metro_class"].map({1: "T1", 2: "T2", 3: "T3"})
    else:
        # 手动城市分级
        t1_cities = ["Beijing", "Shanghai", "Guangzhou", "Shenzhen"]
        df["city_tier"] = df["city"].apply(lambda x: "T1" if x in t1_cities else "T2")
    return df

def calculate_city_features(df, high_tech_industries):
    """
    计算每个城市的聚类特征：
    - 平均薪资
    - 博士占比
    - 高新行业占比
    - 技能多样性（Shannon指数）
    
    skill_keywords: 计算技能多样性用的技能列表
    high_tech_industries: 高新行业列表
    """
    df = df.copy()
    city_features = []

    for city, group in df.groupby("city"):
        N = len(group)
        avg_salary = group["salary_median_cny"].mean()
        phd_ratio = (group["education_requirement"] == "PhD").sum() / N
        hi_industry_ratio = group["industry"].isin(high_tech_industries).sum() / N

        # 将所有技能拆分并展平
        all_skills_list = []
        for skills in group["skills_required"].dropna():
            if isinstance(skills, str):
                skill_items = [s.strip() for s in skills.split(";") if s.strip()]
                all_skills_list.extend(skill_items)
        
        # 计算技能频率分布
        if all_skills_list:
            skill_series = pd.Series(all_skills_list)
            skill_counts = skill_series.value_counts()
            # 转换为概率分布
            skill_probs = skill_counts / skill_counts.sum()
            # 计算香农熵（使用自然对数）
            skill_entropy = -np.sum(skill_probs * np.log(skill_probs))
        else:
            skill_entropy = 0

        city_features.append({
            "city": city,
            "avg_salary": avg_salary,
            "phd_ratio": phd_ratio,
            "hi_industry_ratio": hi_industry_ratio,
            "skill_entropy": skill_entropy
        })

    city_df = pd.DataFrame(city_features)
    return city_df

def standardize_features(df, feature_cols):
    """
    对指定列进行标准化处理（z-score）
    """
    df_scaled = df.copy()
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            df_scaled[col] = 0
        else:
            df_scaled[col] = (df[col] - mean) / std
    return df_scaled

def list_unique_industries(df):
    """
    返回数据集中所有行业的唯一取值列表
    """
    return sorted(df['industry'].dropna().unique().tolist())