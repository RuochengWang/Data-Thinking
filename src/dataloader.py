# src/data_loader.py
import pandas as pd
import numpy as np
import re
import os

def load_and_preprocess_data(file_path: str = "../data/china_job_market_2025.csv"):
    """
    数据清洗
    """
    # 1. 加载数据(自动跳过末尾的编码行)
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"✅ 原始数据加载: {len(df)} 行")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        print(f"✅ 原始数据加载 (UTF-8 BOM): {len(df)} 行")
    
    # 2. 清理：移除 header 行（如 'year,month,job_title,...' 出现在数据中）
    # 检查第一列是否为字符串"year"
    df = df[~df['year'].astype(str).str.contains('year', na=False)]
    
    # 3. 转换关键列类型
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['salary_median_cny'] = pd.to_numeric(df['salary_median_cny'], errors='coerce')
    df['salary_min_cny'] = pd.to_numeric(df['salary_min_cny'], errors='coerce')
    df['salary_max_cny'] = pd.to_numeric(df['salary_max_cny'], errors='coerce')
    
    # 4. 清洗无效薪资数据
    df = df.dropna(subset=['salary_median_cny'])
    df = df[
        (df['salary_median_cny'] >= 5000) & 
        (df['salary_median_cny'] <= 200000)
    ]
    
    # 5. 构造核心变量
    df = df.copy()
    
    # 薪资中位数
    df["salary_mid"] = df["salary_median_cny"]
    df["log_salary"] = np.log(df["salary_mid"] + 1)  # +1 避免 log(0)
    
    # 经验年数映射(根据 experience_level)
    exp_map = {
        "Entry": 0.5,   # 应届/初级
        "Mid": 2.5,     # 中级
        "Senior": 6.0   # 高级
    }
    df["exp_years"] = df["experience_level"].map(exp_map)
    
    # 教育水平标准化
    edu_mapping = {
        "Bachelor's": 3.0,
        "Master's": 4.0,
        "PhD": 5.0,
        "Ausbildung/Diploma": 2.5,   # 德国双元制 ≈ 大专
        "Hauptschule/Realschule": 2.0  # 德国中学 ≈ 高中
    }
    df["edu_level"] = df["education_requirement"].map(edu_mapping)
    
    # 城市等级映射(仅用 city 字段)
    tier1 = ["Beijing", "Shanghai", "Shenzhen", "Guangzhou"]
    new_tier1 = ["Chengdu", "Hangzhou", "Chongqing", "Suzhou", "Wuhan", "Xi'an", 
                "Nanjing", "Tianjin", "Zhengzhou", "Changsha", "Dongguan", "Ningbo", 
                "Foshan", "Hefei", "Qingdao"]
    
    def get_city_tier(city):
        if pd.isna(city):
            return "未知"
        city_clean = str(city).strip()
        if city_clean in tier1:
            return "一线"
        elif city_clean in new_tier1:
            return "新一线"
        else:
            return "二线及以下"
    
    df["city_tier"] = df["city"].apply(get_city_tier)
    
    # 6. 技能特征工程（关键修正：解析 skills_required 的分号/逗号结构）
    def extract_skills(skills_str):
        """解析 'AWS,Docker,Kubernetes;Python,ML,Statistics' 等嵌套结构"""
        if pd.isna(skills_str) or not isinstance(skills_str, str):
            return []
        skills_list = []
        # 按分号分割技能组
        for group in skills_str.split(";"):
            # 每组内按逗号分割具体技能
            skills = [s.strip().lower() for s in group.split(",")]
            skills_list.extend(skills)
        return skills_list
    
    df["skill_list"] = df["skills_required"].apply(extract_skills)
    
    # 关键技能检测（大小写不敏感 + 关键词匹配）
    def has_skill(skill_list, keywords):
        skill_str = " ".join(skill_list)
        return 1 if any(kw in skill_str for kw in keywords) else 0
    
    # 定义技能关键词（根据数据高频词）
    df["has_python"] = df["skill_list"].apply(
        lambda x: has_skill(x, ["python"])
    )
    df["has_ml"] = df["skill_list"].apply(
        lambda x: has_skill(x, ["ml", "machine learning", "statistics"])
    )
    df["has_cloud"] = df["skill_list"].apply(
        lambda x: has_skill(x, ["aws", "docker", "kubernetes", "cloud"])
    )
    df["has_sql"] = df["skill_list"].apply(
        lambda x: has_skill(x, ["sql", "mysql", "postgresql"])
    )
    df["skill_count"] = df["skill_list"].apply(len)
    
    # 7. 行业与职位分类（简化）
    ai_keywords = ["data scientist", "ml", "ai", "machine learning", "algorithm"]
    df["is_ai"] = df["job_title"].str.lower().str.contains(
        "|".join(ai_keywords), na=False
    ).astype(int)
    
    # 8. 高薪定义：按城市分位数（90%）
    city_p90 = df.groupby("city")["salary_mid"].quantile(0.9).to_dict()
    df["city_p90"] = df["city"].map(city_p90)
    df["is_high_salary"] = (df["salary_mid"] > df["city_p90"]).astype(int)
    
    # 9. 最终清洗
    df = df.dropna(subset=["exp_years", "edu_level", "city"])
    df = df[df["city"] != "year"]  # 防御性过滤
    
    print(f"✅ 清洗后有效样本: {len(df)} 条")
    print(f"  - 城市分布: {df['city_tier'].value_counts().to_dict()}")
    print(f"  - Python技能: {df['has_python'].sum()} 个职位 ({df['has_python'].mean():.1%})")
    print(f"  - ML/Statistics: {df['has_ml'].sum()} 个职位 ({df['has_ml'].mean():.1%})")
    
    return df

if __name__ == "__main__":
    # 测试
    df = load_and_preprocess_data()
    print("\n前3行示例:")
    print(df[[
        "city", "job_title", "salary_mid", "exp_years", 
        "has_python", "has_ml", "is_high_salary"
    ]].head(3).to_string(index=False))