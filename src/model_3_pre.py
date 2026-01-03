import pandas as pd
from util import list_unique_industries, list_unique_skills

output_dir = "output"

# 1. 读取原始数据
df = pd.read_csv("./data/china_job_market_2025.csv")  # 根据你的路径修改

# 2. 获取所有行业
all_industries = list_unique_industries(df)
industries_df = pd.DataFrame({"industry": all_industries})
industries_df.to_csv(f"{output_dir}/all_industries.csv", index=False)
print(f"所有行业已导出到 {output_dir}/all_industries.csv")
print(industries_df)

# 3. 获取所有技能
all_skills = list_unique_skills(df)
skills_df = pd.DataFrame({"skill": all_skills})
skills_df.to_csv(f"{output_dir}/all_skills.csv", index=False)
print(f"所有技能已导出到 {output_dir}/all_skills.csv")
print(skills_df)

# 4. 提示用户进行人工筛选
print("\n请打开 'output/all_industries.csv' 和 'output/all_skills.csv' 文件，挑选高新行业和核心技能，")
print("然后在正式模型中使用选定列表进行聚类分析。")