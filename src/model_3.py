import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from util import calculate_city_features, standardize_features


# ======================
# 人为设定的经济学规则
# ======================

HIGH_TECH_INDUSTRIES = [
    "Design/IT",
    "Engineering",
    "IT",
    "Finance",
    "Healthcare"
]

def run_city_clustering(df, n_clusters=4, output_dir="output"):
    """
    城市人才吸引力画像模型（基于产业结构与技能多样性）
    """

    # =====================
    # 1. 构造城市层面指标
    # =====================
    city_df = calculate_city_features(
        df,
        high_tech_industries=HIGH_TECH_INDUSTRIES,
    )

    # 薪资取对数，缓解右偏
    city_df["avg_log_salary"] = np.log(city_df["avg_salary"])

    features = [
        "avg_log_salary",     # 薪资水平
        "phd_ratio",          # 高端人才
        "hi_industry_ratio",  # 高端产业结构
        "skill_entropy"       # 技能多样性
    ]

    # =====================
    # 2. 标准化
    # =====================
    city_scaled = standardize_features(city_df, features)

    # =====================
    # 3. PCA（仅用于可视化）
    # =====================
    pca = PCA(n_components=2, random_state=42)
    city_pca = pca.fit_transform(city_scaled[features])
    city_scaled["pca1"] = city_pca[:, 0]
    city_scaled["pca2"] = city_pca[:, 1]

    # =====================
    # 4. K-means 聚类
    # =====================
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20
    )
    city_scaled["cluster"] = kmeans.fit_predict(city_scaled[features])

    # =====================
    # 5. 输出结果
    # =====================
    city_scaled.to_csv(
        f"{output_dir}/城市画像_聚类结果.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # =====================
    # 6. 可视化
    # =====================
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=city_scaled,
        x="pca1",
        y="pca2",
        hue="cluster",
        palette="tab10",
        s=100
    )

    for _, row in city_scaled.iterrows():
        plt.text(
            row["pca1"] + 0.03,
            row["pca2"] + 0.03,
            row["city"],
            fontsize=9
        )

    plt.title("City Profiles Based on Skill Diversity and Industry Structure (PCA Visualization)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="City Type")
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/城市画像_PCA聚类图.png",
        dpi=300
    )
    plt.close()

    # =====================
    # 7. 聚类画像均值
    # =====================
    cluster_summary = (
        city_scaled
        .groupby("cluster")[features]
        .mean()
        .reset_index()
    )

    cluster_summary.to_csv(
        f"{output_dir}/城市画像_聚类特征均值.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return city_scaled, cluster_summary


if __name__ == "__main__":
    import os
    if not os.path.exists("output"):
        os.makedirs("output")

    df = pd.read_csv("./data/china_job_market_2025.csv")
    city_scaled, cluster_summary = run_city_clustering(df)

    print("各类城市画像特征均值：")
    print(cluster_summary)