import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
from util import evaluate_clustering

def run_clustering_analysis(df, output_dir="../output"):
    """
    运行城市聚类分析
    """
    # 按城市聚合指标
    city_stats = df.groupby("city").agg({
        "salary_mid": "mean",
        "exp_years": "mean",
        "edu_level": "mean",
        "has_llm": "mean",
        "is_ai_industry": "mean",
        "skill_count": "mean"
    }).reset_index()
    
    # 添加城市等级
    tier1 = ["北京市", "上海市", "深圳市", "广州市"]
    new_tier1 = ["成都市", "杭州市", "重庆市", "苏州市", "武汉市", "西安市", 
                "南京市", "天津市", "郑州市", "长沙市", "东莞市", "宁波市", 
                "佛山市", "合肥市", "青岛市"]
    
    def get_tier(city):
        if city in tier1:
            return "一线"
        elif city in new_tier1:
            return "新一线"
        else:
            return "二线及以下"
    
    city_stats["city_tier"] = city_stats["city"].apply(get_tier)
    
    # 选择聚类特征
    cluster_features = [
        "salary_mid", 
        "exp_years", 
        "edu_level", 
        "has_llm", 
        "is_ai_industry",
        "skill_count"
    ]
    
    X = city_stats[cluster_features].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 选择最优K（轮廓系数）
    silhouette_scores = []
    K_range = range(2, 8)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = evaluate_clustering(X_scaled, labels)["silhouette_score"]
        silhouette_scores.append(score)
        print(f"K={k}, 轮廓系数={score:.4f}")
    
    # 选择最优K
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"\n✅ 最优聚类数: K={optimal_k}")
    
    # 最终聚类
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    city_stats["cluster"] = kmeans_final.fit_predict(X_scaled)
    
    # PCA降维可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    city_stats["pca1"] = X_pca[:, 0]
    city_stats["pca2"] = X_pca[:, 1]
    
    # 绘制聚类结果
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        city_stats["pca1"], 
        city_stats["pca2"],
        c=city_stats["cluster"],
        cmap="viridis",
        s=100
    )
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} 方差)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} 方差)")
    plt.title("城市人才吸引力聚类 (PCA降维)")
    plt.colorbar(scatter, label="聚类簇")
    
    # 标注部分城市
    highlight_cities = ["北京市", "上海市", "深圳市", "杭州市", "成都市", "合肥市"]
    for _, row in city_stats[city_stats["city"].isin(highlight_cities)].iterrows():
        plt.annotate(
            row["city"],
            (row["pca1"], row["pca2"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9
        )
    
    plt.tight_layout()
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    plt.savefig(f"{output_dir}/figures/city_clustering.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 计算各类中心
    cluster_summary = city_stats.groupby("cluster")[cluster_features].mean()
    cluster_summary["city_count"] = city_stats["cluster"].value_counts().sort_index().values
    
    # 保存结果
    cluster_summary.to_csv(f"{output_dir}/tables/cluster_summary.csv")
    city_stats.to_csv(f"{output_dir}/tables/city_clusters.csv", index=False)
    
    print("\n✅ 城市聚类完成!")
    print("各类中心特征:")
    print(cluster_summary.round(3))
    
    return city_stats, cluster_summary, optimal_k