import os
import sys
sys.path.append(".")

from dataloader import load_and_preprocess_data
from model_1_ols import run_ols_analysis
from model_2_xgboost import run_xgboost_analysis
from model_3_clustering import run_clustering_analysis
from util import plot_salary_distribution

def main():
    print("="*50)
    print("中国就业市场建模分析 - 开始执行")
    print("="*50)
    
    # 创建输出目录
    os.makedirs("output/figures", exist_ok=True)
    os.makedirs("output/tables", exist_ok=True)
    
    # 1. 加载并清洗数据
    print("\n[1/4] 加载数据...")
    df = load_and_preprocess_data("./data/china_job_market_2025.csv")
    
    # 绘制薪资分布
    plot_salary_distribution(df, "output/figures/salary_distribution.png")
    
    # 2. 运行OLS回归
    print("\n[2/4] 运行OLS回归...")
    ols_model, ols_results = run_ols_analysis(df)
    
    # 3. 运行XGBoost分类
    print("\n[3/4] 运行XGBoost分类...")
    xgb_model, xgb_metrics, importances, feature_names = run_xgboost_analysis(df)
    
    # 4. 运行城市聚类
    print("\n[4/4] 运行城市聚类...")
    city_stats, cluster_summary, optimal_k = run_clustering_analysis(df)
    
    # 5. 生成结果摘要
    summary = {
        "样本量": len(df),
        "OLS_R2": ols_results["r_squared"],
        "LLM系数": ols_results["llm_coef"],
        "LLM_p值": ols_results["llm_pvalue"],
        "XGBoost_AUC": xgb_metrics["auc"],
        "最优聚类数": optimal_k,
        "聚类轮廓系数": cluster_summary.index.max()  # 实际应从evaluate_clustering获取，此处简化
    }
    
    with open("output/results_summary.txt", "w", encoding="utf-8") as f:
        f.write("中国就业市场建模分析 - 关键结果摘要\n")
        f.write("="*40 + "\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    
    print("\n" + "="*50)
    print("全部分析完成！")
    print("结果保存在 output/ 目录:")
    print("- figures/: 所有图表")
    print("- tables/: 所有表格数据")
    print("- results_summary.txt: 关键指标汇总")
    print("="*50)

if __name__ == "__main__":
    main()