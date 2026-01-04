import os
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

from util import calculate_city_features

def run_model_1(df, output_dir="output"):
    """
    模型一：城市层面薪资决定机制（机器学习回归）
    """

    # 1. 构造城市层面特征
    HIGH_TECH_INDUSTRIES = [
    "Design/IT",
    "Engineering",
    "IT",
    "Finance",
    "Healthcare"
    ]
    
    city_df = calculate_city_features(
        df,
        high_tech_industries=HIGH_TECH_INDUSTRIES,
    )

    city_df["avg_log_salary"] = np.log(city_df["avg_salary"])

    features = [
        "phd_ratio",
        "hi_industry_ratio",
        "skill_entropy"
    ]

    X = city_df[features]
    y = city_df["avg_log_salary"]

    # 2. Gradient Boosting 回归
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(X, y)

    # 3. 拟合效果（仅作为参考）
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print(f"Model 1 | R2: {r2:.3f}, RMSE: {rmse:.3f}")

    # 4. 特征重要性
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    # 5. 输出结果
    os.makedirs(output_dir, exist_ok=True)
    importance.to_csv(f"{output_dir}/model_1_feature_importance.csv", index=False)
    city_df.to_csv(f"{output_dir}/model_1_city_data.csv", index=False)

    return model, importance, city_df


if __name__ == "__main__":
    df = pd.read_csv("./data/china_job_market_2025.csv")
    model, importance, city_df = run_model_1(df)
    print(importance)