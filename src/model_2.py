import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import joblib

from util import extract_ai_skills, create_high_salary_label, create_city_tier

# ===========================
# 统一创建 output 文件夹
# ===========================
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)  # 如果不存在就创建

# ===========================
# 特征处理函数
# ===========================
def prepare_features(df, model_type="logistic"):
    """
    构造模型特征
    model_type: "logistic" 或 "xgboost"，决定 dummy 是否 drop_first
    """
    df = df.copy()
    df["AI_Skills"] = df["skills_required"].apply(extract_ai_skills)

    drop_first = True if model_type == "logistic" else False

    exp_dummies = pd.get_dummies(df["experience_level"], prefix="experience", drop_first=drop_first)
    edu_dummies = pd.get_dummies(df["education_requirement"], prefix="edu", drop_first=drop_first)
    city_dummies = pd.get_dummies(df["city_tier"], prefix="city", drop_first=drop_first)

    X = pd.concat([df[["AI_Skills", "demand_index"]], exp_dummies, edu_dummies, city_dummies], axis=1)
    y = df["high_salary"]

    return X, y


# ===========================
# 模型训练函数
# ===========================
def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


# ===========================
# 模型评估函数
# ===========================
def evaluate_model(model, X_test, y_test, model_name="model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    print(f"{model_name} -- Accuracy: {acc:.3f}, ROC-AUC: {auc:.3f}, F1: {f1:.3f}")
    return acc, auc, f1


# ===========================
# SHAP 分析函数
# ===========================
def shap_analysis(model, X_train, feature_names, output_path=os.path.join(output_dir, "model_2_shap.png")):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"SHAP plot saved to {output_path}")


# ===========================
# 主程序
# ===========================
if __name__ == "__main__":
    # 1. 读数据
    df = pd.read_csv("data/china_job_market_2025.csv")

    # 2. 构造 city_tier
    df = create_city_tier(df)

    # 3. 构造高薪岗位标签
    df = create_high_salary_label(df)

    # 4. 构造特征
    X_log, y_log = prepare_features(df, model_type="logistic")
    X_xgb, y_xgb = prepare_features(df, model_type="xgboost")

    # 5. 划分训练集/测试集
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
        X_log, y_log, test_size=0.3, random_state=42, stratify=y_log
    )
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
        X_xgb, y_xgb, test_size=0.3, random_state=42, stratify=y_xgb
    )

    # 6. Logistic 回归
    log_model = train_logistic(X_train_log, y_train_log)
    evaluate_model(log_model, X_test_log, y_test_log, "Logistic Regression")
    joblib.dump(log_model, os.path.join(output_dir, "logistic_model.pkl"))

    # 7. XGBoost
    xgb_model = train_xgboost(X_train_xgb, y_train_xgb)
    evaluate_model(xgb_model, X_test_xgb, y_test_xgb, "XGBoost")
    joblib.dump(xgb_model, os.path.join(output_dir, "xgb_model.pkl"))

    # 8. SHAP 分析
    shap_analysis(xgb_model, X_train_xgb, feature_names=X_xgb.columns.tolist())