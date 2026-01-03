import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression

from util import extract_ai_skills, create_high_salary_label


def prepare_features(df):
    """
    构造模型特征：
    - AI_Skills (0/1)
    - experience_level dummy
    - education_requirement dummy
    - city_tier dummy
    - demand_index (连续)
    """
    df = df.copy()
    df["AI_Skills"] = df["skills_required"].apply(extract_ai_skills)

    # experience_level dummy
    exp_dummies = pd.get_dummies(df["experience_level"], prefix="experience", drop_first=True)

    # education_requirement dummy
    edu_dummies = pd.get_dummies(df["education_requirement"], prefix="edu", drop_first=True)

    # city_tier dummy
    city_dummies = pd.get_dummies(df["city_tier"], prefix="city", drop_first=True)

    X = pd.concat(
        [df[["AI_Skills", "demand_index"]], exp_dummies, edu_dummies, city_dummies],
        axis=1
    )
    y = df["high_salary"]

    return X, y


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
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    print(f"{model_name} -- Accuracy: {acc:.3f}, ROC-AUC: {auc:.3f}, F1: {f1:.3f}")
    return acc, auc, f1


def shap_analysis(model, X_train, feature_names, output_path="output/model_2_shap.png"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"SHAP plot saved to {output_path}")


if __name__ == "__main__":
    # 1. 读数据
    df = pd.read_csv("data/china_job_market_2025.csv")

    # 2. 构造高薪岗位标签（城市 90th percentile）
    df = create_high_salary_label(df, percentile=0.9)

    # 3. 特征处理
    X, y = prepare_features(df)

    # 4. 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 5. Logistic 回归 baseline
    log_model = train_logistic(X_train, y_train)
    evaluate_model(log_model, X_test, y_test, model_name="Logistic Regression")
    joblib.dump(log_model, "output/logistic_model.pkl")

    # 6. XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost")
    joblib.dump(xgb_model, "output/xgb_model.pkl")

    # 7. SHAP 分析
    shap_analysis(xgb_model, X_train, feature_names=X.columns.tolist())
