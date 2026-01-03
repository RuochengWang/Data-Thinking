import pandas as pd
import numpy as np
import re

def extract_ai_skills(skills_str):
    """
    Define AI_Skills = 1 if and only if the job requires
    Python AND ML AND Statistics.
    """

    if pd.isna(skills_str):
        return 0

    skills = {s.strip().lower() for s in skills_str.split(";")}

    required = {"python", "ml", "statistics"}

    return int(required.issubset(skills))


# City tier mapping based on commonly used Chinese city tier system
CITY_TIER_MAP = {
    # Tier 1
    "Beijing": "T1",
    "Shanghai": "T1",
    "Guangzhou": "T1",
    "Shenzhen": "T1",

    # Tier 2 
    "Tianjin": "T2",
    "Chengdu": "T2",
    "Hangzhou": "T2",
    "Wuhan": "T2",
    "Chongqing": "T2",
    "Nanjing": "T2",
    "Suzhou": "T2",
    "Xi'an": "T2",
    "Changsha": "T2",
    "Qingdao": "T2",
    "Zhengzhou": "T2",
    "Jinan": "T2",
    "Hefei": "T2",
    "Fuzhou": "T2",
    "Xiamen": "T2",
    "Ningbo": "T2"
}

def city_to_tier(city: str) -> str:
    """
    Map city name to city tier.

    Parameters
    ----------
    city : str
        City name.

    Returns
    -------
    str
        City tier label: 'T1', 'T2', or 'T3'.
    """
    if pd.isna(city):
        return "T3"

    return CITY_TIER_MAP.get(city, "T3")

def add_city_tier_column(df: pd.DataFrame, city_col: str = "city") -> pd.DataFrame:
    """
    Add city tier column to DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
    city_col : str
        Column name for city.

    Returns
    -------
    pandas.DataFrame
        DataFrame with an added 'city_tier' column.
    """
    df = df.copy()
    df["city_tier"] = df[city_col].apply(city_to_tier)
    return df

def add_log_salary(df: pd.DataFrame,
                   salary_col: str = "salary_median_cny") -> pd.DataFrame:
    """
    Add log-transformed salary variable.

    Parameters
    ----------
    df : pandas.DataFrame
    salary_col : str
        Salary column name.

    Returns
    -------
    pandas.DataFrame
        DataFrame with an added 'log_salary' column.
    """
    df = df.copy()
    df["log_salary"] = np.log(df[salary_col])
    return df

def one_hot_encode(df: pd.DataFrame,
                   categorical_cols: list,
                   drop_first: bool = True) -> pd.DataFrame:
    """
    One-hot encode categorical variables.

    Parameters
    ----------
    df : pandas.DataFrame
    categorical_cols : list
        List of categorical column names.
    drop_first : bool
        Whether to drop the first category to avoid multicollinearity.

    Returns
    -------
    pandas.DataFrame
        Encoded DataFrame.
    """
    df = df.copy()
    df = pd.get_dummies(df,
                        columns=categorical_cols,
                        drop_first=drop_first)
    return df

def prepare_model_1_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataset for Model 1: Wage determination regression.

    Steps:
    1. Construct AI_Skills variable
    2. Construct city_tier
    3. Construct log_salary
    4. One-hot encode categorical controls

    Returns
    -------
    pandas.DataFrame
        Feature-engineered DataFrame ready for regression.
    """
    df = df.copy()

    # Core variables
    df["AI_Skills"] = df["skills_required"].apply(extract_ai_skills)
    df = add_city_tier_column(df)
    df = add_log_salary(df)

    # Select relevant variables
    selected_cols = [
        "log_salary",
        "AI_Skills",
        "city_tier",
        "experience_level",
    ]

    df = df[selected_cols]

    # One-hot encoding for categorical controls
    categorical_cols = ["experience_level", "city_tier"]
    numeric_cols = ["log_salary", "AI_Skills"]
    
    # get dummies
    dummies = pd.get_dummies(
        df[categorical_cols], 
        drop_first=True,
        dtype=int
    )

    df = pd.concat([df[numeric_cols], dummies], axis=1)

    return df

def create_high_salary_label(df, percentile=0.9):
    """
    在 city 维度计算薪资分位数
    生成高薪岗位 binary 标签
    """
    df = df.copy()
    df['high_salary'] = 0
    for city in df['city'].unique():
        threshold = df.loc[df['city']==city, 'salary_median_cny'].quantile(percentile)
        df.loc[(df['city']==city) & (df['salary_median_cny']>threshold), 'high_salary'] = 1
    return df