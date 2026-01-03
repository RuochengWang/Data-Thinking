import pandas as pd
import statsmodels.api as sm

from util import prepare_model_1_data

def load_data(path: str) -> pd.DataFrame:
    """
    Load raw dataset.

    Parameters
    ----------
    path : str
        Path to raw CSV data.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(path)

def run_wage_regression(df: pd.DataFrame):
    """
    Run OLS regression for wage determination.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataset.

    Returns
    -------
    RegressionResults
        Fitted OLS model.
    """
    df_model = prepare_model_1_data(df)

    y = df_model["log_salary"]
    X = df_model.drop(columns=["log_salary"])

    X = sm.add_constant(X)

    model = sm.OLS(y, X)
    results = model.fit(cov_type="HC3")  # robust standard errors

    return results

if __name__ == "__main__":
    data_path = "./data/china_job_market_2025.csv"
    df = load_data(data_path)

    results = run_wage_regression(df)
    print(results.summary())