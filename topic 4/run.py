
import argparse
import json
import os
import numpy as np
import pandas as pd


# --------------------------------------------------
# Basic settings
# --------------------------------------------------

INPUTS = [
    "cement", "slag", "ash", "water",
    "superplasticiser", "coarse", "fine", "ln_age"
]

LAMBDAS = 10.0 ** np.arange(-3, 3.01, 0.5)


# --------------------------------------------------
# Common helper functions
# --------------------------------------------------

def get_inputs(df):
    X = df[
        ["cement", "slag", "ash", "water",
         "superplasticiser", "coarse", "fine"]
    ].to_numpy(float)

    age = np.log(df["age_days"].to_numpy(float))

    return np.column_stack((X, age))


def rmse(predicted, actual):
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def scale_data(A, mean=None, std=None):
    # Leave the intercept column unchanged.
    if mean is None:
        mean = A[:, 1:].mean(axis=0)
        std = A[:, 1:].std(axis=0)

        std[std == 0] = 1.0

    B = A.copy()
    B[:, 1:] = (A[:, 1:] - mean) / std

    return B, mean, std


def fit_model(A, y, lam=0):
    # Scale all columns except the intercept.
    A_scaled, mean, std = scale_data(A)

    if lam == 0:
        # Ordinary least squares
        coef = np.linalg.lstsq(A_scaled, y, rcond=None)[0]
    else:
        # Ridge regression; do not penalise intercept.
        penalty = lam * np.eye(A.shape[1])
        penalty[0, 0] = 0

        coef = np.linalg.solve(
            A_scaled.T @ A_scaled + penalty,
            A_scaled.T @ y
        )

    return coef, mean, std


def predict_model(A, coef, mean, std):
    A_scaled, _, _ = scale_data(A, mean, std)
    return A_scaled @ coef


def make_linear_design(X):
    return np.column_stack((np.ones(len(X)), X))


def make_quadratic_design(X):
    columns = [np.ones(len(X))]

    # Add the original 8 inputs.
    for i in range(X.shape[1]):
        columns.append(X[:, i])

    # Add all 36 products xi*xj, including squares.
    for i in range(X.shape[1]):
        for j in range(i, X.shape[1]):
            columns.append(X[:, i] * X[:, j])

    return np.column_stack(columns)


# --------------------------------------------------
# Part (a): Abrams laws
# --------------------------------------------------

def abrams_design(df, use_binder):
    denom = df["cement"].to_numpy(float)

    if use_binder:
        denom = (
            denom
            + df["slag"].to_numpy(float)
            + df["ash"].to_numpy(float)
        )

    ratio = df["water"].to_numpy(float) / denom
    log_age = np.log(df["age_days"].to_numpy(float))

    return np.column_stack((
        np.ones(len(df)), ratio, log_age
    ))


def fit_abrams(train, use_binder):
    A = abrams_design(train, use_binder)
    y = np.log(train["strength_MPa"].to_numpy(float))

    return np.linalg.lstsq(A, y, rcond=None)[0]


def predict_abrams(df, coef, use_binder):
    A = abrams_design(df, use_binder)
    return np.exp(A @ coef)



# --------------------------------------------------
# Part (b): Correlation and condition number
# --------------------------------------------------

def find_top_correlation(X):
    corr = np.corrcoef(X, rowvar=False)

    best_i = 0
    best_j = 1
    best_value = -1

    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            value = abs(corr[i, j])

            if value > best_value:
                best_value = value
                best_i = i
                best_j = j

    return INPUTS[best_i] + "+" + INPUTS[best_j]


# --------------------------------------------------
# Part (c): Choose ridge lambda using cross-validation
# --------------------------------------------------

def choose_lambda(Q, y, folds):
    scores = []

    for lam in LAMBDAS:
        total_error = 0.0

        for k in range(1, 6):
            train_rows = folds != k
            test_rows = folds == k

            coef, mean, std = fit_model(
                Q[train_rows], y[train_rows], lam
            )

            predictions = predict_model(
                Q[test_rows], coef, mean, std
            )

            errors = predictions - y[test_rows]
            total_error += np.sum(errors ** 2)

        scores.append(total_error)

    best_score = min(scores)

    # If scores tie within relative 1e-12, choose larger lambda.
    best_lambda = LAMBDAS[0]

    for i in range(len(LAMBDAS)):
        if scores[i] <= best_score * (1 + 1e-12):
            best_lambda = LAMBDAS[i]

    return float(best_lambda)


# --------------------------------------------------
# Part (d): Water coefficient over half-samples
# --------------------------------------------------
def water_spread(fit, L, y):
    L, _, _ = scale_data(L)

    ols = []
    ridge = []

    for h in range(1, 11):
        col = "half" + str(h).zfill(2)
        mask = fit[col].to_numpy(int) == 1

        A = L[mask]
        b = y[mask]

        ols_coef = np.linalg.lstsq(A, b, rcond=None)[0]

        penalty = 10 * np.eye(A.shape[1])
        penalty[0, 0] = 0
        ridge_coef = np.linalg.solve(
            A.T @ A + penalty, A.T @ b
        )

        ols.append(ols_coef[4])
        ridge.append(ridge_coef[4])

    return float(np.std(ols, ddof=1)), float(np.std(ridge, ddof=1))

# --------------------------------------------------
# Main program
# --------------------------------------------------

def main(data_folder, out_folder):
    df = pd.read_csv(os.path.join(data_folder, "topic4.csv"))

    fit = df[df["role"] == "fit"].reset_index(drop=True)
    check = df[df["role"] == "check"].reset_index(drop=True)
    new = df[df["role"] == "new"].reset_index(drop=True)

    y = fit["strength_MPa"].to_numpy(float)
    y_check = check["strength_MPa"].to_numpy(float)

    X = get_inputs(fit)
    X_check = get_inputs(check)

    # Part (a): Abrams models
    coef_wc = fit_abrams(fit, False)
    coef_wb = fit_abrams(fit, True)

    pred_wc = predict_abrams(check, coef_wc, False)
    pred_wb = predict_abrams(check, coef_wb, True)

    error_wc = rmse(pred_wc, y_check)
    error_wb = rmse(pred_wb, y_check)

    # Part (b): Linear model conditioning
    L = make_linear_design(X)

    cond_raw = np.linalg.cond(L)

    L_scaled, _, _ = scale_data(L)
    cond_scaled = np.linalg.cond(L_scaled)

    top_pair = find_top_correlation(X)

    # Part (c): Quadratic model
    Q = make_quadratic_design(X)
    Q_check = make_quadratic_design(X_check)

    folds = fit["fold"].to_numpy(int)
    best_lambda = choose_lambda(Q, y, folds)

    coef_ols, mean_ols, std_ols = fit_model(Q, y, 0)
    pred_quad_ols = predict_model(
        Q_check, coef_ols, mean_ols, std_ols
    )

    coef_ridge, mean_ridge, std_ridge = fit_model(
        Q, y, best_lambda
    )
    pred_quad_ridge = predict_model(
        Q_check, coef_ridge, mean_ridge, std_ridge
    )

    error_quad_ols = rmse(pred_quad_ols, y_check)
    error_quad_ridge = rmse(pred_quad_ridge, y_check)

    # Part (d): Coefficient spread
    sd_ols, sd_ridge = water_spread(fit, L, y)

    # Part (e): New mix and extra water

    # Create a second mix with 10 extra units of water.
    new_wet = new.copy()
    new_wet["water"] = new_wet["water"] + 10.0

    # Prepare inputs for both mixes FIRST.
    X_new = get_inputs(new)
    X_wet = get_inputs(new_wet)

    Q_new = make_quadratic_design(X_new)
    Q_wet = make_quadratic_design(X_wet)

    # Predict strength of the original mix.
    strength_original_wb = predict_abrams(
        new, coef_wb, True
    )[0]

    strength_original_ridge = predict_model(
        Q_new, coef_ridge, mean_ridge, std_ridge
    )[0]

    # Predict strength of the mix with extra water.
    strength_extra_water_wb = predict_abrams(
        new_wet, coef_wb, True
    )[0]

    strength_extra_water_ridge = predict_model(
        Q_wet, coef_ridge, mean_ridge, std_ridge
    )[0]

    # Calculate the change in predicted strength.
    change_water_wb = (
        strength_extra_water_wb - strength_original_wb
    )

    change_water_ridge = (
        strength_extra_water_ridge - strength_original_ridge
    )

    # Collect all required answers.
    fields = {
        "rmse_abrams_wc": error_wc,
        "rmse_abrams_wb": error_wb,
        "cond_linear_raw": float(cond_raw),
        "cond_linear_scaled": float(cond_scaled),
        "top_corr_pair": top_pair,
        "lambda_cv": best_lambda,
        "rmse_quad_ols": error_quad_ols,
        "rmse_quad_ridge": error_quad_ridge,
        "sd_water_ols": sd_ols,
        "sd_water_ridge": sd_ridge,
        "f_new_wb": float(strength_original_wb),
        "f_new_ridge": float(strength_original_ridge),
        "df_water_wb": float(change_water_wb),
        "df_water_ridge": float(change_water_ridge)
    }

    result = {
        "schema_version": "T4-1.0",
        "topic": 4,
        "fields": fields,
        "diagnostics": {}
    }

    os.makedirs(out_folder, exist_ok=True)

    output_file = os.path.join(out_folder, "results_T4.json")

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print("Results saved to:", output_file)
    print("Part (a) water/cement RMSE:", error_wc)
    print("Part (a) water/binder RMSE:", error_wb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--topic", type=int, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    if args.topic != 4:
        raise ValueError("This program is for Topic 4 only.")

    main(args.data, args.out)