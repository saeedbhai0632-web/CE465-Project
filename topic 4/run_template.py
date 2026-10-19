"""CE 465 project - run.py template for Topic 4. Rename to run.py, keep the interface, fill in the work.

python run.py --topic 4 --data <folder containing topic4.csv> --out <folder>
All your code must be in this one file.
"""
import argparse, json, os

def topic4(data_folder):
    # read data_folder/topic4.csv: columns role (fit/check/new), fold, half01..half10,
    # cement, slag, ash, water, superplasticiser, coarse, fine, age_days, strength_MPa
    fields = {
        "rmse_abrams_wc": None, "rmse_abrams_wb": None,                          # part (a)
        "cond_linear_raw": None, "cond_linear_scaled": None, "top_corr_pair": None,  # part (b), e.g. "cement+slag"
        "lambda_cv": None, "rmse_quad_ols": None, "rmse_quad_ridge": None,       # part (c)
        "sd_water_ols": None, "sd_water_ridge": None,                            # part (d)
        "f_new_wb": None, "f_new_ridge": None, "df_water_wb": None, "df_water_ridge": None,  # part (e)
    }
    return {"schema_version": "T4-1.0", "topic": 4, "fields": fields, "diagnostics": {}}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    result = topic4(a.data)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"results_T{a.topic}.json"), "w") as f:
        json.dump(result, f, indent=2)
