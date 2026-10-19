"""CE 465 project - run.py template. Rename to run.py, keep the interface, fill in the work.

python run.py --topic 1 --data <folder containing topic1.csv> --out <folder>
"""
import argparse, json, os

def topic1(data_folder):
    # read data_folder/topic1.csv (columns h_m, Q_m3s) and compute every field below
    fields = {
        "a_naive": None, "b_naive": None, "Q5_naive": None,          # part (a)
        "h0_scan": None, "SSE_scan": None,                            # part (b)
        "lna_R1": None, "b_R1": None, "h0_R1": None, "h0_R2": None,   # part (c)
        "Q_hold_naive": [None, None, None], "Q_hold_fit": [None, None, None],
        "holdout_winner": None,                                       # part (d): "naive" or "fitted"
        "Q5": None, "Q5_err": None,                                  # part (e)
    }
    diagnostics = {}   # optional: iterations_R1, iterations_R2, R3_failed, step_norms_R1
    return {"schema_version": "T1-1.0", "topic": 1, "fields": fields, "diagnostics": diagnostics}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    result = {1: topic1}[a.topic](a.data)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"results_T{a.topic}.json"), "w") as f:
        json.dump(result, f, indent=2)
