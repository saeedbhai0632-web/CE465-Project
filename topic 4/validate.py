"""CE 465 project - submission self-check (format and running only; NOT correctness).

Usage:  python validate.py --code <folder with run.py> --topic <1-4> --data <folder> [--schema schema_Tk.json]
Checks that run.py finishes within 120 s and writes a results_T<k>.json that matches the schema.
"""
import argparse, json, os, subprocess, sys, tempfile, math

def check(results, schema):
    errs = []
    if results.get("schema_version") != schema["schema_version"]:
        errs.append(f"schema_version should be {schema['schema_version']!r}")
    if results.get("topic") != schema["topic"]:
        errs.append(f"topic should be {schema['topic']}")
    fields = results.get("fields", {})
    for name, s in schema["fields"].items():
        if name not in fields:
            errs.append(f"missing field {name}"); continue
        v = fields[name]
        if s["type"] == "number":
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                errs.append(f"{name} must be a finite number")
        elif s["type"] == "array":
            if (not isinstance(v, list) or len(v) != s["length"]
                    or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v)):
                errs.append(f"{name} must be a list of {s['length']} finite numbers")
        elif s["type"] == "string":
            if v not in s["values"]:
                errs.append(f"{name} must be one of {s['values']}")
    return errs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True); ap.add_argument("--topic", type=int, required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--schema")
    a = ap.parse_args()
    schema_path = a.schema or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"schema_T{a.topic}.json")
    schema = json.load(open(schema_path))
    with tempfile.TemporaryDirectory() as out:
        cmd = [sys.executable, "run.py", "--topic", str(a.topic), "--data", os.path.abspath(a.data), "--out", out]
        try:
            p = subprocess.run(cmd, cwd=a.code, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            print("FAIL: run.py took longer than 120 s"); return 1
        if p.returncode != 0:
            print("FAIL: run.py stopped with an error:\n" + p.stderr[-2000:]); return 1
        path = os.path.join(out, f"results_T{a.topic}.json")
        if not os.path.exists(path):
            print(f"FAIL: run.py did not write results_T{a.topic}.json in the --out folder"); return 1
        try: results = json.load(open(path))
        except Exception as e:
            print(f"FAIL: results_T{a.topic}.json is not valid JSON ({e})"); return 1
        errs = check(results, schema)
        if errs:
            print("FAIL:\n  " + "\n  ".join(errs)); return 1
    print("OK: run.py ran and wrote a file in the right format. This does not check that the numbers are correct.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
