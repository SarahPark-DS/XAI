"""
parse_results.py — Parser for α,β-Crown stdout output.

α,β-Crown prints per-instance results as:
    Result: unsat      (property holds → verified/robust)
    Result: sat        (counterexample found → falsified/not robust)
    Result: timeout    (exceeded time limit)
    Result: unknown    (incomplete verification, no conclusion)

And timing as:
    Total time: X.XXXs
"""
import re


def parse_abcrown_output(stdout: str, wall_time: float) -> dict:
    """Extract result and timing from a single abcrown subprocess call.

    Args:
        stdout: captured stdout text from the abcrown process
        wall_time: elapsed wall-clock seconds (used as fallback if not parsed)

    Returns:
        dict with keys:
            result  (str): 'verified' | 'falsified' | 'timeout' | 'unknown'
            time_s  (float): verification time in seconds
    """
    result = "timeout"
    time_s = wall_time

    for line in stdout.splitlines():
        low = line.lower().strip()

        # Result classification
        if low.startswith("result:"):
            token = low.split("result:")[-1].strip().split()[0]
            if token in ("unsat", "safe"):
                result = "verified"
            elif token in ("sat", "unsafe", "unsafe-pgd", "falsified"):
                result = "falsified"
            elif token in ("timeout",):
                result = "timeout"
            elif token in ("unknown",):
                result = "unknown"

        # Time extraction: "Total time: 0.342s" or "Time: 0.342"
        m = re.search(r"total time[:\s]+([0-9]+\.?[0-9]*)", low)
        if m:
            time_s = float(m.group(1))

    return {"result": result, "time_s": round(time_s, 4)}


def parse_abcrown_log_file(log_path: str) -> list:
    """Parse a multi-instance abcrown results log file.

    abcrown can write a results file (general.results_file in YAML) with
    lines like: 'path/to/spec.vnnlib,result,time'

    Returns:
        List of dicts: [{spec_path, result, time_s}, ...]
    """
    results = []
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    spec = parts[0].strip()
                    res_token = parts[1].strip().lower()
                    time_s = float(parts[2].strip()) if len(parts) >= 3 else 0.0
                    if res_token == "unsat":
                        result = "verified"
                    elif res_token == "sat":
                        result = "falsified"
                    else:
                        result = res_token  # timeout, unknown
                    results.append({"spec_path": spec, "result": result, "time_s": time_s})
    except FileNotFoundError:
        pass
    return results
