
import subprocess, sys, json, pathlib

scripts = [
    "major_arc.py",
    "minor_arc.py",
    "main_theorem.py",
    "appendix_formalization.py",
    "fractional_cumulants_formalization.py",
    "full_chowla_steps.py",
    "full_chowla_conditional.py"
]

report = {}
for s in scripts:
    p = subprocess.run([sys.executable, s], cwd="/mnt/data", capture_output=True, text=True)
    report[s] = {
        "returncode": p.returncode,
        "stdout_tail": p.stdout[-1200:],
        "stderr_tail": p.stderr[-1200:]
    }

path = pathlib.Path("/mnt/data/prelim_run.json")
path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("Wrote", path)
