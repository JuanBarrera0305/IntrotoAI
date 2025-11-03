import subprocess
import json
from pathlib import Path

print(">>> Starting make_report.py")

try:
    print(">>> Running tic_tac_toe.py ...")
    proc = subprocess.run(
        ["python3", "tic_tac_toe.py", "--report"], capture_output=True, text=True
    )
    print(">>> tic_tac_toe.py finished")
    analysis_text = proc.stdout + ("\n\n(Errors:\n" + proc.stderr + ")" if proc.stderr else "")
except Exception as e:
    analysis_text = f"(Could not run tic_tac_toe.py: {e})"
    print(">>> ERROR:", e)

results_json = {}
json_file = Path("results.json")
if json_file.exists():
    print(">>> Loading results.json ...")
    results_json = json.loads(json_file.read_text())

print(">>> Building HTML report ...")
html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Analysis Report</title>
</head>
<body>
  <h1>Analysis Report</h1>
  <pre>{analysis_text.strip()}</pre>
</body>
</html>
"""

print(">>> Writing report.html ...")
Path("report.html").write_text(html, encoding="utf-8")
print(">>> Done! Wrote report.html ✅")