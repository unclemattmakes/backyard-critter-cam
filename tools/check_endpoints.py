"""Cross-check the dashboard's /api/ calls against the routes web.py actually defines, and print the
orphans -- web.py matches paths by hand, so nothing keeps the two sides in step. Run from anywhere:
python tools/check_endpoints.py
"""
import re
from pathlib import Path

# This lives in tools/; the two files it reads sit at the repo root. Resolve off __file__ so the
# script works from any working directory (it used to only run from the root).
ROOT = Path(__file__).resolve().parent.parent

# Read dashboard.js with UTF-8 encoding
dashboard_js = (ROOT / "dashboard.js").read_text(encoding="utf-8")

# Extract all /api/ calls from dashboard.js
called = set(re.findall(r"/api/[a-zA-Z0-9/_\-?=&]+", dashboard_js))
# Simplify to just the endpoint pattern (remove params)
called = set(re.sub(r'[\?&].*', '', c).rstrip("'\"") for c in called)

print("=== CALLED ENDPOINTS (from dashboard.js) ===")
for c in sorted(called):
    print(f"  {c}")

# Extract all /api/ endpoints from web.py
web_py = (ROOT / "web.py").read_text(encoding="utf-8")

defined = set(re.findall(r"path == ['\"](/api/[^'\"]+)['\"]", web_py))
defined |= set(re.findall(r"path.startswith\(['\"](/api/[^'\"]+)['\"]", web_py))

print("\n=== DEFINED ENDPOINTS (from web.py) ===")
for d in sorted(defined):
    print(f"  {d}")

orphaned = defined - called
print(f"\n=== ORPHANED (defined but never called) ===")
for o in sorted(orphaned):
    print(f"  {o}")
