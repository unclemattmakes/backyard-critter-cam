# docs/ — the project's paper trail

These are dated snapshots, not current documentation. Each one was written on the day in its
banner and is kept because it explains *why* the code looks the way it does; findings in them have
largely been addressed since. For how the system works and how to run it, read the
[README](../README.md) at the repo root.

- [plan.md](plan.md) — the original project plan (2026-06-07), written before a line of it existed:
  the two-axis "looks like X but isn't acting like X" idea, the four phases, and the per-species
  reality check on individual ID.
- [review-2026-06-19.md](review-2026-06-19.md) — a whole-rig code review at the point the project
  turned from a private tool into something a friend could install, plus the fix roadmap it drove.
- [observatory-2026-06-28.md](observatory-2026-06-28.md) — a data/UX/ML analysis of the first 22
  days of live capture: what the yard actually logged, what the pipeline got wrong, and which
  signals were being computed but never surfaced.

Developer utilities live in [../tools/](../tools/); `check_endpoints.py` there cross-checks the
dashboard's `/api/` calls against the routes `web.py` actually defines.
