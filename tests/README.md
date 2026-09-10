# tests

No test framework is committed yet. `pytest` is the presumed choice, but this
is not yet decided or wired into a `pyproject.toml`.

`test_smoke.py` is deliberately framework-agnostic (plain `assert`, no
`pytest`-only syntax) so it stays useful regardless of the final decision.

Once real logic exists (not just stubs), expect this to mirror
`src/visual_radar/`, e.g. `test_geometry.py`, `test_scene_builder.py`,
`test_palette.py`.
