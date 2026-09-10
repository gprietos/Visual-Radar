"""Trivial import/instantiation sanity check. Framework-agnostic on purpose."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import visual_radar
from visual_radar.sources import Frame


def test_package_imports() -> None:
    frame = Frame(image=torch.zeros(3, 1080, 1920), yaw=0.0, pitch=0.0)
    assert frame.roll == 0.0


if __name__ == "__main__":
    test_package_imports()
    print("ok")
