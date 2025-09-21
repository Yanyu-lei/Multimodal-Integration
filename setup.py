"""
Package metadata and installation script
========================================

Purpose
-------
Provide a minimal, reviewer‑friendly `pip install -e .` experience that:
  • exposes the `src/` package (so imports like `from src...` work),
  • reads runtime dependencies from `requirements.txt` (skips test‑only deps),
  • keeps metadata simple and OS‑agnostic.

Typical usage
-------------
# Editable install for development/review
pip install -e .

Notes
-----
• PyTorch / torchvision wheels should be installed separately per platform
  (the README and requirements mention this explicitly).
• This script intentionally avoids complex build steps; it just wires up the
  Python package and reads `requirements.txt`.
"""
from __future__ import annotations

from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent


def read_requirements() -> list[str]:
    """
    Read `requirements.txt`, skipping comments/blank lines and test‑only deps.
    """
    req_path = ROOT / "requirements.txt"
    reqs: list[str] = []
    if req_path.exists():
        for line in req_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # keep runtime deps only (skip pytest etc.)
            if line.lower().startswith("pytest"):
                continue
            reqs.append(line)
    return reqs


setup(
    name="multimodal_integration",
    version="0.1.0",
    description="Evaluation suite for how multimodal models integrate image + text",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8")
    if (ROOT / "README.md").exists()
    else "",
    long_description_content_type="text/markdown",
    url="https://example.com/multimodal_integration",  # optional: replace or remove
    author="",
    license="MIT",
    packages=find_packages(include=["src", "src.*"]),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Source": "https://example.com/multimodal_integration",
        "Issues": "https://example.com/multimodal_integration/issues",
    },
)