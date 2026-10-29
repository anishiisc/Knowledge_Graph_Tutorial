#!/usr/bin/env python3
"""Assemble the lab notebook from its cell definitions and execute it.

Usage: python build_notebook.py [--no-run]
Run from the kg_tutorial directory (the notebook expects ./data and ./figures).
"""
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_part1, nb_part2, nb_part3  # noqa: E402

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
cells = []


def M(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def C(code):
    cells.append(nbf.v4.new_code_cell(code.strip("\n")))


for part in (nb_part1, nb_part2, nb_part3):
    part.cells(M, C)
nb.cells = cells

out = ROOT / "knowledge_graphs_lab.ipynb"
if "--no-run" not in sys.argv:
    client = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
nbf.write(nb, out)
print("wrote", out, "with", len(nb.cells), "cells")
