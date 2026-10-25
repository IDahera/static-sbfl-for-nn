# SPDX-License-Identifier: GPL-3.0-or-later
# Part of a modified version of DeepFault (c) 2019 Bogazici University; written 2026-10-20 by
# Daniel Rashedi for the diss-code runtime comparison. Distributed under the GPL-3.0-or-later.
"""Runtime entry point for DF@tf2 (new file, not upstream).

Runs the shared runner in ../common/deepfault_runtime.py with THIS directory first
on sys.path, so it measures this port's utils / test_nn / spectrum_analysis.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "common"))

from deepfault_runtime import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:] + ["--port", "tf2"])
