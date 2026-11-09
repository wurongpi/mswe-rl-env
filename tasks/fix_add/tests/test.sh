#!/bin/bash
# exit 0 = pass. Runs after the agent finishes; the agent never sees this directory.
set -u
cd "$WORKSPACE"
python -m unittest discover -s "$TESTS_DIR" -p 'test_*.py' -v
