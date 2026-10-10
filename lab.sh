#!/usr/bin/env bash
# Run IsaacLab with conda/venv variables cleared (isaaclab.sh refuses to run inside conda).
# Usage: ./lab.sh -p scripts/foo.py --viz none [...]
HERE="$(cd "$(dirname "$0")" && pwd)"
CLEAN_PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda3 | paste -sd:)
exec env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_PROMPT_MODIFIER -u VIRTUAL_ENV \
  PATH="$CLEAN_PATH" OMNI_KIT_ACCEPT_EULA=yes PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" \
  "$HOME/IsaacLab/isaaclab.sh" "$@"
