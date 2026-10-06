# Sourced by the launchers: picks the Python interpreter of the logix-gin environment.
#   1. $PY if set
#   2. the active conda env's python ($CONDA_PREFIX), if it is logix-gin
#   3. ~/miniconda3/envs/logix-gin/bin/python or ~/anaconda3/envs/logix-gin/bin/python
#   4. python3 on the PATH
if [ -z "${PY:-}" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ "$(basename "$CONDA_PREFIX")" = logix-gin ]; then
        PY="$CONDA_PREFIX/bin/python"
    else
        for c in "$HOME/miniconda3/envs/logix-gin/bin/python" "$HOME/anaconda3/envs/logix-gin/bin/python"; do
            [ -x "$c" ] && PY="$c" && break
        done
    fi
    PY="${PY:-$(command -v python3)}"
fi
export PY
