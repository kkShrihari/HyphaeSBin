#!/usr/bin/env bash
# =============================================================================
# HyphaeSBin installer (v9) -- one command, no manual steps.
#   bash install.sh                 # HyphaeSBin_env + HyphaeSBin_classify (+ pip install of the pipeline)
#   bash install.sh --with-eukcc    # also EukCC 2 in its own env (name: --eukcc-env, default eukcc)
#   bash install.sh --force         # rebuild envs that already exist
# Uses mamba > micromamba > conda (strict channel priority, conda-forge + bioconda only).
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_ENV="HyphaeSBin_env"
CLS_ENV="HyphaeSBin_classify"
EUKCC_ENV="eukcc"
WITH_EUKCC=0
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-eukcc) WITH_EUKCC=1 ;;
    --eukcc-env) EUKCC_ENV="$2"; shift ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done

if command -v mamba >/dev/null 2>&1; then CONDA=mamba
elif command -v micromamba >/dev/null 2>&1; then CONDA=micromamba
elif command -v conda >/dev/null 2>&1; then CONDA=conda
else echo "ERROR: need conda, mamba or micromamba on PATH (https://github.com/conda-forge/miniforge)"; exit 1; fi
CH=(-c conda-forge -c bioconda --strict-channel-priority)
echo "[install] using $CONDA"

env_prefix() {   # works for conda, mamba and micromamba
  $CONDA env list --json 2>/dev/null | tr -d ' \n' | tr ',' '\n' | tr -d '[]"{}' | sed 's/^envs://' \
    | grep -E "/$1\$" | head -1
}
env_exists() { [[ -n "$(env_prefix "$1")" ]]; }
env_create_file() { if [[ $CONDA == micromamba ]]; then micromamba create -y -n "$1" -f "$2"; else $CONDA env create -y -n "$1" -f "$2" || $CONDA env create -n "$1" -f "$2"; fi; }
env_update_file() { if [[ $CONDA == micromamba ]]; then micromamba install -y -n "$1" -f "$2"; else $CONDA env update -n "$1" -f "$2"; fi; }

# 1) main env
if env_exists "$MAIN_ENV" && [[ $FORCE -eq 0 ]]; then
  echo "[install] $MAIN_ENV exists -> updating from environment.yaml"
  env_update_file "$MAIN_ENV" "$HERE/environment.yaml"
else
  if [[ $FORCE -eq 1 ]] && env_exists "$MAIN_ENV"; then $CONDA env remove -y -n "$MAIN_ENV"; fi
  env_create_file "$MAIN_ENV" "$HERE/environment.yaml"
fi
MAIN_PREFIX="$(env_prefix "$MAIN_ENV")"
"$MAIN_PREFIX/bin/python" -m pip install --no-deps -e "$HERE"

# 2) classifier env (Tiara + Whokaryote; their pins conflict with the main env -> isolated, auto-found at run time)
if env_exists "$CLS_ENV" && [[ $FORCE -eq 0 ]]; then
  echo "[install] $CLS_ENV exists -> keeping it"
else
  if [[ $FORCE -eq 1 ]] && env_exists "$CLS_ENV"; then $CONDA env remove -y -n "$CLS_ENV"; fi
  $CONDA create -y -n "$CLS_ENV" "${CH[@]}" python=3.8 numpy=1.19.4 scikit-learn=1.0.2 pandas biopython joblib prodigal pip tqdm numba
  $CONDA install -y -n "$CLS_ENV" "${CH[@]}" --no-deps whokaryote=1.1.2
  CLS_PREFIX="$(env_prefix "$CLS_ENV")"
  "$CLS_PREFIX/bin/python" -m pip install --no-cache-dir tiara==1.0.3
fi

# 3) optional EukCC
if [[ $WITH_EUKCC -eq 1 ]]; then
  if env_exists "$EUKCC_ENV" && [[ $FORCE -eq 0 ]]; then echo "[install] $EUKCC_ENV exists -> keeping it"
  else $CONDA create -y -n "$EUKCC_ENV" "${CH[@]}" "eukcc>=2.1"; fi
  echo "[install] set eukcc_conda_env: $EUKCC_ENV and eukcc_db: <path to eukcc2 db> in your config"
fi

# 4) smoke test
CLS_PREFIX="$(env_prefix "$CLS_ENV")"
PATH="$MAIN_PREFIX/bin:$PATH:$CLS_PREFIX/bin"
for t in minimap2 samtools seqkit skani coverm barrnap nhmmer prodigal mmseqs tiara whokaryote.py; do
  if command -v "$t" >/dev/null 2>&1; then printf "  ok      %s\n" "$t"; else printf "  MISSING %s\n" "$t"; fi
done
"$MAIN_PREFIX/bin/python" -c "import torch, hdbscan, sklearn, Bio, yaml, faiss, psutil, hyphaesbin; print('  ok      python packages')"
echo "[install] done.  conda activate $MAIN_ENV && hyphaesbin --help"
