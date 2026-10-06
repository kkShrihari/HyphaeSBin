#!/usr/bin/env python3
"""Pre-flight: can the pipeline launch EukCC from THIS shell?  (run it before a long job)
   /path/to/envs/classify/bin/python check_eukcc.py camisim_final_tetnfcov_multirun.yaml
Prints which launcher works (conda run -n / conda run -p / the sibling env's own eukcc / $HYPHAESBIN_EUKCC / the yaml's
eukcc_executable) or what was tried. If it says NOT RUNNABLE, set eukcc_executable in the yaml (or export HYPHAESBIN_EUKCC)
to the absolute path of the eukcc binary of your EukCC environment (find it with:  conda activate eukcc22; which eukcc)."""
import os, sys, yaml
sys.path.insert(0, os.getcwd())
import hyphaesbin.clustering.clustering as clu
cfgd = yaml.safe_load(open(sys.argv[1])) or {}
if cfgd.get("eukcc_executable"):
    os.environ["HYPHAESBIN_EUKCC"] = str(cfgd["eukcc_executable"])
cfg = clu.ClusteringConfig.from_dict(cfgd)
print(f"eukcc_conda_env = {cfg.eukcc_conda_env} | eukcc_db = {cfg.eukcc_db} (exists: {os.path.isdir(cfg.eukcc_db)})")
print(f"conda found: {clu._find_conda()}   sibling env prefix: {clu._eukcc_env_prefix(cfg.eukcc_conda_env)}")
argv = clu._eukcc_argv(cfg)
print("RESULT:", ("EukCC launcher = " + " ".join(argv)) if argv else "EukCC NOT RUNNABLE from this shell (see the FLAG line above)")
sys.exit(0 if argv else 1)
