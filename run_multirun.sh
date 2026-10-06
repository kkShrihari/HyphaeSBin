#!/bin/bash
# Launch the CAMISIM multi-run (all 7 modality runs + best_of_it) in a NEW output folder, reusing the cached steps 2-6 of the
# earlier run (steps 7+ recompute with preprocessing v4). Edit the 4 variables below if your paths differ.
set -e
OLD=/DATA_LUN/skkumara/run/hyphaesbin_results/FINAL/tetnfcov
NEW=${OLD}_multirun
PY=/DATA_LUN/skkumara/Final/Zosteria_marina/envs/classify/bin/python
YAML_SRC=/DATA_LUN/skkumara/run/HyphaeSBin/camisim_final_tetnfcov_multirun.yaml
REPO=/DATA_LUN/skkumara/run/HyphaeSBin
cd $REPO
mkdir -p $NEW/preprocessing/checkpoints
for d in 02_merged 03_stats 04_dedup 05_length_prefilter; do [ -e $NEW/preprocessing/$d ] || ln -s $OLD/preprocessing/$d $NEW/preprocessing/$d; done
[ -e $NEW/preprocessing/06_classification ] || cp -r $OLD/preprocessing/06_classification $NEW/preprocessing/
for s in step2_merge step3_stats step4_dedup step5_length_prefilter; do cp -n $OLD/preprocessing/checkpoints/$s.done $NEW/preprocessing/checkpoints/; done
export HYPHAESBIN_REUSE_CLASSIFICATION=1      # only valid because 06_classification came from this same input and parameters
sed -e 's/^all_runs: false/all_runs: true/' -e 's/^best_of_it: false/best_of_it: true/' $YAML_SRC > $NEW/multirun.yaml
READS=""
while read s r1 r2; do if [ "$s" != "sample" ]; then READS="$READS --reads $s:$r1:$r2"; fi; done < $OLD/samples.tsv
/usr/bin/time -v -o "$NEW/pipeline_time.txt" "$PY" main.py --scaffold /DATA_LUN/skkumara/Final/camisim/scaffolds_by_sample $READS --outdir $NEW --config $NEW/multirun.yaml > $NEW/run_multirun.log 2>&1
