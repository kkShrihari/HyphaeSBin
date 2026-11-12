#!/bin/bash
source /DATA_LUN/skkumara/Final/Zosteria_marina/miniforge3/etc/profile.d/conda.sh
EUKDB=/DATA_LUN/skkumara/databases/eukcc2_db_ver_1.1
BDL=/DATA_LUN/skkumara/databases/busco_downloads
LIN=$BDL/lineages/eukaryota_odb10
LOG=/home/skkumara/qc_zostera_bins.log
echo "$(date) start" > $LOG

suffix () { ls $1 | grep -oE '\.(fa|fasta|fna)$' | sort | uniq -c | sort -rn | head -1 | awk '{print $2}'; }

for RUN in zostera_2000 zostera_1000; do
  for TAG in tetnfcov tnfcov; do
    C=/home/skkumara/$RUN/runs/$TAG/clustering
    B=$C/clusters_deduplicated
    if [ ! -f $C/cluster_summary.tsv ] || [ -z "$(ls $B 2>/dev/null)" ]; then
      echo "$(date) SKIP $RUN/$TAG: run not finished (no bins yet)" >> $LOG; continue; fi
    SUF=$(suffix $B); N=$(ls $B/*$SUF | wc -l)

    # ---- EukCC 2 (strict)
    conda activate eukcc22
    V=$(eukcc --version 2>&1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
    if [[ "$V" == 2.* ]]; then
      mkdir -p $C/eukcc2
      echo "$(date) EukCC $V  $RUN/$TAG  $N bins" >> $LOG
      TMPDIR=/tmp eukcc folder --out $C/eukcc2 --db $EUKDB --threads 32 --suffix $SUF $B > $C/eukcc2/eukcc.log 2>&1
    else
      echo "$(date) ABORT EukCC: version '$V' is not 2.x -- EukCC not run" >> $LOG
    fi
    conda deactivate

    # ---- BUSCO eukaryota_odb10 (255 markers)
    conda activate busco
    echo "$(date) BUSCO $RUN/$TAG  $N bins" >> $LOG
    busco -i $B -m genome -l $LIN --offline --download_path $BDL -c 32 \
          --out_path $C -o busco_odb10 -f > $C/busco_odb10.log 2>&1
    conda deactivate
  done
done
echo "$(date) done" >> $LOG
