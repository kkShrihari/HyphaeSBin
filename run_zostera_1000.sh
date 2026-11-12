#!/bin/bash
source /DATA_LUN/skkumara/Final/Zosteria_marina/miniforge3/etc/profile.d/conda.sh
conda activate hyphaesbin311
cd /home/skkumara/HyphaeSBin
python main.py \
  --scaffold /DATA_LUN/skkumara/Final/Zosteria_marina/final_assembly/scaffolds.fasta \
  --outdir /home/skkumara/zostera_1000 \
  --reads-dir /DATA_LUN/skkumara/Final/Zosteria_marina/clean_reads \
  --reads SRR29999183:SRR29999183_R1.clean.fastq.gz:SRR29999183_R2.clean.fastq.gz \
  --reads SRR29999184:SRR29999184_R1.clean.fastq.gz:SRR29999184_R2.clean.fastq.gz \
  --reads SRR29999185:SRR29999185_R1.clean.fastq.gz:SRR29999185_R2.clean.fastq.gz \
  --config zostera_marina_1000bp.yaml \
  --threads 32 > /home/skkumara/zostera_1000.log 2>&1
