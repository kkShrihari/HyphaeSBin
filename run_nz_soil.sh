#!/bin/bash
source /DATA_LUN/skkumara/Final/Zosteria_marina/miniforge3/etc/profile.d/conda.sh
conda activate hyphaesbin311
cd /home/skkumara/HyphaeSBin
python main.py \
  --scaffold /home/skkumara/NZ_assembly/coassembly/scaffolds.fasta \
  --outdir /home/skkumara/nz_soil_hyphaesbin \
  --reads-dir /DATA_LUN/skkumara/nz_soil/clean_reads \
  --reads Nr15:mgshot_S16996Nr15_R1.clean.fastq.gz:mgshot_S16996Nr15_R2.clean.fastq.gz \
  --reads Nr16:mgshot_S16996Nr16_R1.clean.fastq.gz:mgshot_S16996Nr16_R2.clean.fastq.gz \
  --reads Nr17:mgshot_S16996Nr17_R1.clean.fastq.gz:mgshot_S16996Nr17_R2.clean.fastq.gz \
  --reads Nr18:mgshot_S16996Nr18_R1.clean.fastq.gz:mgshot_S16996Nr18_R2.clean.fastq.gz \
  --reads Nr19:mgshot_S16996Nr19_R1.clean.fastq.gz:mgshot_S16996Nr19_R2.clean.fastq.gz \
  --reads Nr20:mgshot_S16996Nr20_R1.clean.fastq.gz:mgshot_S16996Nr20_R2.clean.fastq.gz \
  --reads Nr21:mgshot_S16996Nr21_R1.clean.fastq.gz:mgshot_S16996Nr21_R2.clean.fastq.gz \
  --reads Nr22:mgshot_S16996Nr22_R1.clean.fastq.gz:mgshot_S16996Nr22_R2.clean.fastq.gz \
  --config nz_soil.yaml \
  --threads 32 > /home/skkumara/nz_soil_hyphaesbin.log 2>&1
