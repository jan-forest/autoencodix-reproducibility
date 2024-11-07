#!/bin/bash

mkdir -p data/raw
if [ -f "data/raw/data_methylation_per_gene_formatted.parquet" ]; then
	echo "TCGA data detected in data/raw. Download will be skipped"
else
# https://cloud.scadsai.uni-leipzig.de/index.php/s/2qMDfTzaPM4GXyZ
	wget -nv -P data/raw https://cloud.scadsai.uni-leipzig.de/index.php/s/2qMDfTzaPM4GXyZ/download/TCGA_omics.zip
	unzip -o data/raw/TCGA_omics.zip -d data/raw
	rm data/raw/TCGA_omics.zip
fi



make ml_task RUN_ID=TCGAexample
make ml_task RUN_ID=TuningExample