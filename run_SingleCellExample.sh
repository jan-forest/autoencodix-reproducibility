#!/bin/bash

mkdir -p data/raw
if [ -f "data/raw/scATAC_human_cortex_formatted.parquet" ]; then
	echo "Single Cell human cortex data detected in data/raw. Download will be skipped"
else
	wget -nv -P data/raw https://cloud.scadsai.uni-leipzig.de/index.php/s/3d9HTjxcwsgGKpQ/download/Single-Cell-Cortex.zip
	unzip -o data/raw/Single-Cell-Cortex.zip -d data/raw
	rm data/raw/Single-Cell-Cortex.zip
fi



make ml_task RUN_ID=scExample
