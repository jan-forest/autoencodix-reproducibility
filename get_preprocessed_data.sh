#!/bin/bash
log_message() {
    echo "[$(date +"%Y-%m-%d_%H-%M-%S")] $1"
}
# check if the data is downloaded
if [ -f data/raw/data_downloaded.txt ]; then
    log_message "Data already downloaded"
    log_message "If you want to download the data again, delete the file data/raw/data_downloaded.txt"
else
    log_message "Data not downloaded"
    log_message "Downloading..."
    ## Get TCGA formatted data
    ## Get SCcortex formatted data
    ## Get Ontology files
    wget -nv -P data/raw https://zenodo.org/records/13691753/files/AutoencodixZenodoReproducibility.zip
    unzip -q data/raw/AutoencodixZenodoReproducibility.zip -d data/raw
    rm data/raw/AutoencodixZenodoReproducibility.zip

    log_message "Downloaded all data"
    log_message "Creating file that indicates that the data is downloaded"
    log_message "if you want to download the data again, delete the file data/raw/data_downloaded.txt"
    touch data/raw/data_downloaded.txt
fi
