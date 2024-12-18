#!/bin/bash

#### Getting AUTOENCODIX ready ####
# # ## clone repo before running script
# git clone https://github.com/jan-forest/autoencodix-reproducibility.git
# cd ./autoencodix-reproducibility/ 
# create or copy a .env for nephelai
# cp ../../Gitlab/autoencodix-reproducibility/.env ./
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=autoencoder_build
#SBATCH --output=./reports/slurm_%a_%j.out
#SBATCH --error=./reports/slurm_%a_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=paul
#SBATCH --time=2:00:00
#SBATCH --mem=32G

## create env
#!/bin/bash
if command -v curl &> /dev/null; then
    echo "curl is installed"
    curl -LsSf https://astral.sh/uv/0.5.9/install.sh | sh
    
elif command -v wget &> /dev/null; then
    echo "wget is installed"
    wget -qO- https://astral.sh/uv/0.5.9/install.sh | sh
else
    echo "Please install curl or wget"
    exit 1
fi
make create_environment
## activate env
source venv-gallia/bin/activate 
## make requirements
make requirements
uv pip install nephelai ## For Nextcloud transfer 
####################################

#### Prepare Runs ##################
## get preprocessed data
bash ./get_preprocessed_data.sh 
## create configs
mkdir ./config_runs
python ./create_cfg.py 
# echo "Creating configs"
# python ./create_cfg_test.py

mkdir ./reports/paper-visualizations
mkdir ./reports/paper-visualizations/Exp1
mkdir ./reports/paper-visualizations/Exp2
mkdir ./reports/paper-visualizations/Exp3
mkdir ./reports/paper-visualizations/Exp4
mkdir ./reports/paper-visualizations/Exp5

exit 0
EOT
