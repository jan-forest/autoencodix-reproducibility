#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=autoencoder_Exp1_Annealing
#SBATCH --output=./reports/Exp1_Annealing/slurm_%a_%j.out
#SBATCH --error=./reports/Exp1_Annealing/slurm_%a_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=clara
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx2080ti:1
#### Exp1 Beta influence ###########
echo "Doing Experiment 1: beta influence"

# copy cfg in root
cp ./config_runs/Exp1/Exp1_SC_Annealing_config.yaml .
cp ./config_runs/Exp1/Exp1_TCGA_Annealing_config.yaml .
# run AUTOENCODIX
if [ -z "$VIRTUAL_ENV" ];
then 
	source venv-gallia/bin/activate
	echo $VIRTUAL_ENV
fi
export CUBLAS_WORKSPACE_CONFIG=:16:8
make visualize RUN_ID=Exp1_SC_Annealing
make visualize RUN_ID=Exp1_TCGA_Annealing

# get paper visualization
mkdir -p ./reports/paper-visualizations/Exp1
python src/visualization/Exp1_visualization.py 


## Transfer of paper-visualizations/Exp1 to NextCloud
nephelai upload-with-fs reports/Exp1_SC_Annealing
nephelai upload-with-fs reports/Exp1_TCGA_Annealing
nephelai upload-with-fs reports/paper-visualizations/Exp1

# clean up
bash ./clean.sh -r Exp1_SC_Annealing,Exp1_TCGA_Annealing -d -k # Clean up and keep only reports folder
rm ./Exp1_*_Annealing_config.yaml
###################################

exit 0
EOT