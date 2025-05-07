#!/bin/bash


#### Exp1 beta influence ###########
# copy cfg in root
cp ./config_runs/Exp1/*_config.yaml .

# run AUTOENCODIX
for config in ./Exp1*_config.yaml; do
	sbatch <<-EOT
	#!/bin/bash
	#SBATCH --job-name=autoencoder_Exp1
	#SBATCH --output=./reports/$(basename $config _config.yaml)/slurm_%a_%j.out
	#SBATCH --error=./reports/$(basename $config _config.yaml)/slurm_%a_%j.err
	#SBATCH --nodes=1
	#SBATCH --ntasks-per-node=1
	#SBATCH --time=04:00:00
	#SBATCH --mem-per-cpu=8G
	#SBATCH --cpus-per-task=6
	#SBATCH --gres=gpu:1
	if [ -z "$VIRTUAL_ENV" ];
	then 
		source venv-gallia/bin/activate
		echo $VIRTUAL_ENV
	fi
	export CUBLAS_WORKSPACE_CONFIG=:16:8
	make visualize RUN_ID=$(basename $config _config.yaml)
	nephelai upload-with-fs reports/$(basename $config _config.yaml)
	bash ./clean.sh -r $(basename $config _config.yaml) -k -d # Clean up and keep only reports folder
	exit 0
	EOT
done 

## get paper visualization
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=autoencoder_Exp1
#SBATCH --output=./reports/slurm_%a_%j.out
#SBATCH --error=./reports/slurm_%a_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=6

#SBATCH --dependency singleton
if [ -z "$VIRTUAL_ENV" ];
then 
	source venv-gallia/bin/activate
	echo $VIRTUAL_ENV
fi
python ./src/visualization/Exp1_visualization.py
nephelai upload-with-fs reports/paper-visualizations/Exp1
# clean up
rm ./Exp1*_config.yaml
exit 0
EOT
###################################

