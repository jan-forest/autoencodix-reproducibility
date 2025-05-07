#!/bin/bash


#### Exp2 AE comparison ###########
# copy cfg in root
# cp ./config_runs/Exp2/Exp2_TCGA_train_ontix_beta0.1_dim2_*_config.yaml .
cp ./config_runs/Exp2/*_config.yaml .

# run AUTOENCODIX
for config in ./Exp2*_config.yaml; do
	if [[ $config == *"_tune_"* ]]; then
	runtime=48
	else
	runtime=12
	fi
	sbatch <<-EOT
	#!/bin/bash
	#SBATCH --job-name=autoencoder_Exp2
	#SBATCH --output=./reports/$(basename $config _config.yaml)/slurm_%a_%j.out
	#SBATCH --error=./reports/$(basename $config _config.yaml)/slurm_%a_%j.err
	#SBATCH --nodes=1
	#SBATCH --ntasks-per-node=1
	#SBATCH --time="$runtime":00:00
	#SBATCH --mem-per-cpu=8G
	#SBATCH --cpus-per-task=6
	#SBATCH --gres=gpu:1
	if [ -z "$VIRTUAL_ENV" ];
	then 
		source venv-gallia/bin/activate
		echo $VIRTUAL_ENV
	fi
	export CUBLAS_WORKSPACE_CONFIG=:16:8
	make ml_task RUN_ID=$(basename $config _config.yaml)
	nephelai upload-with-fs reports/$(basename $config _config.yaml)
	bash ./clean.sh -r $(basename $config _config.yaml) -k -d # Clean up and keep only reports folder
	exit 0
	EOT
done 

## get paper visualization
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=autoencoder_Exp2
#SBATCH --output=./reports/slurm_%a_%j.out
#SBATCH --error=./reports/slurm_%a_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=6
#SBATCH --dependency singleton
if [ -z "$VIRTUAL_ENV" ];
then 
	source venv-gallia/bin/activate
	echo $VIRTUAL_ENV
fi
python ./src/visualization/Exp2_visualization.py
nephelai upload-with-fs reports/paper-visualizations/Exp2
# clean up
rm ./Exp2*_config.yaml
exit 0
EOT
###################################

