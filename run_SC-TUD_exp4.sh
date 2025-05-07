sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=autoencoder_Exp4_Celegans
#SBATCH --output=./reports/Exp4_Celegans_TF/slurm_%a_%j.out
#SBATCH --error=./reports/Exp4_Celegans_TF/slurm_%a_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8GB

echo "Doing Experiment 4: Celegans X-Modalix"

export CUBLAS_WORKSPACE_CONFIG=:16:8
./run_exp4.sh
if [ -z "$VIRTUAL_ENV" ];
then 
	source venv-gallia/bin/activate
	echo $VIRTUAL_ENV
fi
nephelai upload-with-fs reports/Exp4_Celegans_TF
nephelai upload-with-fs reports/Exp4_Celegans_TFImgImg
nephelai upload-with-fs reports/paper-visualizations/Exp4
bash ./clean.sh -r Exp4_Celegans_TF,Exp4_Celegans_TFImgImg -k -d # Clean up and keep only reports folder

exit 0
EOT
