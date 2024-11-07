sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=autoencoder_Exp4_Celegans
#SBATCH --output=./reports/Exp4_Celegans_TF/slurm_%a_%j.out
#SBATCH --error=./reports/Exp4_Celegans_TF/slurm_%a_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=clara
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx2080ti:1

echo "Doing Experiment 4: Celegans X-Modalix"
ml cuDNN
export __MODIN_AUTOIMPORT_PANDAS_=:1

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