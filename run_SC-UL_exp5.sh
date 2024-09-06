sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=autoencoder_Exp5_TCGA_MNIST
#SBATCH --output=./reports/Exp5_TCGA_MNIST/slurm_%a_%j.out
#SBATCH --error=./reports/Exp5_TCGA_MNIST/slurm_%a_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=clara
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx2080ti:1

echo "Doing Experiment 5: TCGA MNIST X-Modalix"
ml cuDNN
export __MODIN_AUTOIMPORT_PANDAS_=:1

./run_exp5.sh
if [ -z "$VIRTUAL_ENV" ];
then 
	source venv-gallia/bin/activate
	echo $VIRTUAL_ENV
fi
nephelai upload-with-fs reports/Exp5_TCGA_MNIST
nephelai upload-with-fs reports/Exp5_TCGA_MNISTImgImg
nephelai upload-with-fs reports/paper-visualizations/Exp5
bash ./clean.sh -r Exp5_TCGA_MNIST,Exp5_TCGA_MNISTImgImg -k -d # Clean up and keep only reports folder

exit 0
EOT