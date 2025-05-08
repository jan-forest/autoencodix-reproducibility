sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=autoencoder_Exp6_TCGA_METH_RNA
#SBATCH --output=./reports/Exp6_TCGA_METH_RNA/slurm_%a_%j.out
#SBATCH --error=./reports/Exp6_TCGA_METH_RNA/slurm_%a_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=clara
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx2080ti:1

echo "Doing Experiment 6: TCGA_METH_RNA X-Modalix"
ml cuDNN
export __MODIN_AUTOIMPORT_PANDAS_=:1

./run_exp6.sh
if [ -z "$VIRTUAL_ENV" ];
then
	source venv-gallia/bin/activate
	echo $VIRTUAL_ENV
fi
nephelai upload-with-fs reports/Exp6_TCGA_METH_RNA
nephelai upload-with-fs reports/Exp6_TCGA_RNA_RNA
nephelai upload-with-fs reports/paper-visualizations/Exp6
# bash ./clean.sh -r Exp6_TCGA_METH_RNA,Exp6_TCGA_RNA_RNA -k -d # Clean up and keep only reports folder

exit 0
EOT
