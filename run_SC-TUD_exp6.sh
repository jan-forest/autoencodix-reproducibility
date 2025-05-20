sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=autoencoder_Exp6_TCGA_METH_RNA
#SBATCH --output=./reports/Exp6_TCGA_METH_RNA/slurm_%a_%j.out
#SBATCH --error=./reports/Exp6_TCGA_METH_RNA/slurm_%a_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8GB

echo "Doing Experiment 6: TCGA METH -> RNA X-Modalix"

export CUBLAS_WORKSPACE_CONFIG=:16:8

./run_exp5.sh
if [ -z "$VIRTUAL_ENV" ];
then 
	source venv-gallia/bin/activate
	echo $VIRTUAL_ENV
fi
nephelai upload-with-fs reports/Exp6_TCGA_METH_RNA
nephelai upload-with-fs reports/Expt6_TCGA_RNA_RNA
nephelai upload-with-fs reports/Expt6_TCGA_VARIX
nephelai upload-with-fs reports/paper-visualizations/Exp6

bash ./clean.sh -r Exp6_TCGA_METH_RNA,Exp6_TCGA_RNA_RNA, Exp6_TCGA_VARIX -k -d # Clean up and keep only reports folder

exit 0
EOT
