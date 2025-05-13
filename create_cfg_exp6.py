
import yaml
from pathlib import Path
config_save_root = "./config_runs/"
Path(config_save_root).mkdir(parents=False, exist_ok=True)



#################################################
### cfg for x-modalix examples "Experiment 6" ###

cfg_prefix = "Exp6"
cfg = dict()

cfg_folder = config_save_root + cfg_prefix
Path(cfg_folder).mkdir(parents=False, exist_ok=True)

cfg['FIX_RANDOMNESS'] = 'all'
cfg['GLOBAL_SEED'] = 42
cfg["CHECKPT_PLOT"] = True
cfg["SUBTYPES"] = ["brca", "lusc", "luad", "ov", "coadread", "ucec", "ucs"]
cfg["LATENT_DIM_FIXED"] = 12
cfg["BATCH_SIZE"] = 512

cfg["RECON_SAVE"] = True
cfg['LR_FIXED'] = 0.0005
cfg['EPOCHS'] = 500
cfg['RECONSTR_LOSS'] = "MSE"
cfg['VAE_LOSS'] = "KL"
cfg['PREDICT_SPLIT'] = "all"
cfg['TRAIN_TYPE'] = "train"
cfg['MODEL_TYPE'] = "x-modalix"
cfg['BETA'] = 0.05
cfg['GAMMA'] = 10
cfg['DROP_P'] = 0.2

cfg['SPLIT'] = [0.7, 0.2, 0.1]
# X-modalix specific
cfg["DELTA_PAIR"] = 0
cfg["DELTA_CLASS"] = 10
cfg["ANNEAL_PRETRAINING"] = False
cfg["TRANSLATE"] = "METH_to_RNA"
cfg["PRETRAIN_TARGET_MODALITY"] = ""

cfg['K_FILTER'] = 4000
cfg['DATA_TYPE'] = dict()
cfg['DATA_TYPE']['ANNO'] = dict()
cfg['DATA_TYPE']['ANNO']['TYPE'] = "ANNOTATION"
cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "combined_clin_formatted.parquet"
# cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "data_clinical_formatted.parquet"
dm = ["RNA", "METH"]
for m in dm:
	cfg['DATA_TYPE'][m] = dict()
	cfg['DATA_TYPE'][m]['SCALING'] = "MinMax"
	cfg['DATA_TYPE'][m]['FILTERING'] = "Var"
	cfg['DATA_TYPE'][m]['TYPE'] = "NUMERIC"

	cfg['DATA_TYPE'][m]['FILE_RAW'] = "combined_meth_formatted.parquet"
	# cfg['DATA_TYPE'][m]['FILE_RAW'] = "data_methylation_per_gene_formatted.parquet"
	if m == "RNA":
		cfg['DATA_TYPE'][m]['FILE_RAW'] = "combined_rnaseq_formatted.parquet"
		# cfg['DATA_TYPE'][m]['FILE_RAW'] = "data_mrna_seq_v2_rsem_formatted.parquet"
cfg['CLINIC_PARAM'] = [
	"CANCER_TYPE", 
	"CANCER_TYPE_ACRONYM", 
	"TMB_NONSYNONYMOUS", 
	"AGE", 
	"SEX", 
	"AJCC_PATHOLOGIC_TUMOR_STAGE_SHORT", 
	# "AJCC_PATHOLOGIC_TUMOR_STAGE",
	"OS_STATUS", 
	"OS_MONTHS", 
	"DFS_STATUS", 
	"PFS_STATUS", 
	"MSI_SCORE_MANTIS", 
	"ANEUPLOIDY_SCORE"
]


cfg["CLASS_PARAM"] = "CANCER_TYPE"
# cfg["CLASS_PARAM"] = None
cfg['DIM_RED_METH'] = "PCA"
cfg["ML_TYPE"] = dict()
cfg["ML_TYPE"] = "Auto-detect"
cfg["ML_ALG"] = ["Linear", "RF"]
cfg["ML_SPLIT"] =  "use-split" # OPTIONAL "use-split" or "CV-on-all-data"
cfg["ML_TASKS"]  = ["Latent", "PCA", "UMAP", "RandomFeature"]



run_id = cfg_prefix + "_TCGA_METH_RNA"
with open(cfg_folder+'/'+run_id +"_config.yaml", 'w') as file:
	yaml.dump(cfg, file)
print("Config created for Experiment 6")

 ## Exp 6 RNA to RNA
 # -------------------------------------------------------
 # --------------------------------------------------------
cfg_prefix = "Exp6"
cfg = dict()

cfg["SUBTYPES"] = ["brca", "lusc", "luad", "ov", "coadread", "ucec", "ucs"]
cfg["ML_TYPE"] = dict()
cfg["ML_TYPE"] = "Auto-detect"

cfg["ML_ALG"] = ["Linear", "RF"]
cfg["ML_SPLIT"] =  "use-split" # OPTIONAL "use-split" or "CV-on-all-data"
cfg["ML_TASKS"]  = ["Latent", "PCA", "UMAP", "RandomFeature"]


cfg_folder = config_save_root + cfg_prefix
Path(cfg_folder).mkdir(parents=False, exist_ok=True)

cfg['FIX_RANDOMNESS'] = 'all'
cfg['GLOBAL_SEED'] = 42
cfg["CHECKPT_PLOT"] = True
cfg["LATENT_DIM_FIXED"] = 12
cfg["BATCH_SIZE"] = 512
cfg['LR_FIXED'] = 0.0005
cfg['EPOCHS'] = 500
cfg['RECONSTR_LOSS'] = "MSE"
cfg['VAE_LOSS'] = "KL"
cfg['PREDICT_SPLIT'] = "all"
cfg['TRAIN_TYPE'] = "train"
cfg['MODEL_TYPE'] = "x-modalix"
cfg['BETA'] = 0.05
cfg["GAMMA"] = 10
cfg['DROP_P'] = 0.2

# X-modalix specific
cfg["DELTA_PAIR"] = 0

cfg["PRETRAIN_TARGET_MODALITY"] = ""
cfg["DELTA_CLASS"] = 10
cfg["ANNEAL_PRETRAINING"] = False
cfg["PRETRAIN_EPOCHS"] = 0
cfg["TRANSLATE"] = "RNA_to_RNA"
cfg["RECON_SAVE"] = True
cfg["PLOT_INDPUT2D"] = True


cfg['K_FILTER'] = 4000
cfg['DATA_TYPE'] = dict()
cfg['DATA_TYPE']['ANNO'] = dict()
cfg['DATA_TYPE']['ANNO']['TYPE'] = "ANNOTATION"
cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "combined_clin_formatted.parquet"
# cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "data_clinical_formatted.parquet"
dm = ["RNA"]
for m in dm:
	cfg['DATA_TYPE'][m] = dict()
	cfg['DATA_TYPE'][m] = dict()
	cfg['DATA_TYPE'][m]['SCALING'] = "MinMax"
	cfg['DATA_TYPE'][m]['FILTERING'] = "Var"
	cfg['DATA_TYPE'][m]['TYPE'] = "NUMERIC"
	cfg['DATA_TYPE'][m]['FILE_RAW'] = "combined_rnaseq_formatted.parquet"
	# cfg['DATA_TYPE'][m]['FILE_RAW'] = "data_mrna_seq_v2_rsem_formatted.parquet"
cfg['CLINIC_PARAM'] = [
	"CANCER_TYPE", 
	"CANCER_TYPE_ACRONYM", 
	"TMB_NONSYNONYMOUS", 
	"AGE", 
	"SEX", 
	"AJCC_PATHOLOGIC_TUMOR_STAGE_SHORT", 
	# "AJCC_PATHOLOGIC_TUMOR_STAGE",
	"OS_STATUS", 
	"OS_MONTHS", 
	"DFS_STATUS", 
	"PFS_STATUS", 
	"MSI_SCORE_MANTIS", 
	"ANEUPLOIDY_SCORE"
]

cfg["CLASS_PARAM"] = "CANCER_TYPE"
# cfg["CLASS_PARAM"] = None

cfg['DIM_RED_METH'] = "PCA"
cfg["PLOT_INDPUT2D"] = True



run_id = cfg_prefix + "_TCGA_RNA_RNA"
with open(cfg_folder+'/'+run_id +"_config.yaml", 'w') as file:
	yaml.dump(cfg, file)


# Exp 6 Varix compare run
# -------------------------------------------------------
# --------------------------------------------------------

cfg_prefix = "Exp6"
cfg = dict()

cfg["SUBTYPES"] = ["brca", "lusc", "luad", "ov", "coadread", "ucec", "ucs"]
cfg_folder = config_save_root + cfg_prefix
Path(cfg_folder).mkdir(parents=False, exist_ok=True)

cfg['FIX_RANDOMNESS'] = 'all'
cfg['GLOBAL_SEED'] = 42
cfg["CHECKPT_PLOT"] = True
cfg["LATENT_DIM_FIXED"] = 12
cfg["BATCH_SIZE"] = 512
cfg['LR_FIXED'] = 0.0005
cfg['EPOCHS'] = 500
cfg['RECONSTR_LOSS'] = "MSE"
cfg['VAE_LOSS'] = "KL"
cfg['PREDICT_SPLIT'] = "all"
cfg['TRAIN_TYPE'] = "train"
cfg['MODEL_TYPE'] = "varix"
cfg['BETA'] = 0.05
cfg['DROP_P'] = 0.2


cfg["RECON_SAVE"] = True
cfg["PLOT_INDPUT2D"] = True


cfg['K_FILTER'] = 4000
cfg['DATA_TYPE'] = dict()
cfg['DATA_TYPE']['ANNO'] = dict()
cfg['DATA_TYPE']['ANNO']['TYPE'] = "ANNOTATION"
cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "combined_clin_formatted.parquet"
# cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "data_clinical_formatted.parquet"
dm = ["METH", "RNA"]
for m in dm:
	cfg['DATA_TYPE'][m] = dict()
	cfg['DATA_TYPE'][m] = dict()
	cfg['DATA_TYPE'][m]['SCALING'] = "MinMax"
	cfg['DATA_TYPE'][m]['FILTERING'] = "Var"
	cfg['DATA_TYPE'][m]['TYPE'] = "NUMERIC"
	cfg['DATA_TYPE'][m]['FILE_RAW'] = "combined_meth_formatted.parquet"
	# cfg['DATA_TYPE'][m]['FILE_RAW'] = "data_methylation_per_gene_formatted.parquet"
	if m == "RNA":
		cfg['DATA_TYPE'][m]['FILE_RAW'] = "combined_rnaseq_formatted.parquet"
		# cfg['DATA_TYPE'][m]['FILE_RAW'] = "data_mrna_seq_v2_rsem_formatted.parquet"
cfg['CLINIC_PARAM'] = [
	"CANCER_TYPE", 
	"CANCER_TYPE_ACRONYM", 
	"TMB_NONSYNONYMOUS", 
	"AGE", 
	"SEX", 
	"AJCC_PATHOLOGIC_TUMOR_STAGE_SHORT", 
	# "AJCC_PATHOLOGIC_TUMOR_STAGE",
	"OS_STATUS", 
	"OS_MONTHS", 
	"DFS_STATUS", 
	"PFS_STATUS", 
	"MSI_SCORE_MANTIS", 
	"ANEUPLOIDY_SCORE"
]
cfg["CLASS_PARAM"] = "CANCER_TYPE"
# cfg["CLASS_PARAM"] = None
cfg['DIM_RED_METH'] = "PCA"

cfg["PLOT_INDPUT2D"] = True
cfg["ML_TYPE"] = dict()
cfg["ML_TYPE"] = "Auto-detect"
cfg["ML_ALG"] = ["Linear", "RF"]
cfg["ML_SPLIT"] =  "use-split" # OPTIONAL "use-split" or "CV-on-all-data"
cfg["ML_TASKS"]  = ["Latent", "PCA", "UMAP", "RandomFeature"]




run_id = cfg_prefix + "_TCGA_VARIX"
with open(cfg_folder+'/'+run_id +"_config.yaml", 'w') as file:
	yaml.dump(cfg, file)

