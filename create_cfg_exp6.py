
from itertools import combinations
import yaml
import numpy as np
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
cfg["LATENT_DIM_FIXED"] = 12
cfg["BATCH_SIZE"] = 512
cfg['LR_FIXED'] = 0.0005
cfg['EPOCHS'] = 1000
cfg['RECONSTR_LOSS'] = "MSE"
cfg['VAE_LOSS'] = "KL"
cfg['PREDICT_SPLIT'] = "all"
cfg['TRAIN_TYPE'] = "train"
cfg['MODEL_TYPE'] = "x-modalix"
cfg['BETA'] = 0.2
cfg['GAMMA'] = 10
cfg['DROP_P'] = 0.5

cfg['SPLIT'] = [0.7, 0.2, 0.1]
# X-modalix specific
cfg["DELTA_PAIR"] = 0
cfg["DELTA_CLASS"] = 10
cfg["ANNEAL_PRETRAINING"] = True
cfg["TRANSLATE"] = "RNA_to_METH"


cfg['K_FILTER'] = 4000
cfg['DATA_TYPE'] = dict()
cfg['DATA_TYPE']['ANNO'] = dict()
cfg['DATA_TYPE']['ANNO']['TYPE'] = "ANNOTATION"
cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "data_clinical_formatted.parquet"
dm = ["RNA", "METH"]
for m in dm:								
	cfg['DATA_TYPE'][m] = dict()
	cfg['DATA_TYPE'][m]['SCALING'] = "MinMax"
	cfg['DATA_TYPE'][m]['FILTERING'] = "Var"
	cfg['DATA_TYPE'][m]['TYPE'] = "NUMERIC"

	cfg['DATA_TYPE'][m]['FILE_RAW'] = "data_methylation_per_gene_formatted.parquet" 
	if m == "RNA":
		cfg['DATA_TYPE'][m]['FILE_RAW'] = "data_mrna_seq_v2_rsem_formatted.parquet" 
cfg['CLINIC_PARAM'] = ["CANCER_TYPE"]
cfg["CLASS_PARAM"] = "CANCER_TYPE"
cfg['DIM_RED_METH'] = "UMAP"

run_id = cfg_prefix + "_TCGA_METH_RNA"
with open(cfg_folder+'/'+run_id +"_config.yaml", 'w') as file:
	yaml.dump(cfg, file)
print("Config created for Experiment 6")

### cfg for x-modalix examples "Experiment 6" extra ImgImg ###

cfg_prefix = "Exp6"
cfg = dict()

cfg_folder = config_save_root + cfg_prefix
Path(cfg_folder).mkdir(parents=False, exist_ok=True)

cfg['FIX_RANDOMNESS'] = 'all'
cfg['GLOBAL_SEED'] = 42
cfg["CHECKPT_PLOT"] = True
cfg["LATENT_DIM_FIXED"] = 12
cfg["BATCH_SIZE"] = 512
cfg['LR_FIXED'] = 0.0005
cfg['EPOCHS'] = 1000
cfg['RECONSTR_LOSS'] = "MSE"
cfg['VAE_LOSS'] = "KL"
cfg['PREDICT_SPLIT'] = "all"
cfg['TRAIN_TYPE'] = "train"
cfg['MODEL_TYPE'] = "x-modalix"
cfg['BETA'] = 0.2
cfg["GAMMA"] = 10
cfg['DROP_P'] = 0.5

# X-modalix specific
cfg["DELTA_PAIR"] = 0
cfg["DELTA_CLASS"] = 10
cfg["ANNEAL_PRETRAINING"] = True
cfg["PRETRAIN_EPOCHS"] = 0
cfg["TRANSLATE"] = "METH_to_METH"


cfg['K_FILTER'] = 4000
cfg['DATA_TYPE'] = dict()
cfg['DATA_TYPE']['ANNO'] = dict()
cfg['DATA_TYPE']['ANNO']['TYPE'] = "ANNOTATION"
cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "data_clinical_formatted.parquet"
dm = ["IMG"]
for m in dm:								
	cfg['DATA_TYPE'][m] = dict()
	cfg['DATA_TYPE'][m] = dict()
	cfg['DATA_TYPE'][m]['SCALING'] = "MinMax"
	cfg['DATA_TYPE'][m]['FILTERING'] = "Var"
	cfg['DATA_TYPE'][m]['TYPE'] = "NUMERIC"

	cfg['DATA_TYPE'][m]['FILE_RAW'] = "data_methylation_per_gene_formatted.parquet" 
cfg['CLINIC_PARAM'] = [ "CANCER_TYPE"]
cfg["CLASS_PARAM"] = "CANCER_TYPE"
cfg['DIM_RED_METH'] = "UMAP"

run_id = cfg_prefix + "_TCGA_METH_METH"
with open(cfg_folder+'/'+run_id +"_config.yaml", 'w') as file:
	yaml.dump(cfg, file)