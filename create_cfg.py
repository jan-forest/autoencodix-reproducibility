from itertools import combinations
import yaml
import numpy as np
from pathlib import Path
config_save_root = "./config_runs/"
Path(config_save_root).mkdir(parents=False, exist_ok=True)

#############################################
### cfg for beta-influence "Experiment 1" ###

cfg_prefix = "Exp1"
cfg = dict()

cfg_folder = config_save_root + cfg_prefix
Path(cfg_folder).mkdir(parents=False, exist_ok=True)

cfg['FIX_RANDOMNESS'] = 'all'
cfg['GLOBAL_SEED'] = 42
cfg["CHECKPT_PLOT"] = True
cfg["FIX_XY_LIM"] = [[-10,150], [-10,150]] # Special Param only for Exp1, not in Repo
cfg["ANNEAL"] = "5phase-constant"
cfg["LATENT_DIM_FIXED"] = 2
cfg["BATCH_SIZE"] = 256
cfg['LR_FIXED'] = 0.0001
cfg['EPOCHS'] = 2000
cfg['RECONSTR_LOSS'] = "MSE"
cfg['VAE_LOSS'] = "KL"
cfg['PREDICT_SPLIT'] = "all"
cfg['TRAIN_TYPE'] = "train"
cfg['MODEL_TYPE'] = "varix"
cfg['BETA'] = 10
cfg['SPLIT'] = [0.6, 0.2, 0.2]

### sc-Cortex
cfg['K_FILTER'] = 1500
cfg['DATA_TYPE'] = dict()
cfg['DATA_TYPE']['ANNO'] = dict()
cfg['DATA_TYPE']['ANNO']['TYPE'] = "ANNOTATION"
cfg['CLINIC_PARAM'] = [
				"author_cell_type" 
			]

cfg['DIM_RED_METH'] = "UMAP"

dm = ["RNA", "METH"]
for m in dm:								
	cfg['DATA_TYPE'][m] = dict()
	cfg['DATA_TYPE'][m]['SCALING'] = "Standard"
	cfg['DATA_TYPE'][m]['TYPE'] = "NUMERIC"
	cfg['DATA_TYPE'][m]['FILTERING'] = "Var"
	cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "scATAC_human_cortex_clinical_formatted.parquet"
	if m == "RNA":
		cfg['DATA_TYPE'][m]['FILE_RAW'] = "scRNA_human_cortex_formatted.parquet"
	if m == "METH":
		cfg['DATA_TYPE'][m]['FILE_RAW'] = "scATAC_human_cortex_formatted.parquet"


run_id = cfg_prefix + "_SC_Annealing"
with open(cfg_folder+'/'+run_id +"_config.yaml", 'w') as file:
	yaml.dump(cfg, file)
### TCGA
cfg['K_FILTER'] = 2000
cfg['DATA_TYPE'] = dict()
cfg['DATA_TYPE']['ANNO'] = dict()
cfg['DATA_TYPE']['ANNO']['TYPE'] = "ANNOTATION"
cfg['CLINIC_PARAM'] = [
				"CANCER_TYPE_ACRONYM" 
			]

cfg['DIM_RED_METH'] = "UMAP"

dm = ["RNA", "METH", "MUT"]
for m in dm:								
	cfg['DATA_TYPE'][m] = dict()
	cfg['DATA_TYPE'][m]['SCALING'] = "Standard"
	cfg['DATA_TYPE'][m]['TYPE'] = "NUMERIC"
	cfg['DATA_TYPE'][m]['FILTERING'] = "Var"
	cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "data_clinical_formatted.parquet"
	if m == "RNA":
		cfg['DATA_TYPE'][m]['FILE_RAW'] = "data_mrna_seq_v2_rsem_formatted.parquet"
	if m == "METH":
		cfg['DATA_TYPE'][m]['FILE_RAW'] = "data_methylation_per_gene_formatted.parquet"
	if m == "MUT":
		cfg['DATA_TYPE'][m]['FILE_RAW'] = "data_combi_MUT_CNA_formatted.parquet"


run_id = cfg_prefix + "_TCGA_Annealing"
with open(cfg_folder+'/'+run_id +"_config.yaml", 'w') as file:
	yaml.dump(cfg, file)
print("Config created for Experiment 1")

#####################################################
### cfg for architecture benchmark "Experiment 2" ###

cfg_prefix = "Exp2"

cfg = dict()

cfg_folder = config_save_root + cfg_prefix
Path(cfg_folder).mkdir(parents=False, exist_ok=True)

## Variable cfg params and datasets
dataset = ['TCGA','SC']
arch = ['vanillix',	'varix',	'stackix',	'ontix']
dm = ['RNA','METH','MUT']
beta = [1,	0.1,	0.01]
dim = [2,	8,	29]
train_type = ['train',	'tune']

## Stable params not dependent on other and not standard internal config
# model and training
cfg['FIX_RANDOMNESS'] = 'all'
cfg['GLOBAL_SEED'] = 42
cfg['LR_FIXED'] = 0.0001
cfg['EPOCHS'] = 1000
cfg['BATCH_SIZE'] = 256
cfg['RECONSTR_LOSS'] = "MSE"
cfg['VAE_LOSS'] = "KL"
cfg['PREDICT_SPLIT'] = "all"
## data
cfg['FILE_ONT_LVL2'] = "full_ont_lvl2_reactome.txt"
cfg['SPLIT'] = [0.6, 0.2, 0.2]
## Viz and ML
cfg['DIM_RED_METH'] = "UMAP"
cfg['ML_TYPE'] = "Auto-detect"
cfg['ML_ALG'] = [
	'Linear',
	'RF',
	'SVM'
]
cfg['ML_SPLIT'] = "use-split"
cfg['CV'] =5 # unused if use-split
cfg['ML_TASKS'] = [
	'Latent',
	'UMAP',
	'PCA',
	'RandomFeature'
]

count = 0
for dat in dataset:
	if dat == "TCGA":
		cfg['FILE_ONT_LVL1'] = "full_ont_lvl1_reactome.txt"
		cfg['CLINIC_PARAM'] = [
						"CANCER_TYPE_ACRONYM", 
						"TMB_NONSYNONYMOUS",
						"AGE",
						"SEX",
						"AJCC_PATHOLOGIC_TUMOR_STAGE_SHORT",
						"OS_STATUS",
						"OS_MONTHS",
						"DFS_STATUS",
						"PFS_STATUS",
						"MSI_SCORE_MANTIS",
						"ANEUPLOIDY_SCORE"
					]
	if dat == "SC":
		cfg['FILE_ONT_LVL1'] = "full_ont_lvl1_ensembl_reactome.txt"
		cfg['CLINIC_PARAM'] = [
						"author_cell_type",
						"age_group",
						"sex"
					]

	for t in train_type:
		cfg['TRAIN_TYPE'] = t
		if t == "tune":
			## Tuning
			cfg['LAYERS_LOWER_LIMIT'] = 2
			cfg['LAYERS_UPPER_LIMIT'] = 4

			cfg['LR_LOWER_LIMIT'] = 0.00001
			cfg['LR_UPPER_LIMIT'] = 0.01

			cfg['DROPOUT_LOWER_LIMIT'] = 0.0
			cfg['DROPOUT_UPPER_LIMIT'] = 0.5

			cfg['OPTUNA_TRIALS'] = 50

		for a in arch:
			cfg['MODEL_TYPE'] = a
			if a == "ontix":
				cfg['LAYERS_LOWER_LIMIT'] = 1 # At least one layer necessary to reach latent dim
				cfg['LAYERS_UPPER_LIMIT'] = 2 # At max 2 FC layer + 2 Sparse decoder layer = 4 layers
			for d in dim:
				cfg['LATENT_DIM_FIXED'] = d
				if d == 29:
					cfg['NON_ONT_LAYER'] = 0 # if dim < 29 -> 1
					if t == "tune" and a =="ontix":
						cfg['LAYERS_LOWER_LIMIT'] = 0
						cfg['LAYERS_UPPER_LIMIT'] = 0 ## Fix Layers to maintain explainability
				else:
					cfg['NON_ONT_LAYER'] = 1

				for r in range(1,len(dm)+1):
					all = combinations(dm,r=r)
					for dm_comb in all:
						cfg['DATA_TYPE'] = dict()
						cfg['DATA_TYPE']['ANNO'] = dict()
						cfg['DATA_TYPE']['ANNO']['TYPE'] = "ANNOTATION"

						if len(dm_comb) == 3:	# Adjust K_FILTER to have uniform total number of feature
							cfg['K_FILTER'] = 2000
						if len(dm_comb) == 2:
							if dat == "TCGA":
								cfg['K_FILTER'] = 3000
							if dat == "SC":
								cfg['K_FILTER'] = 1500
						if len(dm_comb) == 1:
							if dat == "TCGA":
								cfg['K_FILTER'] = 6000
							if dat == "SC":
								cfg['K_FILTER'] = 3000

						for m in dm_comb:								
							cfg['DATA_TYPE'][m] = dict()
							cfg['DATA_TYPE'][m]['SCALING'] = "Standard"
							cfg['DATA_TYPE'][m]['TYPE'] = "NUMERIC"
							cfg['DATA_TYPE'][m]['FILTERING'] = "Var"

							if dat == "TCGA":
								cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "data_clinical_formatted.parquet"
								if m == "RNA":
									cfg['DATA_TYPE'][m]['FILE_RAW'] = "data_mrna_seq_v2_rsem_formatted.parquet"
								if m == "METH":
									cfg['DATA_TYPE'][m]['FILE_RAW'] = "data_methylation_per_gene_formatted.parquet"
								if m == "MUT":
									cfg['DATA_TYPE'][m]['FILE_RAW'] = "data_combi_MUT_CNA_formatted.parquet"
							if dat == "SC":
								cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "scATAC_human_cortex_clinical_formatted.parquet"
								if m == "RNA":
									cfg['DATA_TYPE'][m]['FILE_RAW'] = "scRNA_human_cortex_formatted.parquet"
								if m == "METH":
									cfg['DATA_TYPE'][m]['FILE_RAW'] = "scATAC_human_cortex_formatted.parquet"
								if m == "MUT": ## should be unused
									cfg['DATA_TYPE'][m]['FILE_RAW'] = ""



						if not a == "vanillix": # Vanillix needs not multiple beta runs
							for b in beta:
								# if (b == 1) and (not a == "varix"):
								# 	continue ## Skip beta = 1 for other types
								# if a == "stackix":
								# 	b = round(b*0.1, abs(int(np.log10(b*0.1)))) ## Stackix requires lower beta
								cfg['BETA'] = b
								if not (a == "stackix" and len(dm_comb) <2):
									if not (dat == "SC" and "MUT" in "-".join(dm_comb)):
										run_id = cfg_prefix + "_" + \
											dat + "_" + 	\
											t + "_" + 		\
											a + "_" +		\
											"beta"+str(b) + "_" +	\
											"dim"+str(d) + "_" +	\
											"-".join(dm_comb) 
										
										with open(cfg_folder+'/'+run_id +"_config.yaml", 'w') as file:
											yaml.dump(cfg, file)
										
										count +=1
						else:
							b = "NA"
							cfg['BETA'] = 1
							if not (a == "stackix" and len(dm_comb) <2):
								if not (dat == "SC" and "MUT" in "-".join(dm_comb)):
									run_id = cfg_prefix + "_" + \
										dat + "_" + 	\
										t + "_" + 		\
										a + "_" +		\
										"beta"+str(b) + "_" +	\
										"dim"+str(d) + "_" +	\
										"-".join(dm_comb) 
									
									with open(cfg_folder+'/'+run_id +"_config.yaml", 'w') as file:
										yaml.dump(cfg, file)
									count +=1

print("Number of cfg created for Experiment 2: " + str(count))

###############################################
### cfg for ontix robustness "Experiment 3" ###

cfg_prefix = "Exp3"

cfg = dict()

cfg_folder = config_save_root + cfg_prefix
Path(cfg_folder).mkdir(parents=False, exist_ok=True)

## Default param runs for latent dist subpanel
cfg['FIX_RANDOMNESS'] = 'random'
cfg["LATENT_DIM_FIXED"] = 29
cfg["BATCH_SIZE"] = 256
cfg['LR_FIXED'] = 0.0001
cfg['EPOCHS'] = 1000
cfg['RECONSTR_LOSS'] = "MSE"
cfg['VAE_LOSS'] = "KL"
cfg['PREDICT_SPLIT'] = "all"
cfg['TRAIN_TYPE'] = "train"
cfg['MODEL_TYPE'] = "ontix"
cfg['BETA'] = 0.1
cfg['DROP_P'] = 0.5
cfg['NON_ONT_LAYER'] = 0
cfg['SPLIT'] = [0.6, 0.2, 0.2]

cfg['K_FILTER'] = 2000

cfg['DATA_TYPE'] = dict()
cfg['DATA_TYPE']['ANNO'] = dict()
cfg['DATA_TYPE']['ANNO']['TYPE'] = "ANNOTATION"

## Viz and ML 
cfg['DIM_RED_METH'] = "UMAP"
cfg['ML_TYPE'] = "Auto-detect"
cfg['ML_ALG'] = [
	'Linear',
	'RF',
]
cfg['ML_SPLIT'] = "use-split"
cfg['ML_TASKS'] = [
	'Latent'
	]

## variable params
# Ontology Rea + Chr
ontologies = ['Rea', 'Chr']
# Dataset TCGA + SC
dataset = ['TCGA','SC']

param_to_test = dict()

param_to_test = {
	# five beta level
	'B1': 0.001,	
	'B2': 0.01,
	'B3': 0.1,
	'B4': 0.5,
	'B5': 1,
	# five drop out level
	'D1': 0.0,
	'D2': 0.1,
	'D3': 0.25,
	'D4': 0.5,
	'D5': 0.9,
	# five learn rate level
	'L1': 1e-6,
	'L2': 1e-5,
	'L3': 1e-4,
	'L4': 1e-3,
	'L5': 1e-2
	}

# random repetitions
n_rep = 5
count = 0
for param in param_to_test:
	# Revert to default param before setting
	cfg['LR_FIXED'] = 0.0001
	cfg['BETA'] = 0.1
	cfg['DROP_P'] = 0.5

	if param[0] == 'B':
		cfg['BETA'] = param_to_test[param]
	if param[0] == 'D':
		cfg['DROP_P'] = param_to_test[param]
	if param[0] == 'L':
		cfg['LR_FIXED'] = param_to_test[param]		
	for dat in dataset:
		if dat == "TCGA":
			
			cfg['CLINIC_PARAM'] = [
							"CANCER_TYPE_ACRONYM", 
							"TMB_NONSYNONYMOUS",
							"AGE",
							"SEX",
							"AJCC_PATHOLOGIC_TUMOR_STAGE_SHORT"
						]
			
			cfg['DATA_TYPE'] = dict() ## Reset 
			cfg['DATA_TYPE']['ANNO'] = dict()
			cfg['DATA_TYPE']['ANNO']['TYPE'] = "ANNOTATION"
			cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "data_clinical_formatted.parquet"
			dm = ["RNA", "METH", "MUT"]
			for m in dm:
				cfg['DATA_TYPE'][m] = dict()
				cfg['DATA_TYPE'][m]['SCALING'] = "Standard"
				cfg['DATA_TYPE'][m]['TYPE'] = "NUMERIC"
				cfg['DATA_TYPE'][m]['FILTERING'] = "Var"	
				if m == "RNA":
					cfg['DATA_TYPE'][m]['FILE_RAW'] = "data_mrna_seq_v2_rsem_formatted.parquet"
				if m == "METH":
					cfg['DATA_TYPE'][m]['FILE_RAW'] = "data_methylation_per_gene_formatted.parquet"
				if m == "MUT":
					cfg['DATA_TYPE'][m]['FILE_RAW'] = "data_combi_MUT_CNA_formatted.parquet"
		if dat == "SC":
			
			cfg['CLINIC_PARAM'] = [
							"author_cell_type",
							"age_group",
							"sex"
						]
			
			dm = ["RNA", "METH"]
			cfg['DATA_TYPE'] = dict() ## Reset 
			cfg['DATA_TYPE']['ANNO'] = dict()
			cfg['DATA_TYPE']['ANNO']['TYPE'] = "ANNOTATION"
			cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "scATAC_human_cortex_clinical_formatted.parquet"
			for m in dm:
				cfg['DATA_TYPE'][m] = dict()
				cfg['DATA_TYPE'][m]['SCALING'] = "Standard"
				cfg['DATA_TYPE'][m]['TYPE'] = "NUMERIC"
				cfg['DATA_TYPE'][m]['FILTERING'] = "Var"
				if m == "RNA":
					cfg['DATA_TYPE'][m]['FILE_RAW'] = "scRNA_human_cortex_formatted.parquet"
				if m == "METH":
					cfg['DATA_TYPE'][m]['FILE_RAW'] = "scATAC_human_cortex_formatted.parquet"
		
		for ont in ontologies:
			if ont == "Rea":
				if dat == "TCGA":
					cfg['FILE_ONT_LVL1'] = "full_ont_lvl1_reactome.txt" # Entrez ID
				if dat == "SC":
					cfg['FILE_ONT_LVL1'] = "full_ont_lvl1_ensembl_reactome.txt" # Ensembl ID
				cfg['FILE_ONT_LVL2'] = "full_ont_lvl2_reactome.txt" # both
			if ont == "Chr":
				if dat == "TCGA":
					cfg['FILE_ONT_LVL1'] = "chromosome_ont_lvl1_ncbi.txt" # Entrez ID
				if dat == "SC":
					cfg['FILE_ONT_LVL1'] = "chromosome_ont_lvl1_ensembl.txt" # Ensembl ID
				cfg['FILE_ONT_LVL2'] = "chromosome_ont_lvl2.txt" # both

			for rep in range(1,n_rep+1):
				run_id = 	cfg_prefix + "_" + \
							dat + "_" + 	\
							ont + "_" + \
							"rand"+str(rep) + "_" + \
							param 
				with open(cfg_folder+'/'+run_id +"_config.yaml", 'w') as file:
					yaml.dump(cfg, file)
					count += 1

print("Number of cfg created for Experiment 3: " + str(count))
			

#################################################
### cfg for x-modalix examples "Experiment 4" ###

cfg_prefix = "Exp4"
cfg = dict()

cfg_folder = config_save_root + cfg_prefix
Path(cfg_folder).mkdir(parents=False, exist_ok=True)

cfg['FIX_RANDOMNESS'] = 'all'
cfg['GLOBAL_SEED'] = 42
cfg["CHECKPT_PLOT"] = True
cfg["LATENT_DIM_FIXED"] = 32
cfg["BATCH_SIZE"] = 32
cfg['LR_FIXED'] = 0.0005
cfg['EPOCHS'] = 1000
cfg['RECONSTR_LOSS'] = "MSE"
cfg['VAE_LOSS'] = "KL"
cfg['PREDICT_SPLIT'] = "all"
cfg['TRAIN_TYPE'] = "train"
cfg['MODEL_TYPE'] = "x-modalix"
cfg['BETA'] = 0.01
cfg["GAMMA"] = 1.25
cfg["ROOT_IMAGE"] = "data/raw/images/ALY-2_SYS721/"
cfg['SPLIT'] = [0.8, 0.0, 0.2]
# X-modalix specific
cfg["DELTA_PAIR"] = 0.7
cfg["DELTA_CLASS"] = 0.0
cfg["PRETRAIN_TARGET_MODALITY"] = "pretrain_image"
cfg["ANNEAL_PRETRAINING"] = True
cfg["PRETRAIN_EPOCHS"] = 100
cfg["TRANSLATE"] = "RNA_to_IMG"


cfg['K_FILTER'] = 1000
cfg['DATA_TYPE'] = dict()
cfg['DATA_TYPE']['ANNO'] = dict()
cfg['DATA_TYPE']['ANNO']['TYPE'] = "ANNOTATION"
cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "ALY-2_SYS721_mappings.txt"
dm = ["RNA", "IMG"]
for m in dm:								
	cfg['DATA_TYPE'][m] = dict()
	cfg['DATA_TYPE'][m]['SCALING'] = "NoScaler"
	cfg['DATA_TYPE'][m]['FILTERING'] = "NoFilt"
	if m == "RNA":
		cfg['DATA_TYPE'][m]['TYPE'] = "NUMERIC"
		cfg['DATA_TYPE'][m]['SCALING'] = "MinMax"
		cfg['DATA_TYPE'][m]['FILE_RAW'] = "AM3_NO2_raw_cell.tsv"
	if m == "IMG":
		cfg['DATA_TYPE'][m]['FILE_RAW'] = "ALY-2_SYS721_mappings.txt"
		cfg['DATA_TYPE'][m]['TYPE'] = "IMG"
		cfg['DATA_TYPE'][m]['WIDTH'] = 128
		cfg['DATA_TYPE'][m]['HEIGHT'] = 128
cfg['CLINIC_PARAM'] = [
				"extra_class_labels" 
			]
cfg["CLASS_PARAM"] = None
cfg['DIM_RED_METH'] = "UMAP"
cfg["PLOT_NUMERIC"] = True


run_id = cfg_prefix + "_Celegans_TF"
with open(cfg_folder+'/'+run_id +"_config.yaml", 'w') as file:
	yaml.dump(cfg, file)
print("Config created for Experiment 4")

### Exp4 part two ImgImg compare
cfg_prefix = "Exp4"
cfg = dict()

cfg_folder = config_save_root + cfg_prefix
Path(cfg_folder).mkdir(parents=False, exist_ok=True)

cfg['FIX_RANDOMNESS'] = 'all'
cfg['GLOBAL_SEED'] = 42
cfg["CHECKPT_PLOT"] = True
cfg["LATENT_DIM_FIXED"] = 32
cfg["BATCH_SIZE"] = 32
cfg['LR_FIXED'] = 0.0005
cfg['EPOCHS'] = 1000
cfg['RECONSTR_LOSS'] = "MSE"
cfg['VAE_LOSS'] = "KL"
cfg['PREDICT_SPLIT'] = "all"
cfg['TRAIN_TYPE'] = "train"
cfg['MODEL_TYPE'] = "x-modalix"
cfg['BETA'] = 0.01
cfg["GAMMA"] = 1.25
cfg["ROOT_IMAGE"] = "data/raw/images/ALY-2_SYS721/"

# X-modalix specific
cfg["DELTA_PAIR"] = 0.7
cfg["DELTA_CLASS"] = 0.0
cfg["PRETRAIN_TARGET_MODALITY"] = "pretrain_image"
cfg["ANNEAL_PRETRAINING"] = True
cfg["PRETRAIN_EPOCHS"] = 0
cfg["TRANSLATE"] = "IMG_to_IMG"


cfg['K_FILTER'] = 1000
cfg['DATA_TYPE'] = dict()
cfg['DATA_TYPE']['ANNO'] = dict()
cfg['DATA_TYPE']['ANNO']['TYPE'] = "ANNOTATION"
cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "ALY-2_SYS721_mappings.txt"
dm = ["IMG"]
for m in dm:								
	cfg['DATA_TYPE'][m] = dict()
	cfg['DATA_TYPE'][m]['SCALING'] = "NoScaler"
	cfg['DATA_TYPE'][m]['FILTERING'] = "NoFilt"
	if m == "IMG":
		cfg['DATA_TYPE'][m]['FILE_RAW'] = "ALY-2_SYS721_mappings.txt"
		cfg['DATA_TYPE'][m]['TYPE'] = "IMG"
		cfg['DATA_TYPE'][m]['WIDTH'] = 128
		cfg['DATA_TYPE'][m]['HEIGHT'] = 128
cfg['CLINIC_PARAM'] = [
				"extra_class_labels" 
			]
cfg["CLASS_PARAM"] = None
cfg['DIM_RED_METH'] = "UMAP"
cfg["PLOT_NUMERIC"] = True

run_id = cfg_prefix + "_Celegans_TFImgImg"
with open(cfg_folder+'/'+run_id +"_config.yaml", 'w') as file:
	yaml.dump(cfg, file)
print("Config created for Experiment 4 ImgImg")




#################################################
### cfg for x-modalix examples "Experiment 5" ###

cfg_prefix = "Exp5"
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
cfg["ROOT_IMAGE"] = "data/raw/images/tcga_fake/"
cfg['DROP_P'] = 0.5

cfg['SPLIT'] = [0.8, 0.0, 0.2]
# X-modalix specific
cfg["DELTA_PAIR"] = 0
cfg["DELTA_CLASS"] = 10
cfg["PRETRAIN_TARGET_MODALITY"] = "pretrain_image"
cfg["ANNEAL_PRETRAINING"] = True
cfg["PRETRAIN_EPOCHS"] = 100
cfg["TRANSLATE"] = "RNA_to_IMG"


cfg['K_FILTER'] = 4000
cfg['DATA_TYPE'] = dict()
cfg['DATA_TYPE']['ANNO'] = dict()
cfg['DATA_TYPE']['ANNO']['TYPE'] = "ANNOTATION"
cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "tcga_mappings.txt"
dm = ["RNA", "IMG"]
for m in dm:								
	cfg['DATA_TYPE'][m] = dict()
	if m == "RNA":
		cfg['DATA_TYPE'][m]['SCALING'] = "MinMax"
		cfg['DATA_TYPE'][m]['FILTERING'] = "Var"
		cfg['DATA_TYPE'][m]['TYPE'] = "NUMERIC"
		cfg['DATA_TYPE'][m]['FILE_RAW'] = "data_mrna_seq_v2_rsem_formatted.parquet" 
	elif m == "IMG":
		cfg['DATA_TYPE'][m]['SCALING'] = "NoScaler"
		cfg['DATA_TYPE'][m]['FILTERING'] = "NoFilt"
		cfg['DATA_TYPE'][m]['FILE_RAW'] = "tcga_mappings.txt"
		cfg['DATA_TYPE'][m]['TYPE'] = "IMG"
		cfg['DATA_TYPE'][m]['WIDTH'] = 64
		cfg['DATA_TYPE'][m]['HEIGHT'] = 64
cfg['CLINIC_PARAM'] = ["extra_class_labels"]
cfg["CLASS_PARAM"] = "extra_class_labels"
cfg['DIM_RED_METH'] = "UMAP"

run_id = cfg_prefix + "_TCGA_MNIST"
with open(cfg_folder+'/'+run_id +"_config.yaml", 'w') as file:
	yaml.dump(cfg, file)
print("Config created for Experiment 5")

### cfg for x-modalix examples "Experiment 5" extra ImgImg ###

cfg_prefix = "Exp5"
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
cfg["ROOT_IMAGE"] = "data/raw/images/tcga_fake/"

# X-modalix specific
cfg["DELTA_PAIR"] = 0
cfg["DELTA_CLASS"] = 10
cfg["PRETRAIN_TARGET_MODALITY"] = "pretrain_image"
cfg["ANNEAL_PRETRAINING"] = True
cfg["PRETRAIN_EPOCHS"] = 0
cfg["TRANSLATE"] = "IMG_to_IMG"


cfg['K_FILTER'] = 4000
cfg['DATA_TYPE'] = dict()
cfg['DATA_TYPE']['ANNO'] = dict()
cfg['DATA_TYPE']['ANNO']['TYPE'] = "ANNOTATION"
cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "tcga_mappings.txt"
dm = ["IMG"]
for m in dm:								
	cfg['DATA_TYPE'][m] = dict()
	if m == "IMG":
		cfg['DATA_TYPE'][m]['SCALING'] = "NoScaler"
		cfg['DATA_TYPE'][m]['FILTERING'] = "NoFilt"
		cfg['DATA_TYPE'][m]['FILE_RAW'] = "tcga_mappings.txt"
		cfg['DATA_TYPE'][m]['TYPE'] = "IMG"
		cfg['DATA_TYPE'][m]['WIDTH'] = 64
		cfg['DATA_TYPE'][m]['HEIGHT'] = 64
cfg['CLINIC_PARAM'] = [ "extra_class_labels"]
cfg["CLASS_PARAM"] = "extra_class_labels"
cfg['DIM_RED_METH'] = "UMAP"

run_id = cfg_prefix + "_TCGA_MNISTImgImg"
with open(cfg_folder+'/'+run_id +"_config.yaml", 'w') as file:
	yaml.dump(cfg, file)
print("Config created for Experiment 5 ImgImg")