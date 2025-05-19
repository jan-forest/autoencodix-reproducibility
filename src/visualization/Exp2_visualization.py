import glob
import yaml
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib.ticker import LinearLocator, LogLocator

from seaborn import axes_style
import matplotlib.pyplot as plt

import seaborn.objects as so

so.Plot.config.theme.update(axes_style("whitegrid"))

sns.set_context("notebook")
sns.set_style("whitegrid")

## Pre-settings
config_prefix_list = ["Exp2_TCGA_tune", "Exp2_SC_tune"]

rootdir = "./"
# rootdir = "/mnt/c/Users/ewald/Nextcloud/eigene_shares/AutoEncoderOmics/SaveResults/250507_first_revision_results/"

rootsave = "./reports/paper-visualizations/Exp2/"
output_type = ".png"

plot_rec_type = "R2_valid"
plot_rec_type_tune = "R2_valid_diff"

## Read-in

print("Read-in configs and losses of tuned runs")
df_results = pd.DataFrame(
    columns=[
        "Config_ID",
        "Architecture",
        "Data_Modalities",
        "Latent_Dim",
        "Latent_Coverage",
        "R2_valid",
        "R2_train",
        "Rec. loss",
        "Total loss",
        "Data_set",
    ]
)

## Browse configs in reports and get infos
for config_prefix in config_prefix_list:
    file_regex = "reports/" + "*/" + config_prefix + "*_config.yaml"
    file_list = glob.glob(rootdir + file_regex)

    for config_path in file_list:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        dm = list(config["DATA_TYPE"].keys())
        dm.remove("ANNO")

        results_row = {
            "Config_ID": config["RUN_ID"],
            "Architecture": config["MODEL_TYPE"] + "_B" + str(config["BETA"]),
            "Data_Modalities": "+".join(dm),
            "Latent_Dim": config["LATENT_DIM_FIXED"],
            "Latent_Coverage": 0.0,
            "R2_valid": 0.0,
            "R2_train": 0.0,
            "Rec. loss": 0.0,
            "Total loss": 0.0,
            "Data_set": config_prefix.split("_")[1],
            "weight_decay": 0.0,
            "dropout_all": 0.0,
            "encoding_factor": 0,
            "lr": 0.0,
            "fc_n_layers": 0,
        }
        df_results.loc[config["RUN_ID"], :] = results_row

        param_path_regex = (
            rootdir
            + "reports/"
            + config["RUN_ID"]
            + "/CO*"  # match CONCAT or COMBINED
            + "best_model_params.txt"
        )
        if len(glob.glob(param_path_regex)) > 0:
            param_path = glob.glob(param_path_regex)[
                0
            ]  # Should only be one matching file
            best_params_series = pd.read_csv(
                param_path, header=0, names=["parameter", "values"], index_col=0
            )["values"]
            run_id = config["RUN_ID"]
            df_results.loc[run_id, "weight_decay"] = best_params_series["weight_decay"]
            df_results.loc[run_id, "dropout_all"] = best_params_series["dropout_all"]
            df_results.loc[run_id, "encoding_factor"] = best_params_series[
                "encoding_factor"
            ]
            df_results.loc[run_id, "lr"] = best_params_series["lr"]
            df_results.loc[run_id, "fc_n_layers"] = best_params_series["fc_n_layers"]


for run_id in df_results.index:
    dm = df_results.loc[run_id, "Data_Modalities"].split("+")

    if "stackix" in run_id:
        df_results.loc[run_id, "Rec. loss"] = 0
        df_results.loc[run_id, "Total loss"] = 0
        ## single dm
        for d in dm:
            loss_file = (
                rootdir
                + "reports/"
                + run_id
                + "/losses_tuned_"
                + df_results.loc[run_id, "Architecture"].split("_")[0]
                + "_base_"
                + d
                + ".parquet"
            )
            loss_df = pd.read_parquet(loss_file)
            if "train_r2" in loss_df.columns:
                df_results.loc[run_id, "Rec. loss"] += loss_df["valid_recon_loss"].iloc[
                    -1
                ]
                df_results.loc[run_id, "Total loss"] += loss_df[
                    "valid_total_loss"
                ].iloc[-1]

        ## combined
        loss_file = (
            rootdir
            + "reports/"
            + run_id
            + "/losses_tuned_"
            + df_results.loc[run_id, "Architecture"].split("_")[0]
            + "_concat_"
            + "_".join(dm)
            + ".parquet"
        )
        loss_df = pd.read_parquet(loss_file)
        if "train_r2" in loss_df.columns:
            df_results.loc[run_id, "R2_train"] = loss_df["train_r2"].iloc[-1]
            df_results.loc[run_id, "R2_valid"] = loss_df["valid_r2"].iloc[-1]
            df_results.loc[run_id, "Rec. loss"] += loss_df["valid_recon_loss"].iloc[-1]
            df_results.loc[run_id, "Total loss"] += loss_df["valid_total_loss"].iloc[-1]

    else:
        loss_file = (
            rootdir
            + "reports/"
            + run_id
            + "/losses_tuned_"
            + "_".join(dm)
            + "_"
            + df_results.loc[run_id, "Architecture"].split("_")[0]
            + ".parquet"
        )

        loss_df = pd.read_parquet(loss_file)
        if "train_r2" in loss_df.columns:
            df_results.loc[run_id, "R2_train"] = loss_df["train_r2"].iloc[-1]
            df_results.loc[run_id, "R2_valid"] = loss_df["valid_r2"].iloc[-1]
            df_results.loc[run_id, "Rec. loss"] = loss_df["valid_recon_loss"].iloc[-1]
            df_results.loc[run_id, "Total loss"] = loss_df["valid_total_loss"].iloc[-1]

## Plot Recon Loss summaries
print("Plot Recon Loss plots for tuned runs")

arch_order = [
    "vanillix_B1",
    "varix_B1",
    "varix_B0.1",
    "varix_B0.01",
    "ontix_B1",
    "ontix_B0.1",
    "ontix_B0.01",
    "stackix_B1",
    "stackix_B0.1",
    "stackix_B0.01",
]

box_rec_dense = sns.catplot(
    data=df_results,
    x="Architecture",
    # y="Rec. loss",
    y=plot_rec_type,
    hue="Latent_Dim",
    col="Latent_Dim",
    palette="pastel",
    order=arch_order,
    kind="box",
    sharey=True,
    aspect=0.8,
)
box_rec_dense.tick_params(axis="x", rotation=90)
box_rec_dense.tight_layout()
box_rec_dense.savefig(rootsave + "Exp2_Fig2C_recon_dense" + output_type)


row_order = ["RNA", "METH", "MUT", "METH+RNA", "MUT+RNA", "METH+MUT", "METH+MUT+RNA"]

box_rec_detail = sns.catplot(
    data=df_results,
    x="Architecture",
    # y="Rec. loss",
    y=plot_rec_type,
    hue="Latent_Dim",
    col="Data_set",
    row="Data_Modalities",
    palette="pastel",
    order=arch_order,
    row_order=row_order,
    kind="bar",
    sharey=True,
    aspect=2,
)
box_rec_detail.tick_params(axis="x", rotation=90)
box_rec_detail.tight_layout()
box_rec_detail.savefig(rootsave + "Exp2_SuppFig_recon_detail" + output_type)

### Get ML Results of tuned
print("Get ML results of tuned runs")
ml_results = pd.DataFrame(
    columns=[
        "Config_ID",
        "Architecture",
        "Data_Modalities",
        "Latent_Dim",
        "ML_Algorithm",
        "Parameter",
        "Metric",
        "Performance",
        "Perf_std",
        "Split",
        "Data_set",
    ]
)

for config_prefix in config_prefix_list:
    file_regex = "reports/" + config_prefix + "*/" + "*ml_task_performance.txt"
    file_list = glob.glob(rootdir + file_regex)

    sep = "\t"

    for ml_perf_file in file_list:
        ml_df = pd.read_csv(ml_perf_file, sep=sep)
        if not (
            "RandomFeature" in ml_df["ML_TASK"].unique()
        ):  ### Only for fast testing ###
            fake_random = ml_df[ml_df["ML_TASK"] == "Latent"].replace(
                "Latent", "RandomFeature"
            )
            fake_random2 = fake_random.copy()
            fake_random["value"] = fake_random["value"] * np.random.uniform(0, 1, 1)[0]
            fake_random2["value"] = (
                fake_random2["value"] * np.random.uniform(0, 1, 1)[0]
            )
            fake_random["ML_SUBTASK"] = fake_random["ML_SUBTASK"].replace(
                "RandonFeature", "RandonFeature1"
            )
            fake_random2["ML_SUBTASK"] = fake_random2["ML_SUBTASK"].replace(
                "RandonFeature", "RandonFeature2"
            )
            ml_df = pd.concat([ml_df, fake_random, fake_random2]).reset_index(drop=True)

        # run_id = ml_perf_file.split("/")[2]
        # print(ml_perf_file)
        run_id = ml_perf_file.removeprefix(rootdir).split("/")[1]
        arch = df_results.loc[run_id, "Architecture"]
        dm = df_results.loc[run_id, "Data_Modalities"]
        latent_dim = df_results.loc[run_id, "Latent_Dim"]

        ml_std = (
            ml_df.groupby(
                ["metric", "ML_TASK", "ML_ALG", "score_split", "CLINIC_PARAM"],
                as_index=False,
            )
            .std(numeric_only=True)
            .loc[:, "value"]
        )
        ml_df = ml_df.groupby(
            ["metric", "ML_TASK", "ML_ALG", "score_split", "CLINIC_PARAM"],
            as_index=False,
        ).mean(numeric_only=True)

        ml_df = ml_df.rename(
            columns={
                "CLINIC_PARAM": "Parameter",
                "metric": "Metric",
                "value": "Performance",
                "ML_ALG": "ML_Algorithm",
                "ML_TASK": "Architecture",
                "score_split": "Split",
            }
        )

        ml_df.loc[:, "Perf_std"] = ml_std
        ml_df.loc[ml_df.loc[:, "Architecture"] == "Latent", "Architecture"] = arch
        ml_df.loc[:, "Config_ID"] = run_id
        ml_df.loc[:, "Data_Modalities"] = dm
        ml_df.loc[:, "Latent_Dim"] = float(
            latent_dim
        )  ## Avoiding error in seaborn PlotSpecError
        ml_df.loc[:, "Data_set"] = config_prefix.split("_")[1]

        if len(ml_results.index) > 0:
            ml_results = pd.concat([ml_results, ml_df])
        else:
            ml_results = ml_df

ml_results.index = ml_results["Config_ID"]
ml_results.index.name = "Index"

ml_results.reset_index(inplace=True)
# print(ml_results.shape)

ml_results_normed = pd.DataFrame(
    columns=[
        "Config_ID",
        "Architecture",
        "Data_Modalities",
        "Latent_Dim",
        "ML_Algorithm",
        "Parameter",
        "Metric",
        "Performance",
        "Perf_std",
        "Split",
        "Data_set",
    ]
)

for config_id in ml_results["Config_ID"].unique():
    # print(config_id)
    for arch in ml_results.loc[
        ml_results["Config_ID"] == config_id, "Architecture"
    ].unique():
        # print(arch)
        if not arch == "RandomFeature":
            ml_results_normed_new = ml_results.loc[
                (ml_results["Architecture"] == arch)
                & (ml_results["Config_ID"] == config_id),
                :,
            ].reset_index(drop=True)
            ml_random_perf = ml_results.loc[
                (ml_results["Architecture"] == "RandomFeature")
                & (ml_results["Config_ID"] == config_id),
                "Performance",
            ].reset_index(drop=True)
            ml_random_perf_std = ml_results.loc[
                (ml_results["Architecture"] == "RandomFeature")
                & (ml_results["Config_ID"] == config_id),
                "Perf_std",
            ].reset_index(drop=True)
            ml_results_normed_new.loc[:, "Performance"] = (
                ml_results_normed_new.loc[:, "Performance"] - ml_random_perf
            ) / ml_random_perf_std
            if len(ml_results_normed.index) > 0:
                ml_results_normed = pd.concat(
                    [ml_results_normed, ml_results_normed_new]
                )
            else:
                ml_results_normed = ml_results_normed_new

ml_results_normed.rename(columns={"Performance": "Perf. over random"}, inplace=True)
ml_results_normed.reset_index(inplace=True)

print("Plot ML results of tuned runs")
sel_ml_alg = "Linear"

arch_plus_order = [
    "vanillix_B1",
    "varix_B1",
    "varix_B0.1",
    "varix_B0.01",
    "ontix_B1",
    "ontix_B0.1",
    "ontix_B0.01",
    "stackix_B1",
    "stackix_B0.1",
    "stackix_B0.01",
    "PCA",
    "UMAP",
]

box_ml_linear = sns.catplot(
    data=ml_results_normed.loc[
        (ml_results_normed["ML_Algorithm"] == sel_ml_alg)
        & (ml_results_normed["Split"] == "test"),
        :,
    ],
    x="Architecture",
    y="Perf. over random",
    col="Latent_Dim",
    hue="Latent_Dim",
    # row="Data_set",
    palette="pastel",
    order=arch_plus_order,
    kind="box",
    aspect=0.8,
)
# ## Get mean Perf. over random for architecture PCA for each latent dimension
# latent_dims = [2, 8, 29]
# pca_perf = {
#     latent_dim: ml_results_normed.loc[
#         (ml_results_normed["ML_Algorithm"] == sel_ml_alg)
#         & (ml_results_normed["Split"] == "test")
#         & (ml_results_normed["Architecture"] == "PCA")
#         & (ml_results_normed["Latent_Dim"] == latent_dim),
#         "Perf. over random",
#     ].median()
#     for latent_dim in latent_dims
# }


box_ml_linear.tick_params(axis="x", rotation=90)
# box_ml_linear.set(yscale="symlog")
# Set y-axis limit
box_ml_linear.set(ylim=(-10, 12))
# box_ml_linear.refline(y=pca_perf[2], color="blue")
# box_ml_linear.refline(y=pca_perf[8], color="orange")
# box_ml_linear.refline(y=pca_perf[29], color="green")
box_ml_linear.tight_layout()
box_ml_linear.savefig(rootsave + "Exp2_Fig2D_ml-linear_dense" + output_type)

box_ml_detail = sns.catplot(
    data=ml_results_normed.loc[ml_results_normed["Split"] == "test", :],
    x="Architecture",
    y="Perf. over random",
    hue="Latent_Dim",
    row="ML_Algorithm",
    col="Data_set",
    palette="pastel",
    order=arch_plus_order,
    kind="box",
    aspect=2,
)
# box_ml_detail.set(yscale="symlog")
box_ml_detail.set(ylim=(-15, 30))
box_ml_detail.tick_params(axis="x", rotation=90)
box_ml_detail.savefig(rootsave + "Exp2_SuppFig_ml-all_detail" + output_type)

#####################
### Tuning impact ###
#####################
print("Make comparison to untuned runs")
print("Read in data")


not_tuned_list = ["Exp2_TCGA_train", "Exp2_SC_train"]

## read in reconstruction performance

df_results_untuned = pd.DataFrame(
    columns=[
        "Config_ID",
        "Architecture",
        "Data_Modalities",
        "Latent_Dim",
        "Latent_Coverage",
        "Rec. loss",
        "R2_valid",
        "R2_train",
        "Total loss",
        "Data_set",
    ]
)

## Browse configs in reports and get infos
for config_prefix in not_tuned_list:
    file_regex = "reports/" + "*/" + config_prefix + "*_config.yaml"
    file_list = glob.glob(rootdir + file_regex)

    for config_path in file_list:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        dm = list(config["DATA_TYPE"].keys())
        dm.remove("ANNO")

        results_row = {
            "Config_ID": config["RUN_ID"],
            "Architecture": config["MODEL_TYPE"] + "_B" + str(config["BETA"]),
            "Data_Modalities": "+".join(dm),
            "Latent_Dim": config["LATENT_DIM_FIXED"],
            "Latent_Coverage": 0.0,
            "R2_valid": 0.0,
            "R2_train": 0.0,
            "Rec. loss": 0.0,
            "Total loss": 0.0,
            "Data_set": config_prefix.split("_")[1],
            "weight_decay": 0.0,
            "dropout_all": 0.0,
            "encoding_factor": 0,
            "lr": 0.0,
        }
        df_results_untuned.loc[config["RUN_ID"], :] = results_row

for run_id in df_results_untuned.index:

    dm = df_results_untuned.loc[run_id, "Data_Modalities"].split("+")

    # loss_file = max(file_list, key=len)  # combined loss always has the longest name
    if "stackix" in run_id:
        df_results_untuned.loc[run_id, "Rec. loss"] = 0
        df_results_untuned.loc[run_id, "Total loss"] = 0
        ## single dm
        for d in dm:
            loss_file = (
                rootdir
                + "reports/"
                + run_id
                + "/losses_"
                + df_results_untuned.loc[run_id, "Architecture"].split("_")[0]
                + "_base_"
                + d
                + ".parquet"
            )
            loss_df = pd.read_parquet(loss_file)
            if "train_r2" in loss_df.columns:
                df_results_untuned.loc[run_id, "Rec. loss"] += loss_df[
                    "valid_recon_loss"
                ].iloc[-1]
                df_results_untuned.loc[run_id, "Total loss"] += loss_df[
                    "valid_total_loss"
                ].iloc[-1]

        ## combined
        loss_file = (
            rootdir
            + "reports/"
            + run_id
            + "/losses_"
            + df_results_untuned.loc[run_id, "Architecture"].split("_")[0]
            + "_concat_"
            + "_".join(dm)
            + ".parquet"
        )
        loss_df = pd.read_parquet(loss_file)
        if "train_r2" in loss_df.columns:
            df_results_untuned.loc[run_id, "R2_train"] = loss_df["train_r2"].iloc[-1]
            df_results_untuned.loc[run_id, "R2_valid"] = loss_df["valid_r2"].iloc[-1]
            df_results_untuned.loc[run_id, "Rec. loss"] += loss_df[
                "valid_recon_loss"
            ].iloc[-1]
            df_results_untuned.loc[run_id, "Total loss"] += loss_df[
                "valid_total_loss"
            ].iloc[-1]

    else:
        loss_file = (
            rootdir
            + "reports/"
            + run_id
            + "/losses_"
            + "_".join(dm)
            + "_"
            + df_results_untuned.loc[run_id, "Architecture"].split("_")[0]
            + ".parquet"
        )

        loss_df = pd.read_parquet(loss_file)
        if "train_r2" in loss_df.columns:
            df_results_untuned.loc[run_id, "R2_train"] = loss_df["train_r2"].iloc[-1]
            df_results_untuned.loc[run_id, "R2_valid"] = loss_df["valid_r2"].iloc[-1]
            df_results_untuned.loc[run_id, "Rec. loss"] = loss_df[
                "valid_recon_loss"
            ].iloc[-1]
            df_results_untuned.loc[run_id, "Total loss"] = loss_df[
                "valid_total_loss"
            ].iloc[-1]

## read in ml task performance

ml_results_untuned = pd.DataFrame(
    columns=[
        "Config_ID",
        "Architecture",
        "Data_Modalities",
        "Latent_Dim",
        "ML_Algorithm",
        "Parameter",
        "Metric",
        "Performance",
        "Split",
        "Data_set",
    ]
)

for config_prefix in not_tuned_list:
    file_regex = "reports/" + config_prefix + "*/" + "*ml_task_performance.txt"
    file_list = glob.glob(rootdir + file_regex)

    sep = "\t"

    for ml_perf_file in file_list:
        ml_df = pd.read_csv(ml_perf_file, sep=sep)

        run_id = ml_perf_file.removeprefix(rootdir).split("/")[1]
        arch = df_results_untuned.loc[run_id, "Architecture"]
        dm = df_results_untuned.loc[run_id, "Data_Modalities"]
        latent_dim = df_results_untuned.loc[run_id, "Latent_Dim"]

        ml_df = ml_df.groupby(
            ["metric", "ML_TASK", "ML_ALG", "score_split", "CLINIC_PARAM"],
            as_index=False,
        ).mean(numeric_only=True)

        ml_df = ml_df.rename(
            columns={
                "CLINIC_PARAM": "Parameter",
                "metric": "Metric",
                "value": "Performance",
                "ML_ALG": "ML_Algorithm",
                "ML_TASK": "Architecture",
                "score_split": "Split",
            }
        )

        ml_df.loc[ml_df.loc[:, "Architecture"] == "Latent", "Architecture"] = arch
        ml_df.loc[:, "Config_ID"] = run_id
        ml_df.loc[:, "Data_Modalities"] = dm
        ml_df.loc[:, "Latent_Dim"] = latent_dim
        ml_df.loc[:, "Data_set"] = config_prefix.split("_")[1]

        if len(ml_results_untuned.index) > 0:
            ml_results_untuned = pd.concat([ml_results_untuned, ml_df])
        else:
            ml_results_untuned = ml_df


ml_results_untuned.index = ml_results_untuned["Config_ID"]
ml_results_untuned.index.name = "Index"

## calculate difference to tuned AE
tune_train_both = df_results_untuned.index.intersection(
    df_results["Config_ID"].str.replace("_tune_", "_train_")
)
r2_untuned = df_results_untuned.loc[tune_train_both, "R2_valid"]

r2_untuned.index = r2_untuned.index.str.replace("_train_", "_tune_")

df_results.loc[:, "R2_valid_untuned"] = r2_untuned
df_results.loc[:, "R2_valid_diff"] = (
    df_results.loc[:, "R2_valid"] - df_results.loc[:, "R2_valid_untuned"]
)
df_results["R2_valid_diff"] = df_results["R2_valid_diff"].astype(float)

#
recon_untuned = df_results_untuned.loc[tune_train_both, "Rec. loss"]

recon_untuned.index = recon_untuned.index.str.replace("_train_", "_tune_")

df_results.loc[:, "Rec. loss_untuned"] = recon_untuned
df_results.loc[:, "Rec. loss improvement"] = -1 * (
    df_results.loc[:, "Rec. loss"] - df_results.loc[:, "Rec. loss_untuned"]
)  # change sign direction
df_results["Rec. loss improvement"] = df_results["Rec. loss improvement"].astype(float)

#
total_untuned = df_results_untuned.loc[tune_train_both, "Total loss"]

total_untuned.index = total_untuned.index.str.replace("_train_", "_tune_")

df_results.loc[:, "Total loss_untuned"] = total_untuned
df_results.loc[:, "Total loss improvement"] = -1 * (
    df_results.loc[:, "Total loss"] - df_results.loc[:, "Total loss_untuned"]
)  # change sign direction
df_results["Total loss improvement"] = df_results["Total loss improvement"].astype(
    float
)

ml_results_untuned.rename(columns={"Performance": "Performance_untuned"}, inplace=True)
ml_results_untuned["Config_ID"] = ml_results_untuned["Config_ID"].str.replace(
    "_train_", "_tune_"
)

ml_results = ml_results.merge(
    ml_results_untuned,
    how="left",
    on=[
        "Config_ID",
        "Architecture",
        "Data_Modalities",
        "Latent_Dim",
        "ML_Algorithm",
        "Parameter",
        "Metric",
        "Split",
        "Data_set",
    ],
)

for config_id in ml_results["Config_ID"].unique():
    for arch in ml_results.loc[
        ml_results["Config_ID"] == config_id, "Architecture"
    ].unique():

        ml_random_perf_std = ml_results.loc[
            (ml_results["Architecture"] == "RandomFeature")
            & (ml_results["Config_ID"] == config_id),
            "Perf_std",
        ].reset_index(drop=True)

        ml_perf_tuned = ml_results.loc[
            (ml_results["Architecture"] == arch)
            & (ml_results["Config_ID"] == config_id),
            "Performance",
        ].reset_index(drop=True)

        ml_perf_untuned = ml_results.loc[
            (ml_results["Architecture"] == arch)
            & (ml_results["Config_ID"] == config_id),
            "Performance_untuned",
        ].reset_index(drop=True)

        ml_results.loc[
            (ml_results["Architecture"] == arch)
            & (ml_results["Config_ID"] == config_id),
            "Perf. improvement",
        ] = list((ml_perf_tuned - ml_perf_untuned) / ml_random_perf_std)

print("Make Plots for tuning impact")

box_tune_recon_dense = sns.catplot(
    data=df_results,
    x="Architecture",
    # y="Rec. loss improvement",
    y=plot_rec_type_tune,
    hue="Latent_Dim",
    col="Latent_Dim",
    palette="pastel",
    order=arch_order,
    kind="box",
    sharey=True,
    aspect=0.8,
)
box_tune_recon_dense.tick_params(axis="x", rotation=90)
box_tune_recon_dense.tight_layout()
box_tune_recon_dense.savefig(rootsave + "Exp2_Fig2E_tune-recon_dense" + output_type)

box_tune_recon_detail = sns.catplot(
    data=df_results,
    x="Architecture",
    # y="Rec. loss improvement",
    y=plot_rec_type_tune,
    hue="Latent_Dim",
    col="Data_set",
    row="Data_Modalities",
    palette="pastel",
    order=arch_order,
    row_order=row_order,
    kind="bar",
    sharey=False,
    aspect=2,
)
box_tune_recon_detail.tick_params(axis="x", rotation=90)
# box_rec.set(yscale ="symlog")
box_tune_recon_detail.tight_layout()
box_tune_recon_detail.savefig(rootsave + "Exp2_SuppFig_tune-recon_detail" + output_type)

box_tune_ml_dense = sns.catplot(
    data=ml_results.loc[
        (ml_results.ML_Algorithm == sel_ml_alg) & (ml_results.Split == "test"), :
    ],
    x="Architecture",
    y="Perf. improvement",
    hue="Latent_Dim",
    col="Latent_Dim",
    palette="pastel",
    order=arch_order,
    kind="box",
    sharey=True,
    aspect=0.8,
)
box_tune_ml_dense.tick_params(axis="x", rotation=90)
# box_tune_ml_dense.set(yscale="symlog")
# Set y-axis limit
box_tune_ml_dense.set(ylim=(-10, 12))
box_tune_ml_dense.tight_layout()
box_tune_ml_dense.savefig(rootsave + "Exp2_Fig2F_tune-ml-linear_dense" + output_type)

box_tune_ml_detail = sns.catplot(
    data=ml_results.loc[ml_results.Split == "test", :],
    x="Architecture",
    y="Perf. improvement",
    hue="Latent_Dim",
    row="Data_set",
    col="ML_Algorithm",
    palette="pastel",
    order=arch_order,
    kind="box",
    sharey=False,
    aspect=2,
)
box_tune_ml_detail.tick_params(axis="x", rotation=90)
box_tune_ml_detail.set(yscale="symlog")
box_tune_ml_detail.savefig(rootsave + "Exp2_SuppFig_tune-ml_detail" + output_type)

### Plot Tuning Parameter distribution
print("Make plots of hyperparam distribution")

df_results_tuning = df_results.melt(
    id_vars=[
        "Config_ID",
        "Architecture",
        "Data_Modalities",
        "Data_set",
        "Latent_Dim",
        "Latent_Coverage",
    ],
    value_vars=["weight_decay", "dropout_all", "encoding_factor", "lr", "fc_n_layers"],
    var_name="Parameter",
    value_name="Value",
)
df_results_tuning["Value"] = df_results_tuning["Value"].apply(
    pd.to_numeric, args=("coerce",)
)

linear_sel = (
    (df_results_tuning.Parameter == "fc_n_layers")
    | (df_results_tuning.Parameter == "encoding_factor")
    | (df_results_tuning.Parameter == "dropout_all")
)

p_hyperparam_lin = (
    so.Plot(
        df_results_tuning.loc[linear_sel, :],
        y="Architecture",
        x="Value",
        color="Latent_Dim",
    )
    .facet(col="Parameter")
    .share(x=False, y=True)
    .add(so.Dash(), so.Agg(), so.Dodge())
    .add(so.Range(), so.Est(errorbar="sd"), so.Dodge())
    .add(so.Dots(alpha=0.1), so.Dodge(), so.Jitter())
    .scale(
        x=so.Continuous(),
        y=so.Nominal(order=arch_order),
        color=so.Nominal(),
    )
    .layout(size=(12, 5))
)

p_hyperparam_log = (
    so.Plot(
        df_results_tuning.loc[~linear_sel, :],
        y="Architecture",
        x="Value",
        color="Latent_Dim",
    )
    .facet(col="Parameter")
    .share(x=False, y=True)
    .add(so.Dash(), so.Agg(), so.Dodge())
    .add(so.Range(), so.Est(errorbar="sd"), so.Dodge())
    .add(so.Dots(alpha=0.1), so.Dodge(), so.Jitter())
    .scale(
        x=so.Continuous(trans="log"),
        y=so.Nominal(order=arch_order),
        color=so.Nominal(),
    )
    .layout(size=(9, 5))
)

p_hyperparam_lin.save(
    rootsave + "Exp2_SuppFig_hyperparam-linear" + output_type, bbox_inches="tight"
)
p_hyperparam_log.save(
    rootsave + "Exp2_SuppFig_hyperparam-log" + output_type, bbox_inches="tight"
)

## save df with results used for plotting
df_results.to_csv(rootsave + "df_results.txt", sep="\t")
ml_results.to_csv(rootsave + "ml_results.txt", sep="\t")
