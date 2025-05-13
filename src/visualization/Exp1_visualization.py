import pandas as pd
import seaborn as sns
from src.utils.utils_basic import annealer
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import yaml

# rootdir = "./"
rootdir = "/mnt/c/Users/ewald/Nextcloud/eigene_shares/AutoEncoderOmics/SaveResults/250507_first_revision_results/"
output_type = ".png"

print("Make Exp1 Paper plots - SC")
sns.set_context("notebook")
sns.set_theme(style="whitegrid")

### SC

## Loop over all beta values
df_cov = pd.DataFrame()

beta_strings = ["beta0", "beta1e-02", "beta1e-01", "beta1e00", "beta1e01"]
for b_string in beta_strings:
    run_id = "Exp1_SC_Annealing_" + b_string
    df_loss = pd.read_parquet(
        rootdir + "reports/" + run_id + "/" + "losses_METH_RNA_varix.parquet"
    )
    df_cov_temp = pd.read_parquet(
        rootdir
        + "reports/"
        + run_id
        + "/"
        + "/latent_cov_per_epoch_"
        + run_id
        + ".parquet"
    )

    config_path = rootdir + "reports/" + run_id + "/" + run_id + "_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    beta = config["BETA"]
    df_cov_temp["beta"] = [
        beta
        * annealer(
            epoch_current=e, total_epoch=len(df_cov_temp.epoch), func="logistic-mid"
        )
        for e in range(0, len(df_cov_temp.epoch))
    ]
    df_cov_temp["beta_final"] = [beta for e in range(0, len(df_cov_temp.epoch))]
    df_cov_temp["beta_string"] = [b_string for e in range(0, len(df_cov_temp.epoch))]

    # df_cov_temp["Recon. r2"] = df_loss["valid_recon_loss"]
    df_cov_temp["Recon. r2"] = df_loss["valid_r2"]  ## Switch to R2
    # Append df_cov_temp to df_cov
    df_cov = pd.concat([df_cov, df_cov_temp], ignore_index=True)


## Define min and max values across all beta values

exclude_epochs = int(df_cov.shape[0] / (2 * 10))
cmin_recon = df_cov.loc[range(exclude_epochs, df_cov.shape[0]), "Recon. r2"].min()
# cmin_recon = 0.1 ## Set to -1 for R2
cmax_recon = df_cov.loc[range(exclude_epochs, df_cov.shape[0]), "Recon. r2"].max()
# cmax_recon = 0.25

bmin = min(df_cov["beta"])
bmax = max(df_cov["beta"])

tc_min = df_cov.loc[range(exclude_epochs, df_cov.shape[0]), "total_correlation"].min()
tc_max = df_cov.loc[range(exclude_epochs, df_cov.shape[0]), "total_correlation"].max()

cov_min = df_cov.loc[range(exclude_epochs, df_cov.shape[0]), "coverage"].min()
cov_max = df_cov.loc[range(exclude_epochs, df_cov.shape[0]), "coverage"].max()

fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(12, 6))

for b_index, b_string in enumerate(beta_strings):
    df_cov_temp = df_cov[df_cov["beta_string"] == b_string]
    sns.heatmap(
        pd.DataFrame(df_cov_temp.set_index("epoch")["beta"]).T,
        cmap="Greys",
        cbar=(b_index == 0),
        cbar_kws=(
            {"location": "left", "orientation": "vertical", "label": None}
            if b_index == 0
            else None
        ),
        xticklabels=False,
        yticklabels=False,
        ax=axs[0][b_index],
        norm=SymLogNorm(linthresh=0.01, vmin=bmin, vmax=bmax),
    ).set(xlabel=None)

    sns.heatmap(
        pd.DataFrame(df_cov_temp.set_index("epoch")["Recon. r2"]).T,
        cmap="Oranges",
        cbar=(b_index == 0),
        cbar_kws=(
            {"location": "left", "orientation": "vertical", "label": None}
            if b_index == 0
            else None
        ),
        vmin=cmin_recon,
        vmax=cmax_recon,
        ax=axs[1][b_index],
        xticklabels=False,
        yticklabels=False,
    ).set(xlabel=None)
    sns.heatmap(
        pd.DataFrame(df_cov_temp.set_index("epoch")["coverage"]).T,
        cmap="Blues",
        cbar=(b_index == 0),
        cbar_kws=(
            {"location": "left", "orientation": "vertical", "label": None}
            if b_index == 0
            else None
        ),
        ax=axs[2][b_index],
        vmin=cov_min,
        vmax=cov_max,
        xticklabels=False,
        yticklabels=False,
    ).set(xlabel=None)
    sns.heatmap(
        pd.DataFrame(df_cov_temp.set_index("epoch")["total_correlation"]).T,
        cmap="Greens",
        cbar=(b_index == 0),
        cbar_kws=(
            {
                "location": "left",
                "orientation": "vertical",
                "label": None,
            }
            if b_index == 0
            else None
        ),
        ax=axs[3][b_index],
        vmin=tc_min,
        vmax=tc_max,
        xticklabels=250,
        yticklabels=False,
    )
    # Drawing the frames
    for i in range(axs.shape[0]):
        for _, spine in axs[i][b_index].spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1)


fig.tight_layout()
fig.savefig("./reports/paper-visualizations/Exp1/Exp1_SC_summary" + output_type)

#########################################
print("Make Exp1 Paper plots - TCGA")
### TCGA

## Loop over all beta values
df_cov = pd.DataFrame()

beta_strings = ["beta0", "beta1e-02", "beta1e-01", "beta1e00", "beta1e01"]
for b_string in beta_strings:
    run_id = "Exp1_TCGA_Annealing_" + b_string
    df_loss = pd.read_parquet(
        rootdir + "reports/" + run_id + "/" + "losses_METH_MUT_RNA_varix.parquet"
    )
    df_cov_temp = pd.read_parquet(
        rootdir
        + "reports/"
        + run_id
        + "/"
        + "/latent_cov_per_epoch_"
        + run_id
        + ".parquet"
    )

    config_path = rootdir + "reports/" + run_id + "/" + run_id + "_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    beta = config["BETA"]
    df_cov_temp["beta"] = [
        beta
        * annealer(
            epoch_current=e, total_epoch=len(df_cov_temp.epoch), func="logistic-mid"
        )
        for e in range(0, len(df_cov_temp.epoch))
    ]
    df_cov_temp["beta_final"] = [beta for e in range(0, len(df_cov_temp.epoch))]
    df_cov_temp["beta_string"] = [b_string for e in range(0, len(df_cov_temp.epoch))]

    # df_cov_temp["Recon. r2"] = df_loss["valid_recon_loss"]
    df_cov_temp["Recon. r2"] = df_loss["valid_r2"]  ## Switch to R2
    # Append df_cov_temp to df_cov
    df_cov = pd.concat([df_cov, df_cov_temp], ignore_index=True)


## Define min and max values across all beta values

exclude_epochs = int(df_cov.shape[0] / (2 * 10))
cmin_recon = df_cov.loc[range(exclude_epochs, df_cov.shape[0]), "Recon. r2"].min()
# cmin_recon = 0.1 ## Set to -1 for R2
cmax_recon = df_cov.loc[range(exclude_epochs, df_cov.shape[0]), "Recon. r2"].max()
# cmax_recon = 0.25

bmin = min(df_cov["beta"])
bmax = max(df_cov["beta"])

tc_min = df_cov.loc[range(exclude_epochs, df_cov.shape[0]), "total_correlation"].min()
tc_max = df_cov.loc[range(exclude_epochs, df_cov.shape[0]), "total_correlation"].max()

cov_min = df_cov.loc[range(exclude_epochs, df_cov.shape[0]), "coverage"].min()
cov_max = df_cov.loc[range(exclude_epochs, df_cov.shape[0]), "coverage"].max()

fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(12, 6))

for b_index, b_string in enumerate(beta_strings):
    df_cov_temp = df_cov[df_cov["beta_string"] == b_string]
    sns.heatmap(
        pd.DataFrame(df_cov_temp.set_index("epoch")["beta"]).T,
        cmap="Greys",
        cbar=(b_index == 0),
        cbar_kws=(
            {"location": "left", "orientation": "vertical", "label": None}
            if b_index == 0
            else None
        ),
        xticklabels=False,
        yticklabels=False,
        ax=axs[0][b_index],
        norm=SymLogNorm(linthresh=0.01, vmin=bmin, vmax=bmax),
    ).set(xlabel=None)
    sns.heatmap(
        pd.DataFrame(df_cov_temp.set_index("epoch")["Recon. r2"]).T,
        cmap="Oranges",
        cbar=(b_index == 0),
        cbar_kws=(
            {"location": "left", "orientation": "vertical", "label": None}
            if b_index == 0
            else None
        ),
        vmin=cmin_recon,
        vmax=cmax_recon,
        ax=axs[1][b_index],
        xticklabels=False,
        yticklabels=False,
    ).set(xlabel=None)
    sns.heatmap(
        pd.DataFrame(df_cov_temp.set_index("epoch")["coverage"]).T,
        cmap="Blues",
        cbar=(b_index == 0),
        cbar_kws=(
            {"location": "left", "orientation": "vertical", "label": None}
            if b_index == 0
            else None
        ),
        ax=axs[2][b_index],
        vmin=cov_min,
        vmax=cov_max,
        xticklabels=False,
        yticklabels=False,
    ).set(xlabel=None)
    sns.heatmap(
        pd.DataFrame(df_cov_temp.set_index("epoch")["total_correlation"]).T,
        cmap="Greens",
        cbar=(b_index == 0),
        cbar_kws=(
            {
                "location": "left",
                "orientation": "vertical",
                "label": None,
            }
            if b_index == 0
            else None
        ),
        ax=axs[3][b_index],
        vmin=tc_min,
        vmax=tc_max,
        xticklabels=250,
        yticklabels=False,
    )
    # Drawing the frames
    for i in range(axs.shape[0]):
        for _, spine in axs[i][b_index].spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1)

fig.tight_layout()
fig.savefig("./reports/paper-visualizations/Exp1/Exp1_TCGA_summary" + output_type)
