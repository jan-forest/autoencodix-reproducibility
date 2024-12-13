import pandas as pd
import seaborn as sns
from src.utils.utils_basic import annealer
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import yaml

rootdir = "./"
output_type = ".png"

print("Make Exp1 Paper plots - SC")
sns.set_context("notebook")
sns.set_theme(style="white")

### SC
df_loss = pd.read_parquet(
    rootdir + "reports/Exp1_SC_Annealing/losses_METH_RNA_varix.parquet"
)
df_cov = pd.read_parquet(
    rootdir + "reports/Exp1_SC_Annealing/latent_cov_per_epoch_Exp1_SC_Annealing.parquet"
)

config_path = rootdir + "reports/Exp1_SC_Annealing/Exp1_SC_Annealing_config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)
beta = config["BETA"]
df_cov["beta"] = [
    beta
    * annealer(epoch_current=e, total_epoch=len(df_cov.epoch), func="5phase-constant")
    for e in range(0, len(df_cov.epoch))
]

df_cov["Recon. Loss"] = df_loss["valid_recon_loss"]


fig, axs = plt.subplots(nrows=4, figsize=(12, 6))

exclude_epochs = int(df_cov.shape[0] / (5 * 10))
cmin_recon = df_cov.loc[range(exclude_epochs, df_cov.shape[0]), "Recon. Loss"].min()
cmax_recon = df_cov.loc[range(exclude_epochs, df_cov.shape[0]), "Recon. Loss"].max()

sns.heatmap(
    pd.DataFrame(df_cov.set_index("epoch")["beta"]).T,
    cmap="Greys",
    cbar_kws={"location": "top", "orientation": "horizontal", "label": "Beta-value"},
    xticklabels=False,
    yticklabels=False,
    ax=axs[0],
    norm=SymLogNorm(linthresh=0.01),
).set(xlabel=None)
sns.heatmap(
    pd.DataFrame(df_cov.set_index("epoch")["Recon. Loss"]).T,
    cmap="Oranges",
    cbar_kws={"location": "top", "orientation": "horizontal", "label": "Recon. loss"},
    vmin=cmin_recon,
    vmax=cmax_recon,
    ax=axs[1],
    xticklabels=False,
    yticklabels=False,
).set(xlabel=None)
sns.heatmap(
    pd.DataFrame(df_cov.set_index("epoch")["coverage"]).T,
    cmap="Blues",
    cbar_kws={"location": "top", "orientation": "horizontal", "label": "Coverage"},
    ax=axs[2],
    xticklabels=False,
    yticklabels=False,
).set(xlabel=None)
sns.heatmap(
    pd.DataFrame(df_cov.set_index("epoch")["total_correlation"]).T,
    cmap="Greens",
    cbar_kws={
        "location": "top",
        "orientation": "horizontal",
        "label": "Total correlation",
    },
    ax=axs[3],
    xticklabels=200,
    yticklabels=False,
)

fig.tight_layout()
fig.savefig("./reports/paper-visualizations/Exp1/Exp1_SC_summary" + output_type)

print("Make Exp1 Paper plots - TCGA")
### TCGA
df_loss = pd.read_parquet(
    rootdir + "reports/Exp1_TCGA_Annealing/losses_METH_MUT_RNA_varix.parquet"
)
df_cov = pd.read_parquet(
    rootdir
    + "reports/Exp1_TCGA_Annealing/latent_cov_per_epoch_Exp1_TCGA_Annealing.parquet"
)

config_path = rootdir + "reports/Exp1_TCGA_Annealing/Exp1_TCGA_Annealing_config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)
beta = config["BETA"]
df_cov["beta"] = [
    beta
    * annealer(epoch_current=e, total_epoch=len(df_cov.epoch), func="5phase-constant")
    for e in range(0, len(df_cov.epoch))
]

df_cov["Recon. Loss"] = df_loss["valid_recon_loss"]


fig, axs = plt.subplots(nrows=4, figsize=(12, 6))

exclude_epochs = int(df_cov.shape[0] / (5 * 10))
cmin_recon = df_cov.loc[range(exclude_epochs, df_cov.shape[0]), "Recon. Loss"].min()
cmax_recon = df_cov.loc[range(exclude_epochs, df_cov.shape[0]), "Recon. Loss"].max()

sns.heatmap(
    pd.DataFrame(df_cov.set_index("epoch")["beta"]).T,
    cmap="Greys",
    cbar_kws={"location": "top", "orientation": "horizontal", "label": "Beta-value"},
    xticklabels=False,
    yticklabels=False,
    ax=axs[0],
    norm=SymLogNorm(linthresh=0.01),
).set(xlabel=None)
sns.heatmap(
    pd.DataFrame(df_cov.set_index("epoch")["Recon. Loss"]).T,
    cmap="Oranges",
    cbar_kws={"location": "top", "orientation": "horizontal", "label": "Recon. loss"},
    vmin=cmin_recon,
    vmax=cmax_recon,
    ax=axs[1],
    xticklabels=False,
    yticklabels=False,
).set(xlabel=None)
sns.heatmap(
    pd.DataFrame(df_cov.set_index("epoch")["coverage"]).T,
    cmap="Blues",
    cbar_kws={"location": "top", "orientation": "horizontal", "label": "Coverage"},
    ax=axs[2],
    xticklabels=False,
    yticklabels=False,
).set(xlabel=None)
sns.heatmap(
    pd.DataFrame(df_cov.set_index("epoch")["total_correlation"]).T,
    cmap="Greens",
    cbar_kws={
        "location": "top",
        "orientation": "horizontal",
        "label": "Total correlation",
    },
    ax=axs[3],
    xticklabels=200,
    yticklabels=False,
)

fig.tight_layout()
fig.savefig("./reports/paper-visualizations/Exp1/Exp1_TCGA_summary" + output_type)
