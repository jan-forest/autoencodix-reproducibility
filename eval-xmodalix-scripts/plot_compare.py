import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from matplotlib.gridspec import GridSpec

import yaml
import warnings
import matplotlib


warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore", message=".*not compatible with tight_layout.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", category=matplotlib.MatplotlibDeprecationWarning
)  # Add this line to suppress MatplotlibDeprecationWarning
warnings.filterwarnings(
    "ignore", message=".*not compatible with tight_layout.*", category=UserWarning
)


def get_cfg(run_id):
    """A function to read YAML file
    Args:
        rund_id (str): ID of the run
    Returns:
        config (dict): a dictionary of configuration parameters

    """
    with open(os.path.join("reports", f"{run_id}", f"{run_id}_config.yaml")) as f:
        config = yaml.safe_load(f)
    config["RUN_ID"] = run_id

    with open(os.path.join("src", "000_internal_config.yaml")) as f:
        config_internal = yaml.safe_load(f)

    # config.update(config_internal)
    # return config
    config_internal.update(
        config
    )  ### Switch order to be able to overwrite internal config params with normal config
    return config_internal


def load_image(path):
    """Reads tiff image and returns as numpy array
    Args:
        path (str): Path to tiff image
    Returns:
        np.array: Image as numpy array

    """
    with rasterio.open(path) as src:
        img = src.read()
        img = np.transpose(img, (1, 2, 0))
        if img.shape[2] > 3:
            img = img[:, :, :3]
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        return img


def create_plot(samples, split):
    """Create a grid of images for comparison
    Args:
        samples (pd.DataFrame): DataFrame containing sample information
        split (str): Split type (train, valid, test)
    Returns:
        None
    """
    fig = plt.figure(
        figsize=(20, 8)
    )  # Increased figure width to accommodate new columns
    gs = GridSpec(4, 8, figure=fig, hspace=0.6, wspace=0.1)  # Changed to 8 columns

    for i, (_, row) in enumerate(samples.iterrows()):
        # Original image
        ax1 = fig.add_subplot(gs[i, 0:2])
        img_orig = load_image(os.path.join(processed_imgs, row["img_paths"]))
        ax1.imshow(img_orig)
        ax1.axis("off")
        if i == 0:
            ax1.set_title("Original", fontsize=16, fontweight="bold", pad=20)
        ax1.text(
            0.5,
            -0.1,
            row["SAMPLE_ID"],
            transform=ax1.transAxes,
            fontsize=10,
            ha="center",
            va="top",
        )

        # Reconstructed image
        ax2 = fig.add_subplot(gs[i, 2:4])
        img_recon = load_image(
            os.path.join(reconstruction_imgs, row["sample_ids"] + ".tif")
        )
        ax2.imshow(img_recon)
        ax2.axis("off")
        if i == 0:
            ax2.set_title("Reconstructed", fontsize=16, fontweight="bold", pad=20)
        ax2.text(
            0.5,
            -0.1,
            row["sample_ids"],
            transform=ax2.transAxes,
            fontsize=10,
            ha="center",
            va="top",
        )

        # Reconstructed image from img folder
        ax3 = fig.add_subplot(gs[i, 4:6])
        img_recon_img = load_image(
            os.path.join(reconstruction_imgs_img, row["sample_ids"] + ".tif")
        )
        ax3.imshow(img_recon_img)
        ax3.axis("off")
        if i == 0:
            ax3.set_title("Reconstructed (IMG)", fontsize=16, fontweight="bold", pad=20)
        ax3.text(
            0.5,
            -0.1,
            row["sample_ids"],
            transform=ax3.transAxes,
            fontsize=10,
            ha="center",
            va="top",
        )

        # Reconstructed image from pure folder
        ax4 = fig.add_subplot(gs[i, 6:8])
        img_recon_pure = load_image(
            os.path.join(reconstruction_img_pure, row["sample_ids"] + ".tif")
        )
        ax4.imshow(img_recon_pure)
        ax4.axis("off")
        if i == 0:
            ax4.set_title(
                "Reconstructed (Pure)", fontsize=16, fontweight="bold", pad=20
            )
        ax4.text(
            0.5,
            -0.1,
            row["sample_ids"],
            transform=ax4.transAxes,
            fontsize=10,
            ha="center",
            va="top",
        )

        if i == 0:
            ax1.text(
                -0.1,
                1.15,
                row["quartile"],
                transform=ax1.transAxes,
                fontsize=16,
                fontweight="bold",
                va="top",
                ha="right",
            )
        else:
            ax1.text(
                -0.1,
                1.05,
                row["quartile"],
                transform=ax1.transAxes,
                fontsize=16,
                fontweight="bold",
                va="top",
                ha="right",
            )

    fig.text(
        0.5,
        0.02,
        "Each row shows a randomly picked image and its reconstructions for each Quartile",
        ha="center",
        va="center",
        fontsize=12,
        style="italic",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    # plt.tight_layout()
    plt.savefig(
        os.path.join(
            "reports",
            "paper-visualizations",
            RUN_ID.split("_")[0],
            f"image_comparison_grid_{split}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    RUN_ID = sys.argv[1]
    cfg = get_cfg(RUN_ID)
    sample_file = ""
    for d in cfg["DATA_TYPE"]:
        if cfg["DATA_TYPE"][d]["TYPE"] == "IMG":
            sample_file = cfg["DATA_TYPE"][d]["FILE_RAW"]

    reconstruction_imgs = os.path.join("reports", RUN_ID, "Translate_FROM_TO_IMG")
    reconstruction_imgs_img = os.path.join("reports", RUN_ID, "Reference_TO_TO_IMG")
    reconstruction_img_pure = os.path.join("reports", f"{RUN_ID}ImgImg", "Translate_FROM_TO_IMG")
    processed_imgs = os.path.join("data/processed", RUN_ID)
    split_file = os.path.join("data/processed", RUN_ID, "sample_split.parquet")
    mapping_file = os.path.join("data/raw", sample_file)
    sample_mappings = pd.read_csv(mapping_file, sep=cfg["DELIM"])
    sample_mappings["quartile"] = pd.qcut(
        sample_mappings["extra_class_labels"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    sample = sample_mappings.groupby("quartile").apply(
        lambda x: x.sample(n=1, random_state=cfg["GLOBAL_SEED"])
    )
    split_df = pd.read_parquet(split_file)
    split_df.index.name = "index"
    # Set style for a clean, modern look
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["font.sans-serif"] = [
        "DejaVu Sans",
        "Helvetica",
        "Arial",
        "sans-serif",
    ]

    sample_mappings = sample_mappings.merge(
        split_df[["SAMPLE_ID", "SPLIT"]], left_on="sample_ids", right_on="SAMPLE_ID"
    )

    sample_mappings["quartile"] = pd.qcut(
        sample_mappings["extra_class_labels"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    )

    def select_samples(group, n=1):
        return group.sample(n=n, random_state=cfg["GLOBAL_SEED"])

    for split in ["train", "valid", "test"]:
        split_samples = sample_mappings[sample_mappings["SPLIT"] == split]
        if not split_samples.empty:
            samples = (
                split_samples.groupby("quartile")
                .apply(select_samples)
                .reset_index(drop=True)
            )
            create_plot(samples, split)
        else:
            print(f"No samples found for {split} split")

    print(
        "Plots have been saved as image_comparison_grid_train.png, image_comparison_grid_valid.png, and image_comparison_grid_test.png"
    )
