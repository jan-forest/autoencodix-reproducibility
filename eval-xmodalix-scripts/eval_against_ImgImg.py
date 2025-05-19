import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from matplotlib.cm import viridis
import rasterio
import matplotlib.cm as cm
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

    config_internal.update(
        config
    )  ### Switch order to be able to overwrite internal config params with normal config
    return config_internal


def get_objects(RUN_ID):
    cfg = get_cfg(RUN_ID)
    sample_file = ""
    for d in cfg["DATA_TYPE"]:
        if cfg["DATA_TYPE"][d]["TYPE"] == "IMG":
            sample_file = cfg["DATA_TYPE"][d]["FILE_RAW"]

    reconstruction_imgs = os.path.join("reports", RUN_ID, "Translate_FROM_TO_IMG")
    processed_imgs = os.path.join("data/processed", RUN_ID)
    split_file = os.path.join("data/processed", RUN_ID, "sample_split.parquet")
    mapping_file = os.path.join("data/raw", sample_file)
    sample_mappings = pd.read_csv(mapping_file, sep=cfg["DELIM"], index_col=0)
    sample_mappings["sample_ids"] = sample_mappings.index

    split_df = pd.read_parquet(split_file)
    if not "TCGA" in RUN_ID:
        numeric_index = [
            int(x.replace("T_", "")) for x in sample_mappings.index.tolist()
        ]
        sample_mappings.index = numeric_index
        sample_mappings = sample_mappings.sort_index()
        split_numeric = [int(x.replace("T_", "")) for x in split_df.index.tolist()]
        split_df.index = split_numeric
        split_df = split_df.sort_index()
    return sample_mappings, split_df, reconstruction_imgs, processed_imgs


def save_sample(file_path, outpath=None):
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Read the first band
        norm_data = (data * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
        img = Image.fromarray(norm_data)
        if outpath:
            img.save(outpath)


def read_img(file_path, output_path=None):
    """
    Read a TIFF image file, convert it to an RGB format if needed,
    and optionally save it as a PNG file.

    Args:
        file_path (str): Path to the TIFF file
        output_path (str, optional): Path to save the PNG file. If None, the image is not saved.

    Returns:
        np.ndarray: The image converted to an RGB NumPy array
    """
    if "tif" in file_path:
        with rasterio.open(file_path) as src:
            data = src.read(1)
    else:
        img = Image.open(file_path)
        data = np.array(img)
    # be sure both values are between 0 and 1
    if data.max() > 1:
        data = data / 255

    return data


def compute_mse(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for MSE calculation.")
    mse = np.mean((image1 - image2) ** 2)
    return mse


def predict_images_and_calculate_mse(mappings, processed_path):
    mse_values = []
    out_orig = "predict_orig_sample.png"
    out_rec = "predict_rec_sample.png"
    for _, row in mappings.iterrows():
        orgimg_path = os.path.join(processed_path, row["img_paths"])
        file_ext = ".png" if "TCGA" in RUN_ID else ".tif"
        img_img_path = os.path.join(
            "reports", RUN_ID, "Reference_TO_TO_IMG", row["sample_ids"] + file_ext
        )

        img_raw = read_img(orgimg_path, out_orig)
        rec_array = read_img(img_img_path, out_rec)
        mse = compute_mse(img_raw, rec_array)
        mse_values.append(mse)

    return mse_values


def process_images_and_calculate_mse(mappings, processed_path, rec_path):
    mse_values = []
    for _, row in mappings.iterrows():
        orgimg_path = os.path.join(processed_path, row["img_paths"])
        file_ext = ".png" if "TCGA" in RUN_ID else ".tif"
        recimg_path = os.path.join(rec_path, row["sample_ids"] + file_ext)

        img_array = read_img(orgimg_path)
        rec_array = read_img(recimg_path)
        mse = compute_mse(img_array, rec_array)
        mse_values.append(mse)

    mappings["mse"] = mse_values
    return mse_values


def create_boxplot(df, split_name, run_id):

    if split_name != "overall":
        df = df[split_df["SPLIT"] == split_name]
    mse_values = [df["mse_xmodale"], df["mse_predict_img"], df["mse_pure"]]
    mse_labels = ["xmodale", "img_img", "img_img_pure"]

    colormap = cm.get_cmap("viridis")
    colors = colormap(np.linspace(0, 1, len(mse_labels)))

    plt.figure(figsize=(10, 6))

    boxprops = dict(linestyle="-", linewidth=2)
    medianprops = dict(color="k", linewidth=2)
    whiskerprops = dict(color="k", linewidth=2)

    box = plt.boxplot(
        mse_values,
        labels=mse_labels,
        patch_artist=True,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    plt.title(f"MSE Distribution - {split_name.capitalize()} Split", fontsize=16)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # plt.tight_layout()
    plot_path_boxplot = os.path.join(
        "reports",
        "paper-visualizations",
        run_id.split("_")[0],
        f"xmodal_vs_normal_{split_name}_boxplot.png",
    )
    plt.savefig(plot_path_boxplot)
    plt.close()
    print(f"Saved {split_name} boxplot to {plot_path_boxplot}")
    # save underlying data for plots:
    df.to_csv(
        os.path.join(
            "reports",
            "paper-visualizations",
            run_id.split("_")[0],
            f"MSE_xmodale_plotdata_{split_name}.csv",
        )
    )


def create_split_df_and_plot(df, split_name, run_id, split_df):
    if split_name != "overall":
        df = df[split_df["SPLIT"] == split_name]

    sum_row = df[["mse_predict_img", "mse_xmodale", "mse_pure"]].sum()
    sum_row = sum_row.reindex(df.columns, fill_value=np.nan)
    sum_row.name = "Sum"

    # Concatenate the sum row to the DataFrame
    df_with_sum = pd.concat([df, sum_row.to_frame().T])

    output_path = os.path.join(
        "reports",
        "paper-visualizations",
        run_id.split("_")[0],
        f"MSE_xmodale_{split_name}.csv",
    )
    df_with_sum.to_csv(output_path)

    # Check if TCGA is in the RUN_ID
    is_tcga = "TCGA" in run_id

    # Aggregate data for TCGA case
    if is_tcga:
        df_aggregated = (
            df.groupby("extra_class_labels")
            .agg({"mse_xmodale": "mean", "mse_predict_img": "mean", "mse_pure": "mean"})
            .reset_index()
        )
        x_values = df_aggregated["extra_class_labels"]
        df = df_aggregated
    else:
        x_values = df.index

    # Original Line Plot with Dotted Lines
    plt.figure(figsize=(20, 8))
    plt.plot(
        x_values,
        df["mse_xmodale"],
        label="xmodale",
        linestyle="-",
        marker="o",
        linewidth=2,
        color=viridis(0.1),
        markersize=5,
    )
    plt.plot(
        x_values,
        df["mse_predict_img"],
        label="img_img",
        linestyle="-",
        marker="s",
        linewidth=2,
        color=viridis(0.5),
        markersize=5,
    )
    plt.plot(
        x_values,
        df["mse_pure"],
        label="img_img_pure",
        linestyle="-",
        marker="x",
        linewidth=2,
        color=viridis(0.9),
        markersize=5,
    )

    plt.title(f"MSE Comparison - {split_name.capitalize()} Split", fontsize=16)
    plt.xlabel("Class Label" if is_tcga else "Timepoint", fontsize=12)
    plt.ylabel("Mean MSE" if is_tcga else "MSE", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper right", fontsize=12)
    plt.xticks(x_values, fontsize=10, rotation=45)
    plt.yticks(fontsize=10)

    # plt.tight_layout()
    plot_path_line = os.path.join(
        "reports",
        "paper-visualizations",
        run_id.split("_")[0],
        f"xmodal_vs_normal_{split_name}_line.png",
    )
    plt.savefig(plot_path_line)
    plt.close()
    print(f"Saved {split_name} line plot to {plot_path_line}")

    # Scatter Plot
    plt.figure(figsize=(20, 8))
    plt.scatter(
        x_values,
        df["mse_xmodale"],
        label="xmodale",
        marker="o",
        color=viridis(0.1),
        s=100 if is_tcga else 10,
    )
    plt.scatter(
        x_values,
        df["mse_predict_img"],
        label="img_img",
        marker="s",
        color=viridis(0.5),
        s=100 if is_tcga else 10,
    )
    plt.scatter(
        x_values,
        df["mse_pure"],
        label="img_img_pure",
        marker="x",
        color=viridis(0.9),
        s=100 if is_tcga else 10,
    )

    plt.title(
        f"MSE Comparison - {split_name.capitalize()} Split (Scatter)", fontsize=16
    )
    plt.xlabel("Class Label" if is_tcga else "Timepoint", fontsize=12)
    plt.ylabel("Mean MSE" if is_tcga else "MSE", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper right", fontsize=12)

    plt.xticks(x_values, fontsize=10, rotation=45)
    plt.yticks(fontsize=10)

    # plt.tight_layout()
    plot_path_scatter = os.path.join(
        "reports",
        "paper-visualizations",
        run_id.split("_")[0],
        f"xmodal_vs_normal_{split_name}_scatter.png",
    )
    plt.savefig(plot_path_scatter)
    plt.close()
    print(f"Saved {split_name} scatter plot to {plot_path_scatter}")

    # Bar Plot
    plt.figure(figsize=(20, 8))
    bar_width = 0.25
    indices = np.arange(len(x_values))

    plt.bar(
        indices - bar_width,
        df["mse_xmodale"],
        width=bar_width,
        label="xmodale",
        color=viridis(0.1),
    )
    plt.bar(
        indices,
        df["mse_predict_img"],
        width=bar_width,
        label="img_img",
        color=viridis(0.5),
    )
    plt.bar(
        indices + bar_width,
        df["mse_pure"],
        width=bar_width,
        label="img_img_pure",
        color=viridis(0.9),
    )

    plt.title(f"MSE Comparison - {split_name.capitalize()} Split (Bar)", fontsize=16)
    plt.xlabel("Class Label" if is_tcga else "Timepoint", fontsize=12)
    plt.ylabel("Mean MSE" if is_tcga else "MSE", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper right", fontsize=12)
    plt.xticks(indices, x_values, fontsize=10, rotation=45)
    plt.yticks(fontsize=10)

    # plt.tight_layout()
    plot_path_bar = os.path.join(
        "reports",
        "paper-visualizations",
        run_id.split("_")[0],
        f"xmodal_vs_normal_{split_name}_bar.png",
    )
    plt.savefig(plot_path_bar)
    plt.close()
    print(f"Saved {split_name} bar plot to {plot_path_bar}")

    # Stem Plot
    plt.figure(figsize=(20, 8))
    plt.stem(
        x_values,
        df["mse_xmodale"],
        label="xmodale",
        linefmt=":",
        markerfmt="o",
        basefmt=" ",
        use_line_collection=True,
    )
    plt.stem(
        x_values,
        df["mse_predict_img"],
        label="img_img",
        linefmt=":",
        markerfmt="s",
        basefmt=" ",
        use_line_collection=True,
    )
    plt.stem(
        x_values,
        df["mse_pure"],
        label="img_img_pure",
        linefmt=":",
        markerfmt="x",
        basefmt=" ",
        use_line_collection=True,
    )

    plt.title(f"MSE Comparison - {split_name.capitalize()} Split (Stem)", fontsize=16)
    plt.xlabel("Class Label" if is_tcga else "Timepoint", fontsize=12)
    plt.ylabel("Mean MSE" if is_tcga else "MSE", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper right", fontsize=12)
    plt.xticks(x_values, fontsize=10, rotation=45)
    plt.yticks(fontsize=10)

    # plt.tight_layout()
    plot_path_stem = os.path.join(
        "reports",
        "paper-visualizations",
        run_id.split("_")[0],
        f"xmodal_vs_normal_{split_name}_stem.png",
    )
    plt.savefig(plot_path_stem)
    plt.close()
    print(f"Saved {split_name} stem plot to {plot_path_stem}")


if __name__ == "__main__":
    RUN_ID = sys.argv[1]
    cfg = get_cfg(RUN_ID)
    translated_mappings, split_df, translated_imgs, processed_imgs = get_objects(RUN_ID)
    compare_mappings, compare_split, compare_imgs, compare_processed_imgs = get_objects(
        f"{RUN_ID}ImgImg"
    )
    print("Same split in comparison run with suffix ImgImg:")
    print((compare_split["SPLIT"] == split_df["SPLIT"]).sum() == len(compare_split))
    mse_predict = predict_images_and_calculate_mse(
        mappings=compare_mappings, processed_path=compare_processed_imgs
    )
    mse_xmodale = process_images_and_calculate_mse(
        mappings=translated_mappings,
        processed_path=processed_imgs,
        rec_path=translated_imgs,
    )
    mse_pure = process_images_and_calculate_mse(
        mappings=compare_mappings,
        processed_path=compare_processed_imgs,
        rec_path=compare_imgs,
    )
    translated_mappings["mse_predict_img"] = mse_predict
    translated_mappings["mse_xmodale"] = mse_xmodale
    translated_mappings["mse_pure"] = mse_pure

    split_df = split_df.sort_index()

    for split in ["train", "valid", "test", "overall"]:
        create_boxplot(translated_mappings, split, RUN_ID)
        create_split_df_and_plot(translated_mappings, split, RUN_ID, split_df)
