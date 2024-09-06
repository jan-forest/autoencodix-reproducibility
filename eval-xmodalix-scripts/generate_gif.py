import yaml
import os

import sys
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import imageio
import rasterio
import numpy as np
import matplotlib
import warnings

# Suppress specific warnings
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


def overlay_text(image, text, position, font_size=20, font_color="red"):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(position, text, fill=font_color, font=font)
    return image


def read_and_normalize_tif(path):
    with rasterio.open(path) as src:
        img = src.read()
        img = np.transpose(img, (1, 2, 0))
        if img.shape[2] > 3:
            img = img[:, :, :3]
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        # return as Image object
        return Image.fromarray((img * 255).astype(np.uint8))


def create_gif(df, gif_path, reduce_colors):
    frames = []

    for index, row in df.iterrows():
        sample_id = row["sample_ids"]
        rec_name = f"{sample_id}.tif" if not "TCGA" in RUN_ID else f"{sample_id}.png"
        orig_name = row["img_paths"]

        split_info = split_df.loc[sample_id, "SPLIT"]
        timepoint = row["extra_class_labels"]

        rec_path = os.path.join(reconstructed_folder, rec_name)
        orig_path = os.path.join(original_folder, orig_name)

        rec_img = read_and_normalize_tif(rec_path)
        orig_img = read_and_normalize_tif(orig_path)

        # be sure images have same color map
        rec_image_gray = rec_img.convert("L")
        orig_img_gray = orig_img.convert("L")

        if rec_img.size != orig_img.size:
            rec_img = rec_img.resize(orig_img.size)
        combined_img = Image.new(
            "RGB", (orig_img.width + rec_img.width, orig_img.height)
        )
        combined_img.paste(orig_img_gray, (0, 0))
        combined_img.paste(rec_image_gray, (orig_img.width, 0))

        # Reduce the number of colors if required
        if reduce_colors:
            combined_img = combined_img.convert("P", palette=Image.ADAPTIVE, colors=255)
            # Overlay the split information and labels on the combined image
        combined_img = overlay_text(
            combined_img, f"Split: {split_info} T: {timepoint}", (10, 10)
        )
        combined_img = overlay_text(
            combined_img, "Original", (10, orig_img.height - 30)
        )
        combined_img = overlay_text(
            combined_img, "Reconstruction", (orig_img.width + 10, orig_img.height - 30)
        )

        # Add combined image to frames list
        frames.append(combined_img)

    # Save frames as a GIF with optimization
    imageio.mimsave(
        os.path.join(result_dir, gif_path),
        frames,
        format="GIF",
        fps=2,
        palettesize=256,
        loop=0,
    )  # Adjust duration as needed


if __name__ == "__main__":
    RUN_ID = sys.argv[1]
    # RUN_ID="CelegansIterations10"
    cfg = get_cfg(run_id=RUN_ID)
    original_folder = os.path.join("data/processed", RUN_ID)
    reconstructed_folder = os.path.join("reports", RUN_ID, "IMGS")

    global result_dir
    result_dir = os.path.join("reports", "paper-visualizations", RUN_ID.split("_")[0])
    for d in cfg["DATA_TYPE"]:
        if cfg["DATA_TYPE"][d]["TYPE"] == "ANNOTATION":
            mapping_file_path = os.path.join(
                cfg["ROOT_RAW"], cfg["DATA_TYPE"][d]["FILE_RAW"]
            )

    split_file_path = os.path.join("data/processed", RUN_ID, "sample_split.parquet")

    mapping_df = pd.read_csv(mapping_file_path, sep="\t")

    # Read the split information from the parquet file
    split_df = pd.read_parquet(split_file_path)
    # Filter DataFrames by split
    output_gif_paths = {
        "train": ("train_reduced.gif", "train_full.gif"),
        "test": ("test_reduced.gif", "test_full.gif"),
        "valid": ("valid_reduced.gif", "valid_full.gif"),
        "all": ("all_reduced.gif", "all_full.gif"),
    }
    train_df = mapping_df[
        mapping_df["sample_ids"].isin(split_df[split_df["SPLIT"] == "train"].index)
    ]
    test_df = mapping_df[
        mapping_df["sample_ids"].isin(split_df[split_df["SPLIT"] == "test"].index)
    ]
    valid_df = mapping_df[
        mapping_df["sample_ids"].isin(split_df[split_df["SPLIT"] == "valid"].index)
    ]
    all_df = mapping_df  # Use all images

    # Create GIFs
    # create_gif(train_df, output_gif_paths["train"][0], reduce_colors=True)
    create_gif(train_df, output_gif_paths["train"][1], reduce_colors=False)
    # create_gif(test_df, output_gif_paths["test"][0], reduce_colors=True)
    create_gif(test_df, output_gif_paths["test"][1], reduce_colors=False)
    # create_gif(valid_df, output_gif_paths["valid"][0], reduce_colors=True)
    create_gif(valid_df, output_gif_paths["valid"][1], reduce_colors=False)
    # create_gif(all_df, output_gif_paths["all"][0], reduce_colors=True)
    create_gif(all_df, output_gif_paths["all"][1], reduce_colors=False)
