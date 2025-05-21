import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from PIL import Image
import os
import sys
import yaml
import rasterio
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.models as models
import warnings
import matplotlib

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore", message=".*not compatible with tight_layout.*", category=UserWarning
)


global EPOCHS
EPOCHS = 50


def get_cfg(run_id):
    """Read YAML configuration files"""
    with open(os.path.join("reports", run_id, f"{run_id}_config.yaml")) as f:
        config = yaml.safe_load(f)

    with open(os.path.join("src", "000_internal_config.yaml")) as f:
        config_internal = yaml.safe_load(f)

    config_internal.update(config)
    return config_internal


class MyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, col="img_paths"):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.col = col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx][self.col])

        with rasterio.open(img_name) as src:
            # Read image data
            img = src.read()
            img = np.transpose(img, (1, 2, 0))

            # Handle different channel cases
            if img.shape[2] > 1:
                gray_img = np.mean(img, axis=2).astype(img.dtype)
            else:
                gray_img = img[:, :, 0]

            gray_img_uint8 = (gray_img * 255).astype("uint8")
            image = Image.fromarray(gray_img_uint8, mode="L")
            image = torch.from_numpy(np.array(image)).float().unsqueeze(0) / 255.0

        return image, self.df.iloc[idx]["extra_class_labels"]


def prepare_data(cfg, run_id):
    """Prepare and split the data"""
    sample_file = next(
        cfg["DATA_TYPE"][d]["FILE_RAW"]
        for d in cfg["DATA_TYPE"]
        if cfg["DATA_TYPE"][d]["TYPE"] == "IMG"
    )

    split_file = os.path.join("data/processed", run_id, "sample_split.parquet")
    mapping_file = os.path.join("data/raw", sample_file)

    sample_mappings = pd.read_csv(mapping_file, sep=cfg["DELIM"])

    split_df = pd.read_parquet(split_file)
    split_df.index.name = "index"

    sample_mappings = sample_mappings.merge(
        split_df[["SAMPLE_ID", "SPLIT"]], left_on="sample_ids", right_on="SAMPLE_ID"
    )
    sample_mappings["rec_paths"] = sample_mappings["sample_ids"] + ".tif"

    train_df = sample_mappings[sample_mappings["SPLIT"] == "train"]
    valid_df = sample_mappings[sample_mappings["SPLIT"] == "valid"]
    test_df = sample_mappings[sample_mappings["SPLIT"] == "test"]

    return train_df, valid_df, test_df


def create_dataloaders(train_df, valid_df, test_df, cfg, run_id):
    """Create DataLoaders for training, validation, and testing"""

    train_dataset = MyDataset(
        train_df,
        os.path.join("data/processed", run_id),
        col="img_paths",
    )

    valid_dataset = MyDataset(
        valid_df,
        os.path.join("data/processed", run_id),
        col="img_paths",
    )

    test_dataset_original = MyDataset(
        test_df,
        os.path.join("data/processed", run_id),
        col="img_paths",
    )

    test_dataset_reconstructed = MyDataset(
        test_df,
        os.path.join("reports", run_id, "Translate_FROM_TO_IMG"),
        col="rec_paths",
    )

    train_dataset_reconstructed = MyDataset(
        train_df,
        os.path.join("reports", run_id, "Translate_FROM_TO_IMG"),
        col="rec_paths",
    )

    valid_dataset_reconstructed = MyDataset(
        valid_df,
        os.path.join("reports", run_id, "Translate_FROM_TO_IMG"),
        col="rec_paths",
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    test_loader_original = DataLoader(
        test_dataset_original, batch_size=1, shuffle=False
    )
    test_loader_reconstructed = DataLoader(
        test_dataset_reconstructed, batch_size=1, shuffle=False
    )
    train_loader_reconstructed = DataLoader(
        train_dataset_reconstructed, batch_size=1, shuffle=False
    )
    valid_loader_reconstructed = DataLoader(
        valid_dataset_reconstructed, batch_size=1, shuffle=False
    )

    return (
        train_loader,
        valid_loader,
        test_loader_original,
        train_loader_reconstructed,
        valid_loader_reconstructed,
        test_loader_reconstructed,
    )


def train_model(
    model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=EPOCHS
):
    """Train the model and return training and validation loss history"""
    model.to(device)
    train_losses = []
    valid_losses = []

    train_dataset_size = len(train_loader.dataset)

    valid_dataset_size = len(valid_loader.dataset)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = (
                inputs.to(device),
                labels.to(device).float(),
            )  # Ensure labels are float

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1)  # Ensure outputs are of shape [batch_size]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / train_dataset_size
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = (
                    inputs.to(device),
                    labels.to(device).float(),
                )  # Ensure labels are float
                outputs = model(inputs)
                outputs = outputs.view(-1)  # Ensure outputs are of shape [batch_size]

                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)

        epoch_valid_loss = valid_loss / valid_dataset_size
        valid_losses.append(epoch_valid_loss)

    return train_losses, valid_losses


def evaluate(model, data_loader, device):
    """Evaluate the model and return metrics for regression"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    return mse, r2


def plot_losses(
    train_losses, val_losses, run_id, file_name="xmodalix_eval_regression_losses.png"
):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    # plt.tight_layout()
    plt.savefig(
        os.path.join("reports", "paper-visualizations", run_id.split("_")[0], file_name)
    )
    plt.close()
    # also save underlying data for plots
    pd.DataFrame({"train_losses": train_losses, "val_losses": val_losses}).to_csv(
        os.path.join(
            "reports", "paper-visualizations", run_id.split("_")[0], f"{file_name}.csv"
        ),
        index=False,
    )


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                1, 16, kernel_size=3, stride=1, padding=1
            ),  # Changed from 3 to 1 for grayscale
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),  # Single output for regression
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x


def get_model():
    model = models.vgg16(weights="DEFAULT")

    # Freeze all layers, except last one for transfer learning
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last fully connected layer
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(
        num_features, 1
    )  # Output a single value for regression

    return model


def main(run_id):
    cfg = get_cfg(run_id)
    train_df, valid_df, test_df = prepare_data(cfg, run_id)
    (
        train_loader,
        valid_loader,
        test_loader_original,
        train_loader_reconstructed,
        valid_loader_reconstructed,
        test_loader_reconstructed,
    ) = create_dataloaders(train_df, valid_df, test_df, cfg, run_id)
    model = get_model()
    # model = SimpleCNN()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_losses, val_losses = train_model(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        device,
        num_epochs=EPOCHS,
    )
    print("Final validation loss:", val_losses[-1])

    plot_losses(train_losses, val_losses, run_id)

    # Evaluate on train and validation sets (original and reconstructed)
    train_mse, train_r2 = evaluate(model, train_loader, device)
    valid_mse, valid_r2 = evaluate(model, valid_loader, device)

    train_mse_reconstructed, train_r2_reconstructed = evaluate(
        model, train_loader_reconstructed, device
    )
    valid_mse_reconstructed, valid_r2_reconstructed = evaluate(
        model, valid_loader_reconstructed, device
    )

    test_mse_original, test_r2_original = evaluate(model, test_loader_original, device)
    test_mse_reconstructed, test_r2_reconstructed = evaluate(
        model, test_loader_reconstructed, device
    )

    del model
    model_reconstructed = SimpleCNN()
    optimizer_reconstructed = optim.Adam(model_reconstructed.parameters(), lr=0.0005)

    # Train with reconstructed data loaders
    train_losses_reconstructed, val_losses_reconstructed = train_model(
        model_reconstructed,
        train_loader_reconstructed,
        valid_loader_reconstructed,
        criterion,
        optimizer_reconstructed,
        device,
        num_epochs=EPOCHS,
    )
    print("Final validation loss for reconstructed data:", val_losses_reconstructed[-1])
    plot_losses(
        train_losses_reconstructed,
        val_losses_reconstructed,
        run_id,
        file_name="xmodalix_recon_eval_regression_losses.png",
    )

    test_mse_reconstructed_trained, test_r2_reconstructed_trained = evaluate(
        model_reconstructed, test_loader_reconstructed, device
    )

    # Save evaluation metrics in a dataframe
    metrics_df = pd.DataFrame(
        {
            "Dataset": [
                "Train Original",
                "Train Reconstructed",
                "Validation Original",
                "Validation Reconstructed",
                "Test Original",
                "Test Reconstructed",
                "Test Reconstructed(trained_recons)",
            ],
            "MSE": [
                train_mse,
                train_mse_reconstructed,
                valid_mse,
                valid_mse_reconstructed,
                test_mse_original,
                test_mse_reconstructed,
                test_mse_reconstructed_trained,
            ],
            "R^2": [
                train_r2,
                train_r2_reconstructed,
                valid_r2,
                valid_r2_reconstructed,
                test_r2_original,
                test_r2_reconstructed,
                test_r2_reconstructed_trained,
            ],
        }
    )
    print(metrics_df)
    metrics_df.to_csv(
        os.path.join(
            "reports",
            "paper-visualizations",
            RUN_ID.split("_")[0],
            "xmodalix_eval_regression_metrics.csv",
        ),
        index=False,
    )


if __name__ == "__main__":
    global RUN_ID
    RUN_ID = sys.argv[1]
    main(RUN_ID)
