import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import pandas as pd
import numpy as np
from PIL import Image
import os
import sys
import yaml
import rasterio
import matplotlib.pyplot as plt
import warnings
import matplotlib

warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
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


class ExtraClassLabelsDataset(Dataset):
    def __init__(self, df, img_dir, mapping, transform=None, col="img_paths"):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.col = col
        self.mapping = mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx][self.col])
        label = self.mapping[self.df.iloc[idx]["num_class_label"]]

        with rasterio.open(img_name) as src:
            img = src.read()
            img = np.transpose(img, (1, 2, 0))
            img = (
                img[:, :, :3]
                if img.shape[2] > 3
                else np.repeat(img, 3, axis=2)
                if img.shape[2] == 1
                else img
            )
            img = (img * 255).astype("uint8")
            image = Image.fromarray(img, "RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


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
    sample_mappings["num_class_label"] = sample_mappings["extra_class_labels"]
    # if more n 10 classes, num_class_label will be used
    if len(sample_mappings["extra_class_labels"].unique()) > 10:
        sample_mappings["num_class_label"] = pd.qcut(
            sample_mappings["extra_class_labels"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
        )

    split_df = pd.read_parquet(split_file)
    split_df.index.name = "index"

    sample_mappings = sample_mappings.merge(
        split_df[["SAMPLE_ID", "SPLIT"]], left_on="sample_ids", right_on="SAMPLE_ID"
    )
    image_ext = sample_mappings["img_paths"][0].split(".")[-1]
    sample_mappings["rec_paths"] = sample_mappings["sample_ids"] + f".{image_ext}"

    train_df = sample_mappings[sample_mappings["SPLIT"] == "train"]
    valid_df = sample_mappings[sample_mappings["SPLIT"] == "valid"]
    test_df = sample_mappings[sample_mappings["SPLIT"] == "test"]

    return train_df, valid_df, test_df


def create_dataloaders(train_df, valid_df, test_df, cfg, run_id):
    """Create DataLoaders for training, validation, and testing"""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    mapping = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    cancer_mapping = {
        "Breast Cancer": 0,
        "Non-Small Cell Lung Cancer": 1,
        "Endometrial Cancer": 2,
        "Colorectal Cancer": 3,
        "Ovarian Epithelial Tumor": 4,
    }
    if "TCGA" in run_id:
        mapping = cancer_mapping

    train_dataset = ExtraClassLabelsDataset(
        train_df,
        os.path.join("data/processed", run_id),
        mapping=mapping,
        transform=transform,
        col="img_paths",
    )

    valid_dataset = ExtraClassLabelsDataset(
        valid_df,
        os.path.join("data/processed", run_id),
        mapping=mapping,
        transform=transform,
        col="img_paths",
    )

    test_dataset_original = ExtraClassLabelsDataset(
        test_df,
        os.path.join("data/processed", run_id),
        mapping=mapping,
        transform=transform,
        col="img_paths",
    )

    test_dataset_reconstructed = ExtraClassLabelsDataset(
        test_df,
        os.path.join("reports", run_id, "Translate_FROM_TO_IMG"),
        mapping=mapping,
        transform=transform,
        col="rec_paths",
    )

    train_dataset_reconstructed = ExtraClassLabelsDataset(
        train_df,
        os.path.join("reports", run_id, "Translate_FROM_TO_IMG"),
        mapping=mapping,
        transform=transform,
        col="rec_paths",
    )

    valid_dataset_reconstructed = ExtraClassLabelsDataset(
        valid_df,
        os.path.join("reports", run_id, "Translate_FROM_TO_IMG"),
        mapping=mapping,
        transform=transform,
        col="rec_paths",
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader_original = DataLoader(
        test_dataset_original, batch_size=32, shuffle=False
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
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
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
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)

        epoch_valid_loss = valid_loss / valid_dataset_size
        valid_losses.append(epoch_valid_loss)

    return train_losses, valid_losses


def evaluate(model, data_loader, device):
    """Evaluate the model and return accuracy and confusion matrix"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return accuracy, conf_matrix, f1


def plot_losses(
    train_losses, val_losses, run_id, outpath="xmodalix_eval_classifier_losses.png"
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
        os.path.join("reports", "paper-visualizations", run_id.split("_")[0], outpath)
    )
    plt.close()
    # save underlying data for plot:
    pd.DataFrame({"train_losses": train_losses, "val_losses": val_losses}).to_csv(
        os.path.join(
            "reports",
            "paper-visualizations",
            run_id.split("_")[0],
            f"{outpath}.csv",
        ),
        index=False,
    )


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model(num_classes):
    model = models.vgg16(weights="DEFAULT")

    # Freeze all layers, except last one for transfer learning
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last fully connected layer
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)

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

    # num classes dynamic based on unique values in num_class_label
    num_classes = 5 if "TCGA" in run_id else 4
    #### TO USE A SIMPLE CNN MODEL, UNCOMMENT THE FOLLOWING LINES ####
    # model = SimpleCNN(num_classes=num_classes)  # 4 classes for num_class_labels
    # model_recon = SimpleCNN(num_classes=num_classes)
    # In your main function, replace the SimpleCNN instantiation with:
    model = get_model(num_classes)
    model_recon = get_model(num_classes)

    # Modify the optimizer to only update the parameters of the last layer
    optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)
    optimizer_recon = optim.Adam(model_recon.classifier[6].parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"data device: {device}")

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
    train_accuracy, _, train_f1 = evaluate(model, train_loader, device)
    valid_accuracy, _, valid_f1 = evaluate(model, valid_loader, device)

    train_accuracy_reconstructed, _, train_f1_reconstructed = evaluate(
        model, train_loader_reconstructed, device
    )
    valid_accuracy_reconstructed, _, valid_f1_reconstructed = evaluate(
        model, valid_loader_reconstructed, device
    )

    accuracy_original, conf_matrix_original, f1_original = evaluate(
        model, test_loader_original, device
    )
    accuracy_reconstructed, conf_matrix_reconstructed, f1_reconstructed = evaluate(
        model, test_loader_reconstructed, device
    )

    # Get value counts for each dataset
    train_counts = train_df["num_class_label"].value_counts().sort_index()
    valid_counts = valid_df["num_class_label"].value_counts().sort_index()
    test_counts = test_df["num_class_label"].value_counts().sort_index()

    train_recon_lossed, val_recon_losses = train_model(
        model_recon,
        train_loader_reconstructed,
        valid_loader_reconstructed,
        criterion,
        optimizer_recon,
        device,
        num_epochs=EPOCHS,
    )
    plot_losses(
        train_recon_lossed,
        val_recon_losses,
        run_id,
        "xmodalix_eval_classifier_recon_losses.png",
    )
    (
        accuracy_t_reconstructed,
        conf_matrix_t_reconstructed,
        f1_t_reconstructed,
    ) = evaluate(model_recon, test_loader_reconstructed, device)
    # Create a DataFrame for value counts
    counts_df = (
        pd.DataFrame(
            {"Train": train_counts, "Validation": valid_counts, "Test": test_counts}
        )
        .fillna(0)
        .astype(int)
    )

    # Determine group variables based on the run_id
    group_vars = (
        ["Q1", "Q2", "Q3", "Q4"]
        if "TCGA" not in run_id
        else [
            "Non-Small Cell Lung Cancer",
            "Breast Cancer",
            "Endometrial Cancer",
            "Colorectal Cancer",
            "Ovarian Epithelial Tumor",
        ]
    )

    # Create base metrics dictionary
    metrics_dict = {
        "Dataset": [
            "Train Original",
            "Train Reconstructed",
            "Validation Original",
            "Validation Reconstructed",
            "Test Original",
            "Test Reconstructed",
            "Test Reconstructed Trained on Reconstructed",
        ],
        "Accuracy": [
            train_accuracy,
            train_accuracy_reconstructed,
            valid_accuracy,
            valid_accuracy_reconstructed,
            accuracy_original,
            accuracy_reconstructed,
            accuracy_t_reconstructed,
        ],
        "F1 Score": [
            train_f1,
            train_f1_reconstructed,
            valid_f1,
            valid_f1_reconstructed,
            f1_original,
            f1_reconstructed,
            f1_t_reconstructed,
        ],
    }

    # Dynamically add count columns for each group
    for group in group_vars:
        metrics_dict[f"{group} Count"] = (
            [counts_df.loc[group, "Train"]] * 2
            + [counts_df.loc[group, "Validation"]] * 2
            + [counts_df.loc[group, "Test"]]
            * 3  # Note: 3 times for test to match the number of rows
        )

    # Create the DataFrame
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(
        os.path.join(
            "reports",
            "paper-visualizations",
            RUN_ID.split("_")[0],
            "xmodalix_eval_classifier_metrics.csv",
        ),
        index=False,
    )

    print("\nPerformance Comparison:")
    print(metrics_df)

    # print("\nDetailed Performance:")
    # print(f"Original Train Set Accuracy: {train_accuracy:.4f}")
    # print(f"Original Train Set F1 score: {train_f1:.4f}")
    # print(f"Original confusion matrix:\n{conf_matrix_original}")
    # print("-------------------------------------\n")

    # print(f"Reconstructed Train Set Accuracy: {train_accuracy_reconstructed:.4f}")
    # print(f"Reconstructed Train Set F1 score: {train_f1_reconstructed:.4f}")
    # print(f"Reconstructed confusion matrix:\n{conf_matrix_reconstructed}")
    # print("-------------------------------------\n")

    # print(f"Original Validation Set Accuracy: {valid_accuracy:.4f}")
    # print(f"Original Validation Set F1 score: {valid_f1:.4f}")
    # print(f"Reconstructed confusion matrix:\n{conf_matrix_reconstructed}")
    # print("-------------------------------------\n")

    # print(f"Reconstructed Validation Set Accuracy: {valid_accuracy_reconstructed:.4f}")
    # print(f"Reconstructed Validation Set F1 score: {valid_f1_reconstructed:.4f}")
    # print(f"Reconstructed confusion matrix:\n{conf_matrix_reconstructed}")
    # print("-------------------------------------\n")

    # print(f"Original Test Set Accuracy: {accuracy_original:.4f}")
    # print(f"Original Test Set F1 score: {f1_original:.4f}")
    # print(f"Original confusion matrix:\n{conf_matrix_original}")
    # print("-------------------------------------\n")

    # print(f"Reconstructed Test Set Accuracy: {accuracy_reconstructed:.4f}")
    # print(f"Reconstructed Test Set F1 score: {f1_reconstructed:.4f}")
    # print(f"Reconstructed confusion matrix:\n{conf_matrix_reconstructed}")
    # print("-------------------------------------\n")

    # print(
    #     f"Reconstructed Test Set Trained on Reconstructed Accuracy: {accuracy_t_reconstructed:.4f}"
    # )
    # print(
    #     f"Reconstructed Test Set Trained on Reconstructed F1 score: {f1_t_reconstructed:.4f}"
    # )
    # print(f"Reconstructed confusion matrix:\n{conf_matrix_t_reconstructed}")
    # print("-------------------------------------\n")
    # print(f"Test set F1 score difference: {f1_original - f1_reconstructed:.4f}")
    # print(
    #     f"Test Set trained on reconstructed F1 score difference: {f1_original - f1_t_reconstructed:.4f}"
    # )

    # print("\nClass Distribution:")
    # print(counts_df)

    torch.save(
        model.state_dict(),
        os.path.join(
            "reports",
            "paper-visualizations",
            RUN_ID.split("_")[0],
            "cnn_classifier.pth",
        ),
    )


if __name__ == "__main__":
    global RUN_ID
    RUN_ID = sys.argv[1]
    main(RUN_ID)
