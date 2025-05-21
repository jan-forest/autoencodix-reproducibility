import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
import yaml
from sklearn.decomposition import PCA


def get_pca(data, labels):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    embedding_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"], index=data.index)
    embedding_df["label"] = labels
    return embedding_df


def get_umap(data, labels, random_state=42):
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    umap_result = reducer.fit_transform(data)
    embedding_df = pd.DataFrame(
        umap_result, columns=["UMAP1", "UMAP2"], index=data.index
    )
    embedding_df["label"] = labels
    return embedding_df


def plot_embedding_2D(
    data,
    labels,
    title="PCA Plot",
    figsize=(12, 8),
    save_fig="",
    center=True,
    no_leg=False,
):
    """
    Creates a 2D embedding visualization of the input data.
    ARGS:
        data (pd.DataFrame): DataFrame containing the embedding coordinates.
        labels (pd.Series): Labels for coloring the points (e.g., cancer types).
        title (str): Title for the plot.
        figsize (tuple): Figure size specification.
        save_fig (str): File path for saving the plot.
        center (bool): If True, centers of groups are visualized as stars.
        no_leg (bool): If True, disables the legend.
    RETURNS:
        fig (matplotlib.figure): Figure handle.
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = np.unique(labels)
    palette = sns.color_palette("husl", len(unique_labels))

    # Determine the column names for x and y
    # Check if it's PCA or UMAP data based on column names
    if "PC1" in data.columns:
        x_col = "PC1"
        y_col = "PC2"
        x_label = "Principal Component 1"
        y_label = "Principal Component 2"
    elif "UMAP1" in data.columns:
        x_col = "UMAP1"
        y_col = "UMAP2"
        x_label = "UMAP Dimension 1"
        y_label = "UMAP Dimension 2"
    else:
        # Default to the first two columns if neither PCA nor UMAP
        x_col = data.columns[0]
        y_col = data.columns[1]
        x_label = x_col
        y_label = y_col

    sns.scatterplot(
        x=x_col,
        y=y_col,
        hue="label",
        data=data,
        palette=palette,
        s=40,
        alpha=0.8,
        edgecolor="black",
        ax=ax,
    )

    if center:
        means = data.groupby("label")[[x_col, y_col]].mean()
        sns.scatterplot(
            x=means[x_col],
            y=means[y_col],
            hue=means.index,
            palette=palette,
            s=200,
            edgecolor="black",
            alpha=0.9,
            marker="*",
            legend=False,
            ax=ax,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if no_leg:
        ax.legend([], [], frameon=False)
    else:
        ax.legend(title="Cancer Type", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    if save_fig:
        fig.savefig(save_fig, bbox_inches="tight", dpi=300)

    return fig


def compare_translation_to_input():
    rna_input_df = pd.read_parquet(f"data/processed/{run_id}/RNA_data.parquet")
    clinical_df = pd.read_parquet(
        os.path.join("data/raw", cfg["DATA_TYPE"]["ANNO"]["FILE_RAW"])
    )

    # Match RNA input data with clinical data
    rna_input_df = rna_input_df.loc[clinical_df.index.intersection(rna_input_df.index)]
    for param in cfg["CLINIC_PARAM"]:
        if not param == "CANCER_TYPE":
            print("for now we only plot CANCER_TYPE")
            continue

        clinic_params = clinical_df.loc[rna_input_df.index, param]

        # Get PCA and UMAP embeddings
        rna_input_pca_df = get_pca(rna_input_df, clinic_params)
        rna_input_umap_df = get_umap(rna_input_df, clinic_params)

        # Plot PCA for RNA input
        plot_embedding_2D(
            data=rna_input_pca_df,
            labels=clinic_params,
            title="",
            save_fig=os.path.join(OUTPUT_DIR, f"{param}_rna_input_pca.png"),
        )

        # Plot UMAP for RNA input - Note: Save with a different filename
        plot_embedding_2D(
            data=rna_input_umap_df,
            labels=clinic_params,
            title="",
            save_fig=os.path.join(OUTPUT_DIR, f"{param}_rna_input_umap.png"),
        )

    #### dim reducion on output
    #### ------------------------
    # Load translated output from METH_TO_RNA
    meth_to_rna_df = pd.read_csv(
        f"reports/{run_id}/translated.txt", sep="\t", index_col=0
    )

    # Match translated output with clinical data
    meth_to_rna_df = meth_to_rna_df.loc[
        clinical_df.index.intersection(meth_to_rna_df.index)
    ]
    del meth_to_rna_df["shape"]
    for param in cfg["CLINIC_PARAM"]:
        if not param == "CANCER_TYPE":
            print("for now we only plot CANCER_TYPE")
            continue
        clin_param_translated = clinical_df.loc[meth_to_rna_df.index, param]
        meth_to_rna_pca_df = get_pca(meth_to_rna_df, clin_param_translated)

        # Plot PCA for translated output
        plot_embedding_2D(
            data=meth_to_rna_pca_df,
            labels=clin_param_translated,
            title="",
            save_fig=os.path.join(
                OUTPUT_DIR, f"{param}_meth_to_rna_translated_pca.png"
            ),
        )

        meth_to_rna_umap_df = get_umap(meth_to_rna_df, clin_param_translated)
        # Plot UMAP for translated output
        plot_embedding_2D(
            data=meth_to_rna_umap_df,
            labels=clin_param_translated,
            title="",
            save_fig=os.path.join(
                OUTPUT_DIR, f"{param}_meth_to_rna_translated_umap.png"
            ),
        )

def calculate_perf_over_random(emb_df, rand_df, name):
    rows = []
    for alg in emb_df['ML_ALG'].unique():
        for param in emb_df['CLINIC_PARAM'].unique():
            for met in emb_df['metric'].unique():
                filtered_embedding_df = emb_df[(emb_df['ML_ALG']==alg)&
                           (emb_df['CLINIC_PARAM']==param)&
                           (emb_df['metric']==met)]
                filtered_random_df = rand_df[(rand_df['ML_ALG']==alg)&
                            (rand_df['CLINIC_PARAM']==param)&
                            (rand_df['metric']==met)]
                if filtered_embedding_df.empty or filtered_random_df.empty:
                    continue
                mu, stdev = filtered_random_df['value'].mean(), filtered_random_df['value'].std()
                # stdev = stdev if stdev and not np.isnan(stdev) else 0.01
                for _, row in filtered_embedding_df.iterrows():
                    z = (row['value'] - mu) / stdev
                    rows.append({
                        'ML_ALG': alg,
                        'CLINIC_PARAM': param,
                        'Metric': met,
                        'Embedding': name,
                        'Perf. over random': z
                    })
    return pd.DataFrame(rows)


def compare_ml_task():

    varix_control_run_id = "Exp6_TCGA_VARIX"
    xml_task_file = os.path.join("reports", run_id, "ml_task_performance.txt")
    vml_task_file = os.path.join("reports", varix_control_run_id, "ml_task_performance.txt")

    xml_df = pd.read_csv(xml_task_file, sep="\t")
    vml_df = pd.read_csv(vml_task_file, sep="\t")

    xml_test = xml_df[xml_df['score_split']=='test']
    vml_test = vml_df[vml_df['score_split']=='test']

    # pick out tasks
    latent_from_df = xml_test[xml_test['ML_TASK']=='Latent_FROM']
    latent_to_df   = xml_test[xml_test['ML_TASK']=='Latent_TO']
    xml_rand_df    = xml_test[xml_test['ML_TASK']=='RandomFeature']

    varix_latent_df = vml_test[vml_test['ML_TASK']=='Latent']
    vml_rand_df     = vml_test[vml_test['ML_TASK']=='RandomFeature']

    # compute perf-over-random
    from_perf  = calculate_perf_over_random(latent_from_df, xml_rand_df, 'Cross-Modal (FROM)')
    to_perf    = calculate_perf_over_random(latent_to_df,   xml_rand_df, 'Cross-Modal (TO)')
    varix_perf = calculate_perf_over_random(varix_latent_df, vml_rand_df,   'Variational')

    all_perf = pd.concat([from_perf, to_perf, varix_perf])

    # plot settings
    sns.set_style("whitegrid")
    for alg in ['Linear', 'RF']:
        df = all_perf[all_perf['ML_ALG']==alg]
        plt.figure(figsize=(10,6))
        sns.boxplot(x='Embedding', y='Perf. over random',
                    hue='Embedding', data=df, palette='pastel', legend=False)
        plt.title(f'ML Performance Over Random - {alg}')
        # plt.yscale('symlog')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'ml_performance_{alg}.svg'), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, f'ml_performance_{alg}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created plot for {alg}")


if __name__ == "__main__":
    OUTPUT_DIR = os.path.join("reports", "paper-visualizations", "Exp6")
    parser = argparse.ArgumentParser(description="Visualize cross-modal data")
    parser.add_argument(
        "--run_id",
        type=str,
        default="Exp6_TCGA_METH_RNA",
        help="Run ID for the experiment",
    )
    parser.add_argument(
        "--varix_control",
        type=str,
        default="Exp6_TCGA_VARIX",
        help="Control Varix run ID",
    )
    parser.add_argument(
        "--modality_control",
        type=str,
        default="Exp6_TCGA_RNA_RNA",
        help="Control Modality for same modality translation",
    )

    args = parser.parse_args()
    run_id = args.run_id
    varix_control_run_id = args.varix_control
    modality_control_run_id = args.modality_control
    cfg_path = f"{run_id}_config.yaml"
    varix_config_path = f"{varix_control_run_id}_config.yaml"
    modadlity_config_path = f"{modality_control_run_id}_config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    with open(varix_config_path, "r") as f:
        varix_cfg = yaml.safe_load(f)
    with open(modadlity_config_path, "r") as f:
        modality_cfg = yaml.safe_load(f)

    compare_translation_to_input()
    compare_ml_task()