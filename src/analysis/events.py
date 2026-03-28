import polars as pl
import cuml
from sklearn.metrics import silhouette_score
import os
import matplotlib.pyplot as plt
import cupy as cp
import re
import numpy as np
from tqdm import tqdm

ROOT_DIR = "./data/processed/news_embeddings"

def get_X(file_path: str) -> cp.ndarray:
    df = pl.scan_parquet(file_path)
    X = cp.array(df.select("embeddings").collect()[:, 0].to_list(), dtype=cp.float32)
    valid_mask = cp.isfinite(X).all(axis=1)
    return X[valid_mask], valid_mask, df

def get_labels(X: cp.ndarray):
    hdbscan = cuml.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        metric='euclidean'
    )

    return hdbscan.fit_predict(X)

def save_plots(X: cp.ndarray, labels: cp.ndarray, file_path: str):
    umap = cuml.UMAP(n_components=2, metric='euclidean', random_state=42)
    tsne = cuml.TSNE(n_components=2, metric='euclidean', random_state=42)

    x_2d_umap = umap.fit_transform(X)
    x_2d_tsne = tsne.fit_transform(X)

    x_2d_umap_cpu = cp.asnumpy(x_2d_umap)
    x_2d_tsne_cpu = cp.asnumpy(x_2d_tsne)
    labels_cpu = cp.asnumpy(labels)

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    colors = plt.cm.Spectral(np.linspace(0, 1, len(np.unique(labels_cpu))))
    for i, cluster_id in enumerate(np.unique(labels_cpu)):
        mask = labels_cpu == cluster_id
        label_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise (-1)"
        ax[0].scatter(x_2d_umap_cpu[mask, 0], x_2d_umap_cpu[mask, 1], 
                    c=[colors[i]], label=label_name, s=30, alpha=0.7)
    ax[0].set_title('HDBSCAN Clusters (UMAP Projection)')
    ax[0].set_xlabel('UMAP Dimension 1')
    ax[0].set_ylabel('UMAP Dimension 2')
    ax[0].legend()

    for i, cluster_id in enumerate(np.unique(labels_cpu)):
        mask = labels_cpu == cluster_id
        label_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise (-1)"
        ax[1].scatter(x_2d_tsne_cpu[mask, 0], x_2d_tsne_cpu[mask, 1], 
                    c=[colors[i]], label=label_name, s=30, alpha=0.7)
    ax[1].set_title('HDBSCAN Clusters (t-SNE Projection)')
    ax[1].set_xlabel('t-SNE Dimension 1')
    ax[1].set_ylabel('t-SNE Dimension 2')
    ax[1].legend()

    plt.tight_layout()
    fig.savefig(file_path)
    plt.close()


if __name__ == "__main__":

    silhouette_total = []
    country_names = []
    unique_clusters_total = []
    noise_count_total = []

    for file in tqdm(os.listdir(ROOT_DIR), unit="files"):
        file_path = os.path.join(ROOT_DIR, file)
        X, valid_mask, df = get_X(file_path)
        labels = get_labels(X)

        country = re.search(r'news_(\w+)\.parquet', file).group(1)
        country_names.append(country)
        
        labels_cpu = cp.asnumpy(labels)
        silhouette_total.append(silhouette_score(cp.asnumpy(X), labels_cpu))

        # Count unique clusters (excluding -1)
        unique_clusters = len(np.unique(labels_cpu[labels_cpu != -1]))
        noise_count = (labels_cpu == -1).sum()
        unique_clusters_total.append(unique_clusters)
        noise_count_total.append(noise_count)

        valid_mask_np = cp.asnumpy(valid_mask)
        df_filtered = df.collect().filter(pl.Series(valid_mask_np)).with_columns(pl.Series("cluster", labels_cpu))
        df_filtered.write_parquet(file_path)

        plot_file = f"./data/plots/{country}_clusters.png"
        save_plots(X, labels, file_path=plot_file)

    # Silhouette Scores
    plt.figure(figsize=(14, 6))
    plt.bar(country_names, silhouette_total)
    plt.xlabel("Country")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores by Country for Event Clusters")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("./data/plots/silhouette_scores_by_country.png")
    plt.close()

    # Unique Clusters Count
    plt.figure(figsize=(14, 6))
    plt.bar(country_names, unique_clusters_total)
    plt.xlabel("Country")
    plt.ylabel("Number of Unique Clusters")
    plt.title("Number of Unique Clusters by Country")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("./data/plots/unique_clusters_by_country.png")
    plt.close()

    # Noise Count
    plt.figure(figsize=(14, 6))
    plt.bar(country_names, noise_count_total)
    plt.xlabel("Country")
    plt.ylabel("Number of Noise Points")
    plt.title("Number of Noise Points by Country")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("./data/plots/noise_count_by_country.png")
    plt.close()




