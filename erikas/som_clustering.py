import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.decomposition import PCA
from minisom import MiniSom

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.data_loader import load_and_preprocess_data
from shared.visualize import print_results_table, plot_task3_results, plot_task4_results_3d


class SOMClusterer(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=2, epochs=500, random_state=42):
        self.n_clusters = n_clusters
        self.epochs = epochs
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        X_arr = np.array(X)
        som = MiniSom(x=1, y=self.n_clusters, input_len=X_arr.shape[1], 
                      sigma=1, learning_rate=0.5, random_seed=self.random_state)
        som.random_weights_init(X_arr)
        som.train_random(X_arr, self.epochs, verbose=False)

        self.cluster_centers_ = som.get_weights()[0]

        labels = []
        for sample in X_arr:
            bmu = som.winner(sample)
            labels.append(bmu[1])

        self.labels_ = np.array(labels)
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_


def run_som_clustering(data, k, epochs=500, random_state=42):
    data_arr = np.array(data)
    som = MiniSom(x=1, y=k, input_len=data_arr.shape[1], 
                  sigma=1, learning_rate=0.5, random_seed=random_state)
    som.random_weights_init(data_arr)
    som.train_random(data_arr, epochs, verbose=False)

    centroids = som.get_weights()[0]
    labels = []
    inertia = 0.0

    for sample in data_arr:
        bmu = som.winner(sample)
        cluster_id = bmu[1]
        labels.append(cluster_id)
        
        centroid = centroids[cluster_id]
        inertia += np.sum((sample - centroid) ** 2)

    labels = np.array(labels)

    if len(np.unique(labels)) > 1:
        sil_coef = silhouette_score(data_arr, labels)
    else:
        sil_coef = -1.0

    return labels, inertia, sil_coef

def analyze_attribute_pairs(scaled_df):
    import itertools
    cols = scaled_df.columns.tolist()
    pairs = list(itertools.combinations(cols, 2))
    results = []
    print(f"\n[SOM] Skenuojama {len(pairs)} atributų porų...")
    for pair in pairs:
        pair_data = scaled_df[list(pair)].values
        pair_results = {'pair': f"{pair[0]} - {pair[1]}", 'metrics': {}}
        global_max_sil = -1.0
        best_k = 2
        for k in range(2, 9):
            labels, inertia, sil = run_som_clustering(pair_data, k)
            pair_results['metrics'][k] = {'inertia': inertia, 'silhouette': sil}
            if sil > global_max_sil:
                global_max_sil = sil
                best_k = k
        pair_results['global_max_sil'] = global_max_sil
        pair_results['best_k'] = best_k
        results.append(pair_results)
    results.sort(key=lambda x: x['global_max_sil'], reverse=True)
    return results


def analyze_attribute_triplets(scaled_df):
    import itertools
    cols = scaled_df.columns.tolist()
    triplets = list(itertools.combinations(cols, 3))
    results = []
    print(f"\n[SOM] Skenuojama {len(triplets)} atributų trejetų...")
    for triplet in triplets:
        triplet_data = scaled_df[list(triplet)].values
        triplet_results = {'pair': f"{triplet[0]} - {triplet[1]} - {triplet[2]}", 'metrics': {}}
        global_max_sil = -1.0
        best_k = 2
        for k in range(2, 9):
            labels, inertia, sil = run_som_clustering(triplet_data, k)
            triplet_results['metrics'][k] = {'inertia': inertia, 'silhouette': sil}
            if sil > global_max_sil:
                global_max_sil = sil
                best_k = k
        triplet_results['global_max_sil'] = global_max_sil
        triplet_results['best_k'] = best_k
        results.append(triplet_results)
    results.sort(key=lambda x: x['global_max_sil'], reverse=True)
    return results


def analyze_m_dimensional(scaled_df):
    data_arr = scaled_df.values
    metrics = {}

    print("\n[SOM] Vykdoma m-dimensijų (m=10) klasterizacija visoms k reikšmėms...")

    for k in range(2, 9):
        labels, inertia, sil = run_som_clustering(data_arr, k)
        metrics[k] = {'inertia': inertia, 'silhouette': sil, 'labels': labels}

    return metrics


def print_m_dim_table(metrics):
    print("\n### m-Dimensijų (m=10) SOM klasterizacijos rezultatai\n")
    header = "| Rezultatų įverčiai | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 (max) |"
    divider = "|---|---|---|---|---|---|---|---|"
    
    inertia_row = "| **Inercija** |"
    sil_row = "| **Silueto koef.** |"
    
    for k in range(2, 9):
        inertia_row += f" {metrics[k]['inertia']:.1f} |"
        sil_row += f" {metrics[k]['silhouette']:.3f} |"
        
    print(header)
    print(divider)
    print(inertia_row)
    print(sil_row)


def plot_m_dim_results(metrics, scaled_df, output_filename):
    k_scores = [(k, metrics[k]['silhouette']) for k in range(2, 9)]
    k_scores.sort(key=lambda x: x[1], reverse=True)
    top_3_ks = [item[0] for item in k_scores[:3]]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    ax_elbow = axes[0, 0]
    ks = list(range(2, 9))
    inertias = [metrics[k]['inertia'] for k in ks]
    ax_elbow.plot(ks, inertias, 'go-', markersize=8, linewidth=2)
    ax_elbow.set_title("m-Dimensijų SOM Inercijos (Elbow) grafikas", fontsize=11)
    ax_elbow.set_xlabel("Klasterių skaičius (k)")
    ax_elbow.set_ylabel("Inercija")
    ax_elbow.grid(True, linestyle='--', alpha=0.5)

    best_k = top_3_ks[0]
    ax_elbow.axvline(x=best_k, color='r', linestyle='--', label=f'Geriausias siluetas k={best_k}')
    ax_elbow.legend()

    data_arr = scaled_df.values
    plot_positions = [(0, 1), (1, 0), (1, 1)]

    for idx, k in enumerate(top_3_ks):
        pos = plot_positions[idx]
        ax_sil = axes[pos[0], pos[1]]

        labels = metrics[k]['labels']
        sample_sil_vals = silhouette_samples(data_arr, labels)
        y_lower = 10

        for cl in range(k):
            cl_sil_vals = sample_sil_vals[labels == cl]
            cl_sil_vals.sort()
            size_cluster_cl = cl_sil_vals.shape[0]
            y_upper = y_lower + size_cluster_cl

            color = cm.nipy_spectral(float(cl) / k)
            ax_sil.fill_betweenx(np.arange(y_lower, y_upper), 0, cl_sil_vals,
                                  facecolor=color, edgecolor=color, alpha=0.7)
            ax_sil.text(-0.05, y_lower + 0.5 * size_cluster_cl, str(cl))
            y_lower = y_upper + 10

        avg_sil = metrics[k]['silhouette']
        ax_sil.axvline(x=avg_sil, color="red", linestyle="--", label=f"Vidurkis: {avg_sil:.3f}")
        ax_sil.set_title(f"m-Dimensijų Silueto diagrama (k={k})", fontsize=11)
        ax_sil.set_xlabel("Silueto koeficientas")
        ax_sil.set_ylabel("Klasterio ID")
        ax_sil.set_yticks([])
        ax_sil.set_xlim([-0.1, 1.0])
        ax_sil.legend()

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"[SOM] m-dimensijų grafikai sėkmingai išsaugoti: {output_filename}")
    plt.close()


def generate_clustergram_plot(scaled_df, output_filename):
    print("[SOM] Generuojama Clustergram diagrama naudojant PCA projekciją...")
    
    data_arr = scaled_df.values
    
    pca = PCA(n_components=1, random_state=42)
    data_pc1 = pca.fit_transform(data_arr).flatten()
    
    ks = list(range(2, 9))
    labels_by_k = {}
    centers_by_k = {}
    
    for k in ks:
        labels, _, _ = run_som_clustering(data_arr, k)
        labels_by_k[k] = labels
        centers = []
        for c in range(k):
            mask = (labels == c)
            if np.any(mask):
                centers.append(np.mean(data_pc1[mask]))
            else:
                centers.append(0.0)
        centers_by_k[k] = np.array(centers)
        
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(len(ks) - 1):
        k_curr = ks[i]
        k_next = ks[i+1]
        
        labels_curr = labels_by_k[k_curr]
        labels_next = labels_by_k[k_next]
        
        centers_curr = centers_by_k[k_curr]
        centers_next = centers_by_k[k_next]
        
        for c_curr in range(k_curr):
            for c_next in range(k_next):
                transition_mask = (labels_curr == c_curr) & (labels_next == c_next)
                count = np.sum(transition_mask)
                if count > 0:
                    lw = 0.5 + 8.0 * (count / len(data_arr))
                    ax.plot([k_curr, k_next], [centers_curr[c_curr], centers_next[c_next]], 
                            color='black', alpha=0.5, linewidth=lw)
                            
    for k in ks:
        labels = labels_by_k[k]
        centers = centers_by_k[k]
        for c in range(k):
            c_size = np.sum(labels == c)
            marker_size = 20 + 300 * (c_size / len(data_arr))
            ax.scatter(k, centers[c], color='#1f77b4', s=marker_size, zorder=3, alpha=0.9)
            
    ax.set_title("SOM m-dimensijų Clustergram (Schonlau principu, naudojant PCA)", fontsize=12)
    ax.set_xlabel("Klasterių skaičius (k)")
    ax.set_ylabel("Klasterių centrai (PC1 projekcija)")
    ax.set_xticks(ks)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"[SOM] Clustergram sėkmingai išsaugotas: {output_filename}")
    plt.close()


if __name__ == "__main__":
    _, scaled_df = load_and_preprocess_data("data.csv")

    print("\n3 Dalis (Atributų poros)")
    pairs_results = analyze_attribute_pairs(scaled_df)
    print_results_table(pairs_results, top_n=15)
    output_img_2d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "som_task3_results.png")
    plot_task3_results(pairs_results, scaled_df, run_som_clustering, output_img_2d, "SOM")

    print("\n4 Dalis (Atributų trejetai)")
    triplets_results = analyze_attribute_triplets(scaled_df)
    print_results_table(triplets_results, top_n=15)
    output_img_3d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "som_task4_results.png")
    plot_task4_results_3d(triplets_results, scaled_df, run_som_clustering, output_img_3d, "SOM")

    print("\n5 Dalis (m-dimensijų analizė)")
    m_dim_metrics = analyze_m_dimensional(scaled_df)

    print_m_dim_table(m_dim_metrics)

    output_img_m_dim = os.path.join(os.path.dirname(os.path.abspath(__file__)), "som_task5_results.png")
    plot_m_dim_results(m_dim_metrics, scaled_df, output_img_m_dim)

    output_img_cgram = os.path.join(os.path.dirname(os.path.abspath(__file__)), "som_task5_clustergram.png")
    generate_clustergram_plot(scaled_df, output_img_cgram)