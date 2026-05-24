import os
import sys

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from mpl_toolkits.mplot3d import Axes3D  # for 3D scatter plots
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.visualize import print_results_table
from shared.data_loader import load_and_preprocess_data

print("Loading and Preparing Data...")

cleaned_df, scaled_df = load_and_preprocess_data()

print(f"Columns used: {list(cleaned_df.columns)}")

def run_kmeans_analysis(X, k_range=range(2, 9)):
    results = []
    inertias = []
    silhouettes = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        
        inertia = km.inertia_
        sil_score = silhouette_score(X, labels)
        
        inertias.append(inertia)
        silhouettes.append(sil_score)
        results.append({'k': k, 'inertia': inertia, 'silhouette': sil_score})
        
    return pd.DataFrame(results), inertias, silhouettes

pairs_results = []
columns = scaled_df.columns.tolist()

for p in itertools.combinations(columns, 2):
    X_pair = scaled_df[list(p)]
    df_metrics, inertias, silhouettes = run_kmeans_analysis(X_pair)
    
    max_sil = df_metrics['silhouette'].max()
    best_k_for_pair = df_metrics.loc[df_metrics['silhouette'].idxmax(), 'k']
    
    metrics = {}
    for _, r in df_metrics.iterrows():
        metrics[int(r['k'])] = {'inertia': r['inertia'], 'silhouette': r['silhouette']}
    pairs_results.append({
        'Pair': f"{p[0]}, {p[1]}",
        'Max Silhouette': max_sil,
        'Best k': best_k_for_pair,
        'Inertias': inertias,
        'Silhouettes': silhouettes,
        'DataFrame': df_metrics,
        'Data': X_pair,
        'metrics': metrics
    })

pairs_results.sort(key=lambda x: x['Max Silhouette'], reverse=True)
top_3_pairs = pairs_results[:3]

# Add 'pair' key alias for print_results_table compatibility
for r in pairs_results:
    r['pair'] = r['Pair']

table_data = []
for res in pairs_results:
    row = {
        'Atributų pora': res['Pair'],
        'Max Sil (Global)': res['Max Silhouette']
    }
    df_m = res['DataFrame']
    for _, r in df_m.iterrows():
        row[f"k={r['k']} Inertia"] = r['inertia']
        row[f"k={r['silhouette']:.2f} Sil"] = r['silhouette']
        
    table_data.append(row)

plt.close('all')
fig, axes = plt.subplots(3, 3, figsize=(20, 15))

for i, res in enumerate(top_3_pairs):
    pair_name = res['Pair']
    X_pair = res['Data']
    best_k = int(res['Best k'])
    
    km_best = KMeans(n_clusters=best_k, random_state=42).fit(X_pair)
    axes[i, 0].scatter(X_pair.iloc[:, 0], X_pair.iloc[:, 1], c=km_best.labels_, cmap='viridis')
    axes[i, 0].set_title(f"Data: {pair_name} (k={best_k})")

    axes[i, 1].plot(range(2, 9), res['Inertias'], marker='o', color='orange')
    axes[i, 1].set_title(f"Inertia: {pair_name}")
    axes[i, 1].set_xlabel("k")

    labels = km_best.labels_
    sil_samples = silhouette_samples(X_pair, labels)
    unique_labels = np.unique(labels)
    viridis = plt.cm.get_cmap('viridis', best_k)
    y_lower = 10

    for label in unique_labels:
        cluster_sil_values = sil_samples[labels == label]
        cluster_sil_values.sort()
        size = cluster_sil_values.shape[0]
        y_upper = y_lower + size
        axes[i, 2].hlines(y=y_lower, xmin=0, xmax=cluster_sil_values.max(), color=viridis(label), linewidth=2)
        axes[i, 2].fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil_values, facecolor=viridis(label), alpha=0.7)
        axes[i, 2].text(0.05, (y_lower + y_upper)/2, str(label), va='center')
        y_lower = y_upper + 10
        
    axes[i, 2].axvline(x=silhouette_score(X_pair, labels), color="red", linestyle="--")
    axes[i, 2].set_title(f"Silhouette: {pair_name}")
    axes[i, 2].set_xlabel("Silhouette Coeff.")

plt.tight_layout()
plt.savefig("audrius/task1_pairs.png", dpi=150, bbox_inches='tight')
print("Saved: audrius/task1_pairs.png")

triplets_results = []
for t in itertools.combinations(columns, 3):
    X_trip = scaled_df[list(t)]
    df_metrics, inertias, silhouettes = run_kmeans_analysis(X_trip)
    max_sil = df_metrics['silhouette'].max()
    
    triplets_results.append({
        'Triplet': f"{t[0]}, {t[1]}, {t[2]}",
        'Max Silhouette': max_sil,
        'Best k': df_metrics.loc[df_metrics['silhouette'].idxmax(), 'k'],
        'Inertias': inertias,
        'Silhouettes': silhouettes,
        'DataFrame': df_metrics,
        'Data': X_trip
    })

triplets_results.sort(key=lambda x: x['Max Silhouette'], reverse=True)
top_3_triplets = triplets_results[:3]

print("\n--- Top Triplets Summary ---")
for i, res in enumerate(top_3_triplets):
    print(f"Rank {i+1}: {res['Triplet']} | Best k={res['Best k']} | Max Silhouette={res['Max Silhouette']:.4f}")

# Collect all (pair, k) silhouette scores
all_scores = []
for res in pairs_results:
    for _, r in res['DataFrame'].iterrows():
        all_scores.append({
            'Combination': res['Pair'],
            'Dimension': 2,
            'k': r['k'],
            'Silhouette': r['silhouette'],
            'Inertia': r['inertia']
        })

all_scores.sort(key=lambda x: x['Silhouette'], reverse=True)

# Keep only unique combinations (best silhouette per combination)
seen = set()
unique_scores = []
for s in all_scores:
    if s['Combination'] not in seen:
        seen.add(s['Combination'])
        unique_scores.append(s)

print("\n--- Top 10 Attribute Combinations by Silhouette Coefficient ---")
for i, s in enumerate(unique_scores[:10]):
    print(f"Rank {i+1}: [{s['Dimension']}D] {s['Combination']} | k={s['k']} | Silhouette={s['Silhouette']:.4f} | Inertia={s['Inertia']:.2f}")

# Find the top 10 unique pairs in pairs_results order and print detailed table
top_10_pairs_for_table = []
seen_pairs = set()
for res in pairs_results:
    if res['Pair'] not in seen_pairs:
        seen_pairs.add(res['Pair'])
        top_10_pairs_for_table.append(res)
    if len(top_10_pairs_for_table) == 10:
        break

print("\n--- Detailed Results Table (Top 10 Pairs) ---")
print_results_table(top_10_pairs_for_table, top_n=10)

plt.close('all')
fig = plt.figure(figsize=(20, 15))

for i, res in enumerate(top_3_triplets):
    triplet_name = res['Triplet']
    X_trip = res['Data']
    best_k = int(res['Best k'])
    
    ax = fig.add_subplot(2, 3, i+1, projection='3d')
    km_best = KMeans(n_clusters=best_k, random_state=42).fit(X_trip)
    scatter = ax.scatter(X_trip.iloc[:, 0], X_trip.iloc[:, 1], X_trip.iloc[:, 2], c=km_best.labels_, cmap='viridis')
    ax.set_title(f"3D Data: {triplet_name} (k={best_k})")
    triplet_cols = [c.strip() for c in triplet_name.split(',')]
    ax.set_xlabel(triplet_cols[0])
    ax.set_ylabel(triplet_cols[1])
    ax.set_zlabel(triplet_cols[2])
    
    ax_in = fig.add_subplot(2, 3, i+4)
    ax_in.plot(range(2, 9), res['Inertias'], marker='o', color='red')
    ax_in.set_title(f"Inertia: {triplet_name}")
    ax_in.set_xlabel("k")

plt.tight_layout()
plt.savefig("audrius/task2_triplets_3d.png", dpi=150, bbox_inches='tight')

plt.close('all')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, res in enumerate(top_3_triplets):
    triplet_name = res['Triplet']
    X_trip = res['Data']
    best_k = int(res['Best k'])
    
    km_best = KMeans(n_clusters=best_k, random_state=42).fit(X_trip)
    labels = km_best.labels_
    sil_samples = silhouette_samples(X_trip, labels)
    unique_labels = np.unique(labels)
    y_lower = 10
    
    for label in unique_labels:
        cluster_sil_values = sil_samples[labels == label]
        cluster_sil_values.sort()
        size = cluster_sil_values.shape[0]
        y_upper = y_lower + size
        axes[i].hlines(y=y_lower, xmin=0, xmax=cluster_sil_values.max(), color='blue', linewidth=2)
        axes[i].fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil_values, facecolor='blue', alpha=0.7)
        axes[i].text(0.05, (y_lower + y_upper)/2, str(label), va='center')
        y_lower = y_upper + 10
        
    axes[i].axvline(x=silhouette_score(X_trip, labels), color="red", linestyle="--")
    axes[i].set_title(f"Silhouette: {triplet_name}")
    axes[i].set_xlabel("Silhouette Coeff.")

plt.tight_layout()
plt.savefig("audrius/task2_triplets_silhouette.png", dpi=150, bbox_inches='tight')
print("Saved: audrius/task2_triplets_silhouette.png")


print("\nStep 4: Full Dimensionality Analysis...")

df_full_metrics, full_inertias, full_silhouettes = run_kmeans_analysis(scaled_df)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 9), full_inertias, marker='o', linestyle='--', color='green')
plt.title("Elbow Method (Full Dimensionality)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.savefig("audrius/task3_elbow.png", dpi=150, bbox_inches='tight')
print("Saved: audrius/task3_elbow.png")

best_k_full = int(df_full_metrics.loc[df_full_metrics['silhouette'].idxmax(), 'k'])
print(f"Best k for Full Dimensionality: {best_k_full}")

top_ks = df_full_metrics.nlargest(3, 'silhouette')['k'].tolist()
plt.close('all')
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i, k_val in enumerate(top_ks):
    km = KMeans(n_clusters=k_val, random_state=42).fit(scaled_df)
    labels = km.labels_
    sil_samples = silhouette_samples(scaled_df, labels)
    unique_labels = np.unique(labels)
    y_lower = 10
    
    for label in unique_labels:
        cluster_sil_values = sil_samples[labels == label]
        cluster_sil_values.sort()
        size = cluster_sil_values.shape[0]
        y_upper = y_lower + size
        axes[i].hlines(y=y_lower, xmin=0, xmax=cluster_sil_values.max(), color='purple', linewidth=2)
        axes[i].fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil_values, facecolor='purple', alpha=0.7)
        axes[i].text(0.05, (y_lower + y_upper)/2, str(label), va='center')
        y_lower = y_upper + 10
        
    axes[i].axvline(x=silhouette_score(scaled_df, labels), color="red", linestyle="--")
    axes[i].set_title(f"Silhouette Diagram (k={k_val})")
    axes[i].set_xlabel("Silhouette Coeff.")

plt.tight_layout()
plt.savefig("audrius/task3_silhouette.png", dpi=150, bbox_inches='tight')


def generate_clustergram_plot(scaled_df, output_filename):
    print("Generating K-Means Clustergram using PCA projection...")

    data_arr = scaled_df.values

    pca = PCA(n_components=1, random_state=42)
    data_pc1 = pca.fit_transform(data_arr).flatten()

    ks = list(range(2, 9))
    labels_by_k = {}
    centers_by_k = {}

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(data_arr)
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

    ax.set_title("K-Means Clustergram (Schonlau principle, PCA projection)", fontsize=12)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Cluster centers (PC1 projection)")
    ax.set_xticks(ks)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Clustergram saved: {output_filename}")
    plt.close()


# Generate clustergram
generate_clustergram_plot(scaled_df, "audrius/task3_clustergram.png")

print("\nAnalysis Complete.")
