import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples

def print_results_table(results, top_n=15):
    header = "|Nr.|Atributų pora|k=2|k=3|k=4|k=5|k=6|k=7|k=8|"
    print(header)

    for idx, r in enumerate(results[:top_n]):
        row_str = f"|{idx+1}|{r['pair']}|"
        for k in range(2, 9):
            m = r['metrics'][k]
            row_str += f"{m['inertia']:.1f} / {m['silhouette']:.3f}|"
        print(row_str)

def plot_task3_results(results, scaled_df, clustering_func, output_filename, algorithm_name):
    top_3 = results[:3]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    for i, res in enumerate(top_3):
        pair_name = res['pair']
        col1, col2 = pair_name.split(" - ")
        best_k = res['best_k']
        pair_data = scaled_df[[col1, col2]].values

        labels, _, _ = clustering_func(pair_data, best_k)

        # Duomenų pasiskirstymas
        ax_data = axes[i, 0]
        scatter = ax_data.scatter(pair_data[:, 0], pair_data[:, 1], c=labels, cmap='viridis', alpha=0.6)
        ax_data.set_title(f"{algorithm_name} klasteriai: {pair_name}\n(k={best_k})")
        ax_data.set_xlabel(col1)
        ax_data.set_ylabel(col2)
        fig.colorbar(scatter, ax=ax_data, ticks=range(best_k))

        # Inercijos grafikai
        ax_elbow = axes[i, 1]
        ks = list(range(2, 9))
        inertias = [res['metrics'][k]['inertia'] for k in ks]
        ax_elbow.plot(ks, inertias, 'go-', markersize=6)
        ax_elbow.axvline(x=best_k, color='r', linestyle='--', label=f'Geriausias k={best_k}')
        ax_elbow.set_title(f"Inercijos grafikas: {pair_name}")
        ax_elbow.set_xlabel("Klasterių skaičius (k)")
        ax_elbow.set_ylabel("Inercija")
        ax_elbow.legend()

        # Silueto diagramos
        ax_sil = axes[i, 2]
        sample_sil_vals = silhouette_samples(pair_data, labels)
        y_lower = 10

        for cl in range(best_k):
            cl_sil_vals = sample_sil_vals[labels == cl]
            cl_sil_vals.sort()
            size_cluster_cl = cl_sil_vals.shape[0]
            y_upper = y_lower + size_cluster_cl

            color = cm.nipy_spectral(float(cl) / best_k)
            ax_sil.fill_betweenx(np.arange(y_lower, y_upper), 0, cl_sil_vals,
                                  facecolor=color, edgecolor=color, alpha=0.7)
            ax_sil.text(-0.05, y_lower + 0.5 * size_cluster_cl, str(cl))
            y_lower = y_upper + 10

        avg_sil = res['global_max_sil']
        ax_sil.axvline(x=avg_sil, color="red", linestyle="--", label=f"Vidurkis: {avg_sil:.3f}")
        ax_sil.set_title(f"Silueto diagrama: {pair_name} (k={best_k})")
        ax_sil.set_xlabel("Silueto koeficientas")
        ax_sil.set_ylabel("Klasterio ID")
        ax_sil.set_yticks([])
        ax_sil.set_xlim([-0.1, 1.0])
        ax_sil.legend()

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"graphs saved at: {output_filename}")
    plt.close()

def plot_task4_results_3d(results, scaled_df, clustering_func, output_filename, algorithm_name):
    top_3 = results[:3]
    fig = plt.figure(figsize=(18, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    for i, res in enumerate(top_3):
        triplet_name = res['pair']
        col1, col2, col3 = triplet_name.split(" - ")
        best_k = res['best_k']
        triplet_data = scaled_df[[col1, col2, col3]].values

        labels, _, _ = clustering_func(triplet_data, best_k)

        # 3D Duomenų grafikai
        ax_data = fig.add_subplot(3, 3, 3 * i + 1, projection='3d')
        scatter = ax_data.scatter(
            triplet_data[:, 0], triplet_data[:, 1], triplet_data[:, 2],
            c=labels, cmap='viridis', alpha=0.6, edgecolors='none'
        )
        ax_data.set_title(f"{algorithm_name} 3D klasteriai:\n{col1}\n{col2}\n{col3} (k={best_k})", fontsize=10)
        ax_data.set_xlabel(col1, fontsize=8)
        ax_data.set_ylabel(col2, fontsize=8)
        ax_data.set_zlabel(col3, fontsize=8)

        # Inercijos grafikai
        ax_elbow = fig.add_subplot(3, 3, 3 * i + 2)
        ks = list(range(2, 9))
        inertias = [res['metrics'][k]['inertia'] for k in ks]
        ax_elbow.plot(ks, inertias, 'go-', markersize=6)
        ax_elbow.axvline(x=best_k, color='r', linestyle='--', label=f'Geriausias k={best_k}')
        ax_elbow.set_title(f"Inercija ({algorithm_name}):\n{col1}-{col2}-{col3}", fontsize=10)
        ax_elbow.set_xlabel("Klasterių skaičius (k)")
        ax_elbow.set_ylabel("Inercija")
        ax_elbow.legend()

        # Silueto diagramos
        ax_sil = fig.add_subplot(3, 3, 3 * i + 3)
        sample_sil_vals = silhouette_samples(triplet_data, labels)
        y_lower = 10

        for cl in range(best_k):
            cl_sil_vals = sample_sil_vals[labels == cl]
            cl_sil_vals.sort()
            size_cluster_cl = cl_sil_vals.shape[0]
            y_upper = y_lower + size_cluster_cl

            color = cm.nipy_spectral(float(cl) / best_k)
            ax_sil.fill_betweenx(np.arange(y_lower, y_upper), 0, cl_sil_vals,
                                  facecolor=color, edgecolor=color, alpha=0.7)
            ax_sil.text(-0.05, y_lower + 0.5 * size_cluster_cl, str(cl))
            y_lower = y_upper + 10

        avg_sil = res['global_max_sil']
        ax_sil.axvline(x=avg_sil, color="red", linestyle="--", label=f"Vidurkis: {avg_sil:.3f}")
        ax_sil.set_title(f"Silueto diagrama (k={best_k})", fontsize=10)
        ax_sil.set_xlabel("Silueto koeficientas")
        ax_sil.set_ylabel("Klasterio ID")
        ax_sil.set_yticks([])
        ax_sil.set_xlim([-0.1, 1.0])
        ax_sil.legend()

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"[visualization] 3D Grafikai sėkmingai išsaugoti: {output_filename}")
    plt.close()

def print_m_dim_table(metrics):
    print("\n### m-Dimensijų (m=10) SOM klasterizacijos rezultatai\n")
    header = "|Rezultatų įverčiai|k=2|k=3|k=4|k=5|k=6|k=7|k=8|"

    inertia_row = "|**Inercija**|"
    sil_row = "|**Silueto koef.**|"

    for k in range(2, 9):
        inertia_row += f"{metrics[k]['inertia']:.1f}|"
        sil_row += f"{metrics[k]['silhouette']:.3f}|"

    print(header)
    print(inertia_row)
    print(sil_row)
