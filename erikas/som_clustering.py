# erikas/som_clustering.py
import os
import sys
import itertools
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.data_loader import load_and_preprocess_data
from shared.visualize import print_results_table, plot_task3_results, plot_task4_results_3d
from minisom import MiniSom
from sklearn.metrics import silhouette_score

def run_som_clustering(data, k, epochs=500, random_state=42):
    data_arr = np.array(data)

    # 1D SOM struktūra su k neuronų
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
        
        # Inercijos skaičiavimui naudojame atstumus iki laimėjusių neuronų svorių
        centroid = centroids[cluster_id]
        inertia += np.sum((sample - centroid) ** 2)
        
    labels = np.array(labels)
    
    if len(np.unique(labels)) > 1:
        sil_coef = silhouette_score(data_arr, labels)
    else:
        sil_coef = -1.0
        
    return labels, inertia, sil_coef

def analyze_attribute_pairs(scaled_df):
    cols = scaled_df.columns.tolist()
    pairs = list(itertools.combinations(cols, 2))
    results = []
    
    print(f"[SOM] Skenuojama {len(pairs)} atributų porų...")
    
    for pair in pairs:
        pair_data = scaled_df[list(pair)].values
        pair_results = {
            'pair': f"{pair[0]} - {pair[1]}",
            'metrics': {}
        }
        
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
    cols = scaled_df.columns.tolist()
    triplets = list(itertools.combinations(cols, 3))
    results = []
    
    print(f"\n[SOM] Skenuojama {len(triplets)} atributų trejetų...")
    
    for triplet in triplets:
        triplet_data = scaled_df[list(triplet)].values
        triplet_results = {
            'pair': f"{triplet[0]} - {triplet[1]} - {triplet[2]}",
            'metrics': {}
        }
        
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


if __name__ == "__main__":
    _, scaled_df = load_and_preprocess_data("data.csv")
    
    analysis_results = analyze_attribute_pairs(scaled_df)
    print_results_table(analysis_results, top_n=15)
    
    output_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "som_task3_results.png")
    plot_task3_results(
        results=analysis_results, 
        scaled_df=scaled_df, 
        clustering_func=run_som_clustering, 
        output_filename=output_img, 
        algorithm_name="SOM"
    )

    triplet_results = analyze_attribute_triplets(scaled_df)
    print_results_table(triplet_results, top_n=15)

    output_img_3d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "som_task4_results.png")
    plot_task4_results_3d(
        results=triplet_results,
        scaled_df=scaled_df,
        clustering_func=run_som_clustering,
        output_filename=output_img_3d,
        algorithm_name="SOM"
    )