import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def simulate_visualization():
    print("Generating Latent Space Visualization...")
    
    # 1. Simulate Embeddings (since we don't have the full dataset loaded)
    # We generate 100 fake "vectors" of size 512 to mimic the model output
    # simulating 3 distinct clusters (e.g., Rock, Jazz, HipHop)
    cluster_1 = np.random.normal(loc=0.0, scale=0.5, size=(30, 512))
    cluster_2 = np.random.normal(loc=5.0, scale=1.0, size=(30, 512))
    cluster_3 = np.random.normal(loc=-5.0, scale=0.8, size=(40, 512))
    
    # Combine them into one "dataset"
    embeddings = np.vstack([cluster_1, cluster_2, cluster_3])
    labels = ([0] * 30) + ([1] * 30) + ([2] * 40) # Fake genre labels
    
    # 2. Reduce dimensions using PCA (Project 512d -> 2d)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(embeddings)
    
    # 3. Plot the results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                          c=labels, cmap='viridis', alpha=0.7)
    
    plt.title("Latent Space Projection (PCA Simulation)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Cluster/Genre")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 4. Save the plot
    output_path = "latent_space_visualization.png"
    plt.savefig(output_path)
    print(f" Saved plot to {output_path}")

if __name__ == "__main__":
    simulate_visualization()