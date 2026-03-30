import numpy as np
from preprocess import preprocess_data
from EDA import file_name,get_input_output_data
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def normalize_data(df):
    X,y = get_input_output_data(df)
   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled,y

def get_top_eigenvecs(matrix,no_of_components=5):   # By performing the scree plot(see the bottom of this file), it was found that 5 top eigenvectors are needed
        # to get above 80% variance of the data. Hence 5 components selected.
        eigenvalues,eigenvectors = np.linalg.eigh(matrix)
        #print(eigenvalues)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        top_eigenvalues = eigenvalues[idx[:no_of_components]]
        top_eigenvectors = eigenvectors[:, idx[:no_of_components]]  # ---> Shape = (8,5). Since scatter matrix was 8*8, the eigenvectors will be (8*1) vectors.

        # Explained variance ratio:
        exp_var_ratio = eigenvalues[idx]/np.sum(eigenvalues)


        return top_eigenvalues,top_eigenvectors,exp_var_ratio

def perform_PCA(x):
    scatter_matrix = 0
    for i in range(x.shape[0]):
        column_vector = x[i][:,np.newaxis]    # x[i]'s shape will be (8,) but after this line it will be (8,1). 
        scatter_matrix += column_vector@column_vector.T
    scatter_matrix /= x.shape[0]         # Scatter matrix will be 8*8 shape

    top_eigenvalues,top_eigenvectors,evr = get_top_eigenvecs(scatter_matrix,no_of_components=5)
    
    # We originally had 8 dimensional features. But we are projecting to 5 top eigenvalues of the scatter matrix.
    # Hence now we will have 5 dimensional features of every data point.

    new_data = x@top_eigenvectors     # x.shape = 768,8 and top_eigenvectors.shape = 8,5. Hence new_data.shape = (768,5)
    return new_data,evr


if __name__ == '__main__':
    df_clean = preprocess_data(file_name)
    X,y = normalize_data(df_clean)
    new_X,evr = perform_PCA(X)


    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1 — 2D scatter of PC1 vs PC2, colored by true diabetes label
    scatter = axes[0].scatter(new_X[:, 0], new_X[:, 1],
                            c=y.to_numpy(), cmap='RdYlGn_r', alpha=0.6, edgecolors='none')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].set_title('perform_PCA — PC1 vs PC2 (colored by Outcome)')
    plt.colorbar(scatter, ax=axes[0], label='0=No Diabetes, 1=Diabetes')

    # Plot 2 — Scree plot (how much variance each component explains)
    axes[1].bar(range(1, len(evr)+1), evr * 100, color='steelblue', edgecolor='white')
    axes[1].plot(range(1, len(evr)+1), np.cumsum(evr) * 100, 'ro-', label='Cumulative')
    axes[1].axhline(y=80, color='gray', linestyle='--', label='80% threshold')
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Explained Variance (%)')
    axes[1].set_title('Scree Plot')
    axes[1].legend()

    plt.show()
    





