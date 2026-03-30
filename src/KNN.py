import numpy as np
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Kmeans import perform_best_Kmeans
from EDA import get_input_output_data,file_name,read_data
from preprocess import preprocess_data,get_train_test_split

def perform_KNN():
    df_clean = preprocess_data(file_name)
    X_original,y = get_input_output_data(df_clean)
    # print(y.shape) --> (768,1)
    cluster_labels,_,label_map = perform_best_Kmeans(no_clusters = 3)  # label_map = {0:'High Risk',1:'Low Risk',2:'Moderate Risk'}
    class_names = [label_map[k] for k in sorted(label_map.keys())]
    # print(cluster_labels.shape) ---> (768,)
    X_train,X_test,y_train,y_test = get_train_test_split(X_original,cluster_labels)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    opt_k = get_optimal_K_for_KNN(X_train_scaled,X_test_scaled,y_train,y_test)

    knn_final = KNeighborsClassifier(n_neighbors=opt_k)
    knn_final.fit(X_train_scaled,y_train)
    y_pred = knn_final.predict(X_test_scaled)

    print(classification_report(y_test,y_pred,target_names=class_names))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'KNN Confusion Matrix (k={opt_k})')
    plt.tight_layout()
    plt.show()

    # print(np.unique(y_test))
    # print(class_names)

    return knn_final

def get_optimal_K_for_KNN(X_train,X_test,y_train,y_test):
    K = range(1,21)
    train_acc = []
    test_acc = []
    
    for k in K:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        train_acc.append(accuracy_score(y_train,knn.predict(X_train)))
        test_acc.append(accuracy_score(y_test,knn.predict(X_test)))

    plt.figure("Optimal K for KNN",figsize=(9, 4))
    plt.plot(K, train_acc, 'bo-', label='Train accuracy')
    plt.plot(K, test_acc, 'ro-', label='Test accuracy')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('KNN — Finding optimal k')
    plt.legend()
    plt.show()

    test_acc_arr = np.array(test_acc)
    optimal_k = np.argmax(test_acc_arr) + 1
    print(f"Optimal k is {optimal_k} and its Test Accuracy is {test_acc[optimal_k-1]}")

    return optimal_k

if __name__ == '__main__':
    perform_KNN()