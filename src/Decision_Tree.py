import numpy as np
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

from Kmeans import perform_best_Kmeans
from EDA import get_input_output_data,file_name,read_data
from preprocess import preprocess_data,get_train_test_split


def decision_tree():
    df_clean = preprocess_data(file_name)
    X_original,y = get_input_output_data(df_clean)
    # print(y.shape) --> (768,1)
    cluster_labels,_,label_map = perform_best_Kmeans(no_clusters = 3)
    # print(cluster_labels.shape) ---> (768,)
    class_names = [label_map[k] for k in sorted(label_map.keys())]

    X_train,X_test,y_train,y_test = get_train_test_split(X_original,cluster_labels)

    dt = DecisionTreeClassifier(max_depth=4,random_state=42)
    dt.fit(X_train,y_train)

    # y_pred = dt.predict(X_test)

    # print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))


    features = df_clean.columns.drop('Outcome').to_list()
    
    plt.figure(figsize=(24, 8))
    plot_tree(dt,
            feature_names=features,
            class_names=class_names,
            filled=True,       # colors the nodes by class
            rounded=True,      # rounded box corners
            fontsize=9)
    plt.title('Decision Tree — Patient Risk Stratification')
    plt.tight_layout()
    plt.show()

    return dt




if __name__ == '__main__':
    decision_tree()