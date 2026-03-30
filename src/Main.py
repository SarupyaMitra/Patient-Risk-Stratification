import numpy as np
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,recall_score, classification_report, confusion_matrix,precision_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Decision_Tree import decision_tree
from KNN import perform_KNN
from preprocess import preprocess_data,get_train_test_split
from EDA import get_input_output_data,file_name
from Kmeans import perform_best_Kmeans


def get_metrics(y_test,y_pred,model_name):
    return {
        'Model': model_name,
        'Accuracy' : round(accuracy_score(y_test,y_pred),3),
        'Precision' : round(precision_score(y_test,y_pred,average='weighted'),3),
        'Recall':        round(recall_score(y_test, y_pred, average='weighted'), 3),
        'F1 (weighted)': round(f1_score(y_test, y_pred, average='weighted'), 3),
        'F1 (macro)':    round(f1_score(y_test, y_pred, average='macro'), 3),
    }


if __name__=='__main__':
    df_clean = preprocess_data(file_name)
    X_org,y = get_input_output_data(df_clean)
    cluster_labels,_,label_map = perform_best_Kmeans(no_clusters = 3)
    class_names = [label_map[k] for k in sorted(label_map.keys())]
    X_train,X_test,y_train,y_test = get_train_test_split(X_org,cluster_labels)

    dt = decision_tree()
    knn = perform_KNN()

    y_pred_dt = dt.predict(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_pred_knn = knn.predict(X_test_scaled)

    results = pd.DataFrame([
    get_metrics(y_test, y_pred_dt,  'Decision Tree'),
    get_metrics(y_test, y_pred_knn, 'KNN')
    ])

    results.set_index('Model',inplace=True)
    print(results)


    dt_acc = round(accuracy_score(y_test,y_pred_dt),3)
    dt_precision = round(precision_score(y_test,y_pred_dt,average='weighted'),3)
    dt_recall = round(recall_score(y_test, y_pred_dt, average='weighted'), 3)
    dt_f1_weighted = round(f1_score(y_test, y_pred_dt, average='weighted'), 3)
    dt_f1_macro = round(f1_score(y_test, y_pred_dt, average='macro'), 3)

    knn_acc =round(accuracy_score(y_test,y_pred_knn),3)
    knn_precision = round(precision_score(y_test,y_pred_knn,average='weighted'),3)
    knn_recall = round(recall_score(y_test, y_pred_knn, average='weighted'), 3)
    knn_f1_weighted = round(f1_score(y_test, y_pred_knn, average='weighted'), 3)
    knn_f1_macro = round(f1_score(y_test, y_pred_knn, average='macro'), 3)


    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 (Weighted)', 'F1 (Macro)']

    dt_scores  = [dt_acc, dt_precision, dt_recall, dt_f1_weighted, dt_f1_macro]
    knn_scores = [knn_acc, knn_precision, knn_recall, knn_f1_weighted, knn_f1_macro]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))

    bars1 = ax.bar(x - width/2, dt_scores,  width, label='Decision Tree', color='steelblue',  edgecolor='white')
    bars2 = ax.bar(x + width/2, knn_scores, width, label='KNN',           color='darkorange', edgecolor='white')

    # Add value labels on top of each bar
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Decision Tree vs KNN — Metric Comparison')
    ax.legend()
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    # plt.savefig('results/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, y_pred, title in zip(axes,
                                [y_pred_dt, y_pred_knn],
                                ['Decision Tree', f'KNN (k={16})']):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    ax=ax)
        ax.set_title(title)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    plt.suptitle('Confusion Matrix Comparison', fontsize=14, y=1)
    plt.tight_layout()
    fig.subplots_adjust(top=0.9) 
    # plt.savefig('results/confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()






