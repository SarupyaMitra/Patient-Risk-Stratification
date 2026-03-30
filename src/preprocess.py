import numpy as np
import pandas as pd
from EDA import file_name
from sklearn.model_selection import train_test_split

def preprocess_data(file_name):
    df = pd.read_csv(file_name)
    invalid_columns = ['Glucose','BMI','BloodPressure','SkinThickness','Insulin']
    df_clean = df.copy()
    for col in invalid_columns:
        median_value_col = df[col].median()
        df_clean[col] = df[col].replace(0,median_value_col)  # Replacing the zero values by the median of that column

    # for col in invalid_columns:
    #     zero_count_col = (df_clean[col]==0).sum()
    #     pct = (zero_count_col / len(df_clean))*100
    #     print(f"{col}:{zero_count_col} zeros ({pct:.1f}%)")
    
    #print(f'{df.shape} , {type(df)}')

    return df_clean

def get_train_test_split(X,y_clusters,test_size=0.2,random_state=42):
    X_train, X_test,y_train,y_test = train_test_split(
        X,
        y_clusters,
        test_size=test_size,
        random_state=random_state
    )
    return X_train,X_test,y_train,y_test
if __name__ == "__main__":
    df_clean = preprocess_data(file_name)


