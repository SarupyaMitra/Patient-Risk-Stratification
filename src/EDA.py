import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_name = 'dataset/diabetes.csv'

def read_data(file_name):
    df = pd.read_csv(file_name)
    # print(df.columns)
    # print(df.describe())
    return df

def visualise_data(df):
    invalid_columns = ['Glucose','BMI','BloodPressure','SkinThickness','Insulin']   # These are the columns which cannot have '0' values.
    for col in invalid_columns:
        zero_count = (df[col] == 0).sum()
        pct = (zero_count / len(df))*100
        print(f"{col}:{zero_count} zeros ({pct:.1f}%)")

    # Visualising the zeros in invalid columns:
    df_clean = df.copy()
    df_clean[invalid_columns] = df_clean[invalid_columns].replace(0, np.nan)  # Since nan will be treated as missing value

    sns.heatmap(df_clean.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Value Map')
    plt.show()

    # Visualising the feature distributions (Univariate Analysis)
    fig,axes = plt.subplots(3,3,figsize=(14,10))
    fig.canvas.manager.set_window_title('Feature Distributions with KDE')
    axes = axes.flatten()
    for i,col in enumerate(df.columns):
        sns.histplot(data=df,x=col, kde=True, color='steelblue',ax = axes[i])
        axes[i].set_title(f'{col} Distribution with KDE')
    plt.suptitle("Feature Distributions with KDE", fontsize = 16,y=1.026)
    plt.subplots_adjust(hspace=0.7,wspace=0.5)
    plt.show()

    # Visualising how features relate to the outcome (Bivariate Analysis)

    # a) Class Wise Analysis 
    fig,axs = plt.subplots(2,4,figsize=(14,10))
    axs = axs.flatten()
    fig.canvas.manager.set_window_title('Feature Distribution with outcome')
    for i,col in enumerate(df.columns[:8]):   # I don't need the outcome column hence [:8]
        sns.kdeplot(data= df[df['Outcome'] == 0],x=col,ax=axs[i],color='green',label = 'No Diabetes')
        sns.kdeplot(data=df[df['Outcome'] == 1],x=col,ax=axs[i],color='red', label = "Diabetes")
        axs[i].set_title(col)
        axs[i].legend()
    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    plt.show()
    
    # b) Correlation HeatMap  ---  Will tell me, how each feature is correlated with each each other feature.
    fig,ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title('Correlation Matrix')
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True,ax=ax)
    plt.tight_layout()
    plt.title('Correlation Matrix')
    plt.show()

    # c) Scatter Plot for top features:
    sns.scatterplot(data=df, x='Pregnancies', y='Age', 
                hue='Outcome', palette={0:'green', 1:'red'}, alpha=0.6)
    plt.title('Glucose vs BMI by Outcome')
    plt.show() 

    # Pair Plot
    # subset = df[['Glucose', 'BMI', 'Age', 'Insulin', 'Outcome']]
    # sns.pairplot(subset, hue='Outcome', palette={0:'green', 1:'red'}, 
    #          plot_kws={'alpha':0.5})
    # plt.suptitle('Pair Plot', y=1.02)
    # plt.show()

def get_input_output_data(df):
    input_features = []
    output = []
    for col in df.columns:
        if col == 'Outcome':
            output.append(col)
        else:
            input_features.append(col)
    X = df[input_features]
    y = df[output]

    #print(X.shape)    ---> (768,8)

    return X,y

if __name__ == "__main__":
    df = read_data(file_name)
    #visualise_data(df)
    X, y = get_input_output_data(df)
    

