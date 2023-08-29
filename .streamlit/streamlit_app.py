import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# import load_iris function from datasets module
from sklearn import neighbors, datasets
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


sns.set_palette('husl')

st.set_page_config(page_title="Streamlit Dashboard", layout="wide")

st.write("""
# Welcome to My Dashboard.
     
[Utus Karta Sanggam](https://www.linkedin.com/in/utusks01)

""")
         
add_selectitem = st.sidebar.selectbox("Want to open about?", ("Iris Prediction!", "Scatter Plot Sepal!", "Scatter Plot Petal!", "Plot Sepal and Petal!", "EDA!"))

def iris():
    st.write("""
        Bagian ini Prediksi untuk **Species Bunga Iris**
                          
        """)

    # Collects user input features into dataframe
    st.sidebar.header('User Input :')
    def user_input():
        st.sidebar.header("Input Manual")
        SepalLengthCm = st.sidebar.slider('Sepal Length (cm)', 4.0, 10.0, 7.0)
        SepalWidthCm = st.sidebar.slider('Sepal Width (cm)', 2.0, 6.0, 3.2)
        PetalLengthCm = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 4.7)
        PetalWidthCm = st.sidebar.slider('Petal Width (cm)', 0.1, 3.0, 1.4)
        data ={'SepalLengthCm' : SepalLengthCm, 'SepalWidthCm' : SepalWidthCm,
               'PetalLengthCm': PetalLengthCm, 'PetalWidthCm' : PetalWidthCm}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input()

    #Add picture
    img_iris = Image.open("irises.jpg")
    st.image(img_iris,width=200)

    if st.sidebar.button('Predict'):
        df = input_df

        st.subheader('User Input parameters')
        st.write(df)
                
        iris = datasets.load_iris()
        X = iris.data
        Y = iris.target

        clf = RandomForestClassifier()
        clf.fit(X, Y)

        prediction = clf.predict(df)
        prediction_proba = clf.predict_proba(df)

        # st.subheader('Class labels and their corresponding index number')
        # st.write(iris.target_names)

        result = ["Iris_sentosa" if prediction == 0 else ("Iris_versicolor" if prediction == 1 else "Iris_virginica")]
        st.subheader("Prediction:")
        output = str(result[0])
         
        with st.spinner("Please wait for the moment !"):
            time.sleep(4)
        st.success(f"Prediction is {output}")

        st.subheader('Prediction Probability')
        st.write(prediction_proba)



def Sepal():
    # Load iris dataset
    iris = datasets.load_iris()

    # Extract the values for features and create a list called featuresAll
    featuresAll = [sum(observation) for observation in iris.data]

    # Extract the values for targets
    targets = iris.target

    # Finding the relationship between Sepal Length and Sepal width
    sepal_lengths = [feature[0] for feature in iris.data]
    sepal_widths = [feature[1] for feature in iris.data]

    groups = ('Iris_setosa', 'Iris_versicolor', 'Iris_virginica')
    colors =  ('magenta', 'Chocolate', 'green')
    data = (
        (sepal_lengths[:50], sepal_widths[:50]),
        (sepal_lengths[50:100], sepal_widths[50:100]),
        (sepal_lengths[100:150], sepal_widths[100:150])
    )

    st.subheader('Scatter Plot Relation between Sepal Length and Sepal Width')

    # Create a Matplotlib figure
    fig, ax = plt.subplots()

    # Scatter plot using Matplotlib
    for item, color, group in zip(data, colors, groups):
        x, y = item
        ax.scatter(x, y, color=color, alpha=1, label=group)

    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
    ax.legend()

    # Display the Matplotlib figure using Streamlit
    st.pyplot(fig)



def Petal():
# Load iris dataset
    iris = datasets.load_iris()
    features = iris.data

    # Finding the relationship between Petal Length and Petal width
    featuresAll = []
    targets = []
    for feature in features:
        featuresAll.append(feature[2])  # Petal length
        targets.append(feature[3])  # Petal width

    groups = ('Iris_setosa', 'Iris_versicolor', 'Iris_virginica')
    colors = ('magenta', 'Chocolate', 'green')
    data = (
        (featuresAll[:50], targets[:50]),
        (featuresAll[50:100], targets[50:100]),
        (featuresAll[100:150], targets[100:150])
    )

    st.subheader('Scatter Plot Relation between Petal Length vs Petal Width')

    # Create a Matplotlib figure
    fig, ax = plt.subplots()

    # Scatter plot using Matplotlib
    for item, color, group in zip(data, colors, groups):
        x0, y0 = item
        ax.scatter(x0, y0, color=color, alpha=1, label=group)

    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Petal Width')
    ax.set_title('Iris Dataset Scatter Plot')
    ax.legend()

    # Display the Matplotlib figure using Streamlit
    st.pyplot(fig)


    
def ViolinPlot():
    # Load iris dataset
    iris_data = sns.load_dataset("iris")

    # Set up the layout
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Create violin plots
    sns.violinplot(x='species', y='petal_length', data=iris_data, ax=axs[0, 0])
    sns.violinplot(x='species', y='petal_width', data=iris_data, ax=axs[0, 1])
    sns.violinplot(x='species', y='sepal_length', data=iris_data, ax=axs[1, 0])
    sns.violinplot(x='species', y='sepal_width', data=iris_data, ax=axs[1, 1])

    # Set titles
    axs[0, 0].set_title('Petal Length')
    axs[0, 1].set_title('Petal Width')
    axs[1, 0].set_title('Sepal Length')
    axs[1, 1].set_title('Sepal Width')

    # Set layout spacing
    plt.tight_layout()

    # Display the plot using Streamlit
    st.pyplot(fig)



def EDA():
    # Load iris dataset
    data = sns.load_dataset("iris")

    # Encode target labels
    le = LabelEncoder()
    data['species_encoded'] = le.fit_transform(data['species'])

    st.title('Exploratory Data Analysis (EDA) of Iris Dataset')

    st.write('Data Top 5:')
    st.write(data.head(5))

    st.write('Dimension Row x Column :')    
    st.write(data.shape)

    # # Display data info
    # st.write('Data Info:')
    # st.write(data.info())

    st.write('Null Values Check:')
    st.write(data.isnull().sum())

    # Display value counts
    st.write('Value Counts:')
    st.write(data.value_counts())

    # Display histograms
    st.write('Histograms:')
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.ravel()
    for i, column in enumerate(data.columns[:-2]):  # Exclude encoded and species columns
        sns.histplot(data[column], ax=axs[i])
    st.pyplot(fig)

    # Display box plots
    st.write('Box Plots:')
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.ravel()
    for i, column in enumerate(data.columns[:-2]):  # Exclude encoded and species columns
        sns.boxplot(x='species', y=column, data=data, ax=axs[i])
    st.pyplot(fig)

    # Display correlation heatmap
    st.write('Correlation Heatmap:')
    corr_matrix = data.drop(['species', 'species_encoded'], axis=1).corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=True, linewidths=1)
    st.pyplot()


if add_selectitem == "Iris Prediction!":
    iris()
elif add_selectitem == "Scatter Plot Sepal!":
    Sepal()
elif add_selectitem == "Scatter Plot Petal!":
    Petal()
elif add_selectitem == "Plot Sepal and Petal!":
    ViolinPlot()
elif add_selectitem == "EDA!":
    EDA()
