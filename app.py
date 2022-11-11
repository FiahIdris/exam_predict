

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.linalg import Vectors
from pyspark import SparkContext
import numpy as np
import pandas as pd
from pyspark import SparkConf
from pyspark.sql import SparkSession
import numpy as np

sc = SparkContext.getOrCreate()

# Sidebar widget
st.sidebar.header('Menu')
# loading our model
model = RandomForestClassificationModel.load("modelRF")




def main():


    page = st.sidebar.selectbox(
        "Select a page", ["Homepage","Exploration", "Prediction","Model"])

    if page == "Homepage":
        homepage_screen()
    elif page == "Exploration":
        exploration_screen()
    elif page == "Model":
        model_screen()
    elif page == "Prediction":
        model_predict()


@st.cache()
def load_data():
    data = pd.read_csv('clean_data.csv', delimiter=",")
    return data


df = load_data()


def homepage_screen():
    # ["Difficulty", "Gender", "a1", "a2", "a3", "a4", "a5", "a6", "AstExpMonths", "AstWLCount", "midScore"]
    st.title('EXAM PASSED PREDICTION')
    st.header("Dataset Information")
    st.write("""  
         ## 📌 About Dataset  
            Difficulty - Subject difficulty level  
            Gender - Students gender  
            A1 - First meeting presence  
            A2 - Second meeting presence  
            A3 - Third meeting presence  
            A4 - Fourth meeting presence  
            A5 - Fifth meeting presence  
            A6 - Sixth meeting presence  
            AstExpMonths - Assistant experience length  
            AstWLCount - Total warning letter from assistant  
            MidScore - Students score on mid semester  
        
        The dataset contains of 361 instances and 11 columns
        
     """)
    
    # Load data
    df = load_data()


    if st.checkbox('See clean dataset'):
     
        data_load_state = st.text('Loading data...')
        
        st.write(df)
        data_load_state.text('')
        
    st.write("""
    ## 📌 Target Label Frequency  
    
    """)
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    df['passed'].value_counts().plot(kind='bar', ax=axs[0])
    df['passed'].value_counts().plot.pie(
        autopct='%1.1f%%', startangle=90, ax=axs[1], colors=['green', 'teal'])
    st.write(fig)


def exploration_screen():
    st.title("Data Exploration")
    st.write(""" 
        This page contains general exploratory data analysis in order to get basic insight of the dataset information and get the feeling about what this dataset is about.
    """)

    st.write("""
        ## 📌 Correlational Matrix  
        
    """)
    # Matrix correlation.

    df_pandas = df[["Difficulty", "Gender", "a1", "a2", "a3", "a4", "a5", "a6", "AstExpMonths", "AstWLCount","midScore","passed"]]
    corr_matrix = df_pandas.corr()
    mask = np.triu(corr_matrix)
    plt.figure(figsize=(11,5))

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(corr_matrix, annot=True, ax=ax,mask= mask )
    plt.title('Correlation Among Features')
    st.write(fig)

    st.write("""
        ## 📌  Features Correlation Value toward Target Column
        
    """)
    # Display correlation towards target column
    fig, axs = plt.subplots(figsize=(10, 4))
    corr = df.corr()['passed'].reset_index()
    # corr.drop( axis=0, inplace=True)
    sns.barplot(data=corr, x='index', y='passed', ax=axs)
    plt.xticks(rotation=70)
    st.write(fig)

#     st.write("""
#         ## 📌 Target Label Frequency  
#         
#     """)
# 
#     fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
#     df['passed'].value_counts().plot(kind='bar', ax=axs[0])
#     df['passed'].value_counts().plot.pie(
#         autopct='%1.1f%%', startangle=90, ax=axs[1], colors=['green', 'teal'])
#     st.write(fig)
# 
#     st.write("""
#         ## 📌  AstExpMonths and passed
#         
#     """)
#     fig, axs = plt.subplots()
#     sns.scatterplot(data=df, x='AstExpMonths', y='AstWLCount', hue='passed', ax=axs)
#     st.write(fig)

    st.write("""
        ## 📌 MidScore and Passed
        
    """)
    fig, axs = plt.subplots()
    sns.scatterplot(data=df, x='midScore', y='AstExpMonths', hue='passed', ax=axs)
    # sns.kdeplot(data=df, x='midScore', hue='passed', ax=axs)
    plt.show()
    st.write(fig)

    st.write("""
        ## 📌  Difficulty and Passed
        
    """)
    fig, axs = plt.subplots()
    sns.countplot(data=df, x='Difficulty', hue='passed', ax=axs)
    st.write(fig)
    
    st.write("""
        ## 📌  Gender and Passed
        
    """)
    fig, axs = plt.subplots()
    sns.countplot(data=df, x='Gender', hue='passed', ax=axs)
    st.write(fig)


def model_screen():
    st.title("Model")
    st.write(""" 
             
             """)
    model_selected = st.selectbox("Select model: ", [
                                  'Logistic Regression', 'RandomForest Classifier'])
    if model_selected == 'RandomForest Classifier':
        # TP=39, FP:0, FN=3, TN=1
        matrix = [[1, 0], [3, 39]]
        accuracy = 0.9302325581395349
        precision = 1.0
        recall = 0.9285714285714286
     
        validation(matrix, accuracy, precision,
                   recall)
    if model_selected == 'Logistic Regression':
        # TP=37, FP:2, FN=4, TN=0
        matrix = [[0, 2], [4, 37]]
        accuracy = 0.8604651162790697
        precision = 0.9487179487179487
        recall = 0.9024390243902439
     
        validation(matrix, accuracy, precision,
                   recall)


def validation(matrix, cross_score, prec_score, rec_score):
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt='g', ax=ax)
    ax.set_xlabel('PREDICTED')
    ax.set_ylabel('ACTUAL')
    ax.set_title('Confusion Matrix')
    st.write(fig)
    st.write(f"""
    👉 Cross Validation  mean: {cross_score}   
    👉 Precision : {prec_score}  
    👉 Recall : {rec_score}  
             """)


def model_predict():
    # ["Difficulty", "Gender", "a1", "a2", "a3", "a4", "a5", "a6", "AstExpMonths", "AstWLCount", "midScore"]

    st.title("Prediction")
    st.write("### Field this form to predict the student passed!")
    
    difficulty = st.radio("Difficulty Level", [1, 2, 3,])
    gender = st.radio("Gender", ['male', 'female'])
    meeting1 = st.radio("Meeting 1", ['presence', 'not presence'])
    meeting2 = st.radio("Meeting 2", ['presence', 'not presence'])
    meeting3 = st.radio("Meeting 3", ['presence', 'not presence'])
    meeting4 = st.radio("Meeting 4", ['presence', 'not presence'])
    meeting5 = st.radio("Meeting 5", ['presence', 'not presence'])
    meeting6 = st.radio("Meeting 6", ['presence', 'not presence'])
    astExpMonth = st.slider("Assistant Experience Month", 1, 160)
    astWLcount = st.slider("Assistant Total Warning Letter", 0, 10)
    midScore = st.slider("Mid Score", 0, 100)
    
    submit_button = st.button("Predict")


    if gender == 'male':
        gender = 1
    else:
        gender = 0

    if meeting1 == 'presence':
        meeting1 = 1
    else:
        meeting1 = 0
        
    if meeting2 == 'presence':
        meeting2 = 1
    else:
        meeting2 = 0
        
    if meeting3 == 'presence':
        meeting3 = 1
    else:
        meeting3 = 0

    if meeting4 == 'presence':
        meeting4 = 1
    else:
        meeting4 = 0
        
    if meeting5 == 'presence':
        meeting5 = 1
    else:
        meeting5 = 0

    if meeting6 == 'presence':
        meeting6 = 1
    else:
        meeting6 = 0
        

    new_data = [difficulty,gender,meeting1,meeting2,meeting3,meeting4,meeting5,meeting6,astExpMonth,astWLcount,midScore]

    if submit_button:
        result = model.predict(Vectors.dense(new_data))
      
        print('=========result')
        print(new_data)
        print(result)
        if result == 1.0:
            result = "🤩 Hooray..... We predict you will passed in the mid semester."
            
        else:
            result = '😭 Sorry. It seems that you wont pass in the mid semester.'
        st.success(
            '{}'.format(result))


main()
