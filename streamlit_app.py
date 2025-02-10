import streamlit as st
import pandas as pd
import plotly.express as px

# File Uploader
st.title('Anxiety and Depression Analysis App')
uploaded_file = st.file_uploader('Upload your CSV file', type='csv')

if uploaded_file is not None:
    # Load the new dataset
    df = pd.read_csv(uploaded_file)
    st.write("Here are the columns in your dataset:")
    st.write(df.columns)  # Display column names

    # Display first few rows of the dataset
    st.write("Here are the first few rows of your dataset:")
    st.write(df.head())

    # Display basic statistics of the dataset
    st.write("Basic statistics of your dataset:")
    st.write(df.describe())

    # Sidebar input for column selection
    selected_column = st.sidebar.selectbox('Select a column to search', df.columns)

    # Sidebar input for search term
    search_term = st.sidebar.text_input('Enter search term')
    # Display the search results
    search_results = df[df[selected_column].str.contains(search_term, case=False)]
    st.write(search_results)

    # Display a bar plot of the selected column
    st.write('Bar plot of the selected column:')
    fig = px.histogram(df, x=selected_column)
    st.plotly_chart(fig)

    #display a scatter plot from search results
    st.write('Scatter plot of the search results:')
    fig = px.scatter(search_results, x='anxiety', y='depression')
    st.plotly_chart(fig)
    
    
else:
    st.write('Please upload a CSV file to proceed.')