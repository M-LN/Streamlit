import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

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

    # Display the entire dataset in a table
    st.write("Here is the entire dataset:")
    st.dataframe(df)

    # Display basic statistics of the dataset
    st.write("Basic statistics of your dataset:")
    st.write(df.describe())

    # Sidebar input for column selection
    selected_column = st.sidebar.selectbox('Select a column to search', df.columns)

    # Sidebar input for search term
    search_term = st.sidebar.text_input('Enter search term')

    # Display the search results
    if pd.api.types.is_string_dtype(df[selected_column]):
        search_results = df[df[selected_column].str.contains(search_term, case=False, na=False)]
        st.write("Search results:")
        st.write(search_results)
    else:
        st.write(f"The selected column '{selected_column}' is not a string column and cannot be searched.")

    # Display a bar plot of the selected column
    st.write('Bar plot of the selected column:')
    fig = px.histogram(df, x=selected_column)
    st.plotly_chart(fig)

    # Display a scatter plot from search results if 'Value' column exists
    if 'Value' in search_results.columns:
        st.write('Scatter plot of the search results:')
        fig = px.scatter(search_results, x=selected_column, y='Value')
        st.plotly_chart(fig)
    else:
        st.write("The 'Value' column does not exist in the search results, so a scatter plot cannot be created.")

    # Display a correlation heatmap
    st.write('Correlation heatmap:')
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

    # Display a pair plot for numerical columns
    st.write('Pair plot of numerical columns:')
    fig = sns.pairplot(numeric_df)
    st.pyplot(fig)

else:
    st.write('Please upload a CSV file to proceed.')