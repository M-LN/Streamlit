import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# File Uploader
st.title('Anxiety and Depression Analysis App')
st.markdown("## Upload your CSV file to get started")
uploaded_file = st.file_uploader('', type='csv')

if uploaded_file is not None:
    # Load the new dataset
    df = pd.read_csv(uploaded_file)
    
    st.markdown("### Dataset Overview")
    st.write("Here are the columns in your dataset:")
    st.write(df.columns)  # Display column names

    # Display first few rows of the dataset
    st.write("Here are the first few rows of your dataset:")
    st.write(df.head())

    # Display the entire dataset in a table
    st.write("Here is the entire dataset:")
    st.dataframe(df)

    # Display basic statistics of the dataset
    st.markdown("### Basic Statistics")
    st.write(df.describe())

    # Add a summary section
    st.markdown("### Dataset Summary")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")

    # Sidebar input for column selection
    st.sidebar.markdown("## Search in Dataset")
    selected_column = st.sidebar.selectbox('Select a column to search', df.columns)

    # Sidebar input for search term
    search_term = st.sidebar.text_input('Enter search term')

    # Display the search results
    if pd.api.types.is_string_dtype(df[selected_column]):
        search_results = df[df[selected_column].str.contains(search_term, case=False, na=False)]
        st.markdown("### Search Results")
        st.write(search_results)
    else:
        st.write(f"The selected column '{selected_column}' is not a string column and cannot be searched.")

    # Sidebar options for visualizations
    st.sidebar.markdown("## Visualization Options")
    show_bar_plot = st.sidebar.checkbox('Show Bar Plot', value=True)
    show_scatter_plot = st.sidebar.checkbox('Show Scatter Plot', value=True)
    show_heatmap = st.sidebar.checkbox('Show Correlation Heatmap', value=True)
    show_pair_plot = st.sidebar.checkbox('Show Pair Plot', value=True)

    # Sidebar options for scatter plot customization
    st.sidebar.markdown("## Scatter Plot Customization")
    scatter_x = st.sidebar.selectbox('Select X-axis column for scatter plot', df.columns)
    scatter_y = st.sidebar.selectbox('Select Y-axis column for scatter plot', df.columns)

    # Display visualizations in columns
    st.markdown("### Visualizations")
    col1, col2 = st.columns(2)

    if show_bar_plot:
        with col1:
            st.write('Bar plot of the selected column:')
            fig = px.histogram(df, x=selected_column, title=f'Bar plot of {selected_column}', color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig)

    if show_scatter_plot:
        with col2:
            st.write('Scatter plot of the selected columns:')
            fig = px.scatter(df, x=scatter_x, y=scatter_y, title=f'Scatter plot of {scatter_x} vs {scatter_y}', color=scatter_y, color_continuous_scale='Viridis')
            st.plotly_chart(fig)

    if show_heatmap:
        # Display a correlation heatmap
        st.markdown("### Correlation Heatmap")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        corr = numeric_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')
        st.pyplot(fig)

    if show_pair_plot:
        # Display a pair plot for numerical columns
        st.markdown("### Pair Plot of Numerical Columns")
        fig = sns.pairplot(numeric_df)
        st.pyplot(fig)

    # Add a download button for the filtered dataset
    st.markdown("### Download Filtered Dataset")
    csv = search_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='filtered_dataset.csv',
        mime='text/csv',
    )

else:
    st.write('Please upload a CSV file to proceed.')