import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import plotly.colors
from scipy import stats
import warnings

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Set page configuration
st.set_page_config(
    page_title="Mental Health Data Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/mental-health-dashboard/issues',
        'Report a bug': 'https://github.com/yourusername/mental-health-dashboard/issues',
        'About': """
        # Mental Health Data Analysis Dashboard
        
        This app helps analyze anxiety and depression data through interactive visualizations.
        
        **Version:** 1.0.0
        **GitHub:** https://github.com/yourusername/mental-health-dashboard
        
        Made with ❤️ for mental health awareness
        """
    }
)

# Enhanced CSS styling
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stTitle { color: #2c3e50; font-size: 2.5rem; text-align: center; margin-bottom: 2rem; }
    .stAlert { padding: 1rem; margin-bottom: 1rem; border-radius: 0.5rem; }
    .element-container { background-color: #ffffff; padding: 1.5rem; border-radius: 0.8rem; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem; }
    .stButton>button { background-color: #2c3e50; color: white; border-radius: 0.3rem;
                      padding: 0.5rem 1rem; transition: all 0.3s ease; }
    .stButton>button:hover { background-color: #34495e; transform: translateY(-2px); }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { padding: 1rem 2rem; }
    .plot-container { background-color: #f8f9fa; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0; }
    </style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.markdown("### 🛠️ Analysis Settings")
theme = st.sidebar.selectbox("Select Theme", ["light", "dark", "plotly", "plotly_dark"])
plot_theme = st.sidebar.selectbox("Plot Theme", ["plotly", "plotly_white", "plotly_dark", "seaborn"])
color_scale = st.sidebar.selectbox("Color Scale", ["Viridis", "Plasma", "Inferno", "Magma", "RdBu"])

# Main app title and description
st.title('🧠 Mental Health Data Analysis Dashboard')
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p style='font-size: 1.2rem; color: #666;'>
        Analyze anxiety and depression data through interactive visualizations and statistical analysis
    </p>
    <p style='font-size: 0.9rem; color: #888;'>
        📊 Upload your CSV • 📈 Analyze trends • 🔍 Discover insights • 📥 Export results
    </p>
</div>
""", unsafe_allow_html=True)

# Add a info banner for demo
st.info("👆 **New to the app?** Try uploading the sample data file (`sample_data.csv`) included in this repository to see the dashboard in action!", icon="💡")

# File uploader
uploaded_file = st.file_uploader('Choose a CSV file', type='csv', help='Upload a CSV file containing your data')

if uploaded_file is not None:
    try:
        # Load and process data with progress indicator
        with st.spinner('Loading data... Please wait.'):
            progress_bar = st.progress(0)
            df = pd.read_csv(uploaded_file)
            progress_bar.progress(100)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📊 Statistics", "🔍 Analysis", "📈 Advanced"])
        
        # Tab 1: Overview
        with tab1:
            st.markdown("### Dataset Overview")
            
            # Data quality check
            with st.expander("🔍 Data Quality Report"):
                quality_issues = []
                missing = df.isnull().sum()
                if missing.any():
                    quality_issues.append(f"Missing values found in columns: {missing[missing > 0].index.tolist()}")
                duplicates = df.duplicated().sum()
                if duplicates > 0:
                    quality_issues.append(f"Found {duplicates} duplicate rows")
                
                if quality_issues:
                    for issue in quality_issues:
                        st.warning(issue)
                else:
                    st.success("No major data quality issues found!")

            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Dataset Columns:")
                st.write(df.columns.tolist())
                
            with col2:
                st.write("#### Quick Summary:")
                st.write(f"Total Rows: {df.shape[0]:,}")
                st.write(f"Total Columns: {df.shape[1]}")
                st.write(f"Missing Values: {df.isna().sum().sum():,}")

            # Display first few rows with styling
            st.write("#### Sample Data:")
            st.dataframe(df.head(), use_container_width=True)
            
            # Display missing values heatmap
            st.write("#### Missing Values Visualization:")
            fig = px.imshow(df.isnull(), 
                          aspect="auto",
                          color_continuous_scale=['#ffffff', '#2c3e50'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Add download button with better styling
            st.download_button(
                "📥 Download Sample Data",
                df.head().to_csv(index=False).encode('utf-8'),
                "sample_data.csv",
                "text/csv",
                key='download-sample'
            )

        # Tab 2: Statistics
        with tab2:
            st.markdown("### Statistical Analysis")
            # Display basic statistics with better formatting
            st.write("#### Numerical Statistics:")
            st.dataframe(df.describe().round(2), use_container_width=True)
            
            # Add correlation analysis
            if len(df.select_dtypes(include=[np.number]).columns) > 0:
                # Add correlation threshold slider
                corr_threshold = st.slider(
                    "Correlation Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="Show correlations above this threshold"
                )
                
                st.write("#### Correlation Analysis:")
                corr_matrix = df.select_dtypes(include=[np.number]).corr()
                fig = px.imshow(corr_matrix, 
                              color_continuous_scale='RdBu_r',
                              title='Correlation Matrix')
                st.plotly_chart(fig, use_container_width=True)
                
                # Add correlation significance test
                st.write("#### Correlation Significance:")
                significant_corr_data = []
                for col1 in corr_matrix.columns:
                    for col2 in corr_matrix.columns:
                        if col1 != col2:
                            corr = corr_matrix.loc[col1, col2]
                            if abs(corr) > corr_threshold:  # Use the slider value
                                significant_corr_data.append({
                                    'Variable 1': col1,
                                    'Variable 2': col2,
                                    'Correlation': corr
                                })
                                
                significant_corr = pd.DataFrame(significant_corr_data)
                if not significant_corr.empty:
                    st.dataframe(significant_corr.sort_values('Correlation', ascending=False),
                                use_container_width=True)
                else:
                    st.info(f"No correlations found above threshold {corr_threshold}")

        # Tab 3: Analysis
        with tab3:
            st.markdown("### Detailed Analysis")
            analysis_type = st.radio(
                "Choose Analysis Type",
                ["Time Series", "Distribution", "Correlation", "Custom"],
                horizontal=True
            )
            
            # Get available columns
            date_columns = df.select_dtypes(include=['datetime64', 'object']).columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if analysis_type == "Time Series" and len(date_columns) > 0 and len(numeric_cols) > 0:
                st.write("#### Time Series Analysis")
                date_col = st.selectbox("Select Date Column", date_columns)
                metric_col = st.selectbox("Select Metric to Analyze", numeric_cols)
                
                # Convert to datetime if not already
                try:
                    # Try common date formats first to avoid parsing warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
                    
                    # Check if conversion was successful
                    if df[date_col].isna().all():
                        st.warning(f"Could not convert column '{date_col}' to datetime format")
                    else:
                        # Remove rows with invalid dates
                        df = df.dropna(subset=[date_col])
                    
                    # Create time series plot
                    fig = px.line(df, x=date_col, y=metric_col,
                                title=f'{metric_col} Over Time')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add trend analysis
                    ma_window = st.select_slider(
                        "Moving Average Window",
                        options=[3, 7, 14, 30, 60],
                        value=7,
                        help="Select window size for moving average"
                    )
                    
                    # Calculate moving averages
                    df_temp = df.copy()
                    df_temp[f'MA{ma_window}'] = df_temp[metric_col].rolling(window=ma_window).mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_temp[date_col], y=df_temp[metric_col],
                                           name='Raw Data'))
                    fig.add_trace(go.Scatter(x=df_temp[date_col], y=df_temp[f'MA{ma_window}'],
                                           name=f'{ma_window}-day Moving Average'))
                    fig.update_layout(title='Trend Analysis',
                                    xaxis_title='Date',
                                    yaxis_title='Value')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as date_error:
                    st.warning(f"Could not convert selected column to date format: {str(date_error)}")
            
            elif analysis_type == "Distribution" and len(numeric_cols) > 0:
                st.write("#### Distribution Analysis")
                selected_col = st.selectbox("Select Column for Distribution", numeric_cols)
                fig = px.histogram(df, x=selected_col, marginal="box", 
                                 title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)

        # Tab 4: Advanced
        with tab4:
            st.markdown("### Advanced Analysis")
            
            # PCA Analysis for numerical columns
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                st.write("#### Principal Component Analysis (PCA)")
                
                # Handle missing values
                imputer = SimpleImputer(strategy='mean')
                imputed_features = imputer.fit_transform(numeric_df)
                
                # Standardize the features
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(imputed_features)
                
                # Perform PCA
                pca = PCA()
                pca_features = pca.fit_transform(scaled_features)
                
                # Plot explained variance ratio
                fig = px.line(y=np.cumsum(pca.explained_variance_ratio_),
                             title='Cumulative Explained Variance Ratio',
                             labels={'index': 'Number of Components',
                                    'y': 'Cumulative Explained Variance Ratio'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot first two principal components
                if pca_features.shape[1] >= 2:
                    fig = px.scatter(x=pca_features[:, 0], y=pca_features[:, 1],
                                   title='First Two Principal Components',
                                   labels={'x': 'First Principal Component',
                                          'y': 'Second Principal Component'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explained variance information
                    st.write("#### Explained Variance Ratio:")
                    explained_variance = pd.DataFrame({
                        'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                        'Explained Variance Ratio': pca.explained_variance_ratio_,
                        'Cumulative Explained Variance': np.cumsum(pca.explained_variance_ratio_)
                    })
                    st.dataframe(explained_variance.round(4), use_container_width=True)
            
                # Add PCA components selector
                n_components = st.slider(
                    "Number of PCA Components",
                    min_value=2,
                    max_value=min(10, len(numeric_df.columns)),
                    value=2,
                    help="Select number of principal components to display"
                )

            # Export options
            st.markdown("### 📤 Export Options")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📊 Export Full Dataset",
                    df.to_csv(index=False).encode('utf-8'),
                    "full_dataset.csv",
                    "text/csv",
                    key='download-full'
                )
            with col2:
                st.download_button(
                    "📑 Export Analysis Report",
                    f"""Analysis Report
                    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    Dataset Size: {df.shape}
                    Missing Values: {df.isna().sum().sum()}""",
                    "analysis_report.txt",
                    key='download-report'
                )

    except Exception as e:
        st.error(f"""
        ### 😕 Oops! Something went wrong
        
        **Error Details:** {str(e)}
        
        **Possible Solutions:**
        - Check if your data format is correct
        - Ensure all required columns are present
        - Make sure numerical columns contain valid numbers
        
        Need help? Check the documentation below 👇
        """)
        
        with st.expander("📚 Documentation"):
            st.markdown("""
            ### Required Data Format
            - CSV file with headers
            - At least one date column
            - Numerical columns for analysis
            
            ### Common Issues
            1. Missing values in numerical columns
            2. Incorrect date format
            3. Non-numeric data in numeric columns
            """)
        st.stop()

else:
    # Enhanced welcome message
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h2>👋 Welcome to the Mental Health Analysis Dashboard!</h2>
        <p style='font-size: 1.2rem; color: #666;'>
            Upload your CSV file to start analyzing your data
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample data format with better styling
    with st.expander("📋 Sample Data Format"):
        st.markdown("""
        Your CSV file should contain:
        - **Date**: Timestamp of the measurement
        - **Anxiety Level**: Numerical score (e.g., 0-10)
        - **Depression Level**: Numerical score (e.g., 0-10)
        - **Additional Metrics**: Any other relevant measurements
        
        Example:
        ```csv
        Date,Anxiety,Depression,Sleep_Hours
        2024-01-01,5,4,7.5
        2024-01-02,6,5,6.8
        ```
        """)
