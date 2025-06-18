import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import plotly.colors
from scipy import stats
import warnings
import json

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'bookmarked_views' not in st.session_state:
    st.session_state.bookmarked_views = []
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'default_theme': 'plotly_white',
        'default_color_scale': 'Viridis',
        'show_tips': True
    }

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
        
        **Version:** 2.0.0
        **GitHub:** https://github.com/yourusername/mental-health-dashboard
        
        Made with ❤️ for mental health awareness
        """
    }
)

# Sidebar configuration
st.sidebar.markdown("### 🛠️ Analysis Settings")

# Dark/Light Mode Toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

# Dynamic CSS based on mode
if dark_mode:
    css_theme = """
    <style>
    .stApp { background-color: #1e1e1e; color: #ffffff; }
    .main { padding: 1rem 2rem; background-color: #1e1e1e; }
    .stTitle { color: #ffffff; font-size: 2.5rem; text-align: center; margin-bottom: 2rem; }
    .stAlert { padding: 1rem; margin-bottom: 1rem; border-radius: 0.5rem; background-color: #2d2d2d; }
    .element-container { background-color: #2d2d2d; padding: 1.5rem; border-radius: 0.8rem; 
                        box-shadow: 0 2px 4px rgba(255,255,255,0.1); margin-bottom: 1rem; }
    .stButton>button { background-color: #4a90e2; color: white; border-radius: 0.3rem;
                      padding: 0.5rem 1rem; transition: all 0.3s ease; border: none; }
    .stButton>button:hover { background-color: #357abd; transform: translateY(-2px); }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    .stTabs [data-baseweb="tab"] { padding: 0.5rem 1.5rem; }
    .plot-container { background-color: #2d2d2d; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    
    /* Mobile Responsive */
    @media (max-width: 768px) {
        .main { padding: 0.5rem 1rem; }
        .stTitle { font-size: 1.8rem; }
        .element-container { padding: 1rem; margin-bottom: 0.5rem; }
        .stTabs [data-baseweb="tab"] { padding: 0.5rem 1rem; font-size: 0.9rem; }
        .stColumns > div { margin-bottom: 1rem; }
    }
    
    /* Better mobile charts */
    @media (max-width: 480px) {
        .js-plotly-plot .plotly { min-height: 300px !important; }
        .stDataFrame { font-size: 0.8rem; }
        .metric-card { padding: 0.8rem; }
    }
    </style>
    """
else:
    css_theme = """
    <style>
    .stApp { background-color: #ffffff; color: #262730; }
    .main { padding: 1rem 2rem; background-color: #ffffff; }
    .stTitle { color: #2c3e50; font-size: 2.5rem; text-align: center; margin-bottom: 2rem; }
    .stAlert { padding: 1rem; margin-bottom: 1rem; border-radius: 0.5rem; }
    .element-container { background-color: #ffffff; padding: 1.5rem; border-radius: 0.8rem; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem; }
    .stButton>button { background-color: #2c3e50; color: white; border-radius: 0.3rem;
                      padding: 0.5rem 1rem; transition: all 0.3s ease; border: none; }
    .stButton>button:hover { background-color: #34495e; transform: translateY(-2px); }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    .stTabs [data-baseweb="tab"] { padding: 0.5rem 1.5rem; }
    .plot-container { background-color: #f8f9fa; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; color: white; }
    
    /* Mobile Responsive */
    @media (max-width: 768px) {
        .main { padding: 0.5rem 1rem; }
        .stTitle { font-size: 1.8rem; }
        .element-container { padding: 1rem; margin-bottom: 0.5rem; }
        .stTabs [data-baseweb="tab"] { padding: 0.5rem 1rem; font-size: 0.9rem; }
        .stColumns > div { margin-bottom: 1rem; }
    }
    
    /* Better mobile charts */
    @media (max-width: 480px) {
        .js-plotly-plot .plotly { min-height: 300px !important; }
        .stDataFrame { font-size: 0.8rem; }
        .metric-card { padding: 0.8rem; }
    }
    </style>
    """

st.markdown(css_theme, unsafe_allow_html=True)

# Enhanced theme and color settings
st.sidebar.markdown("### 🎨 Visualization Settings")

# Adapt theme and color settings based on dark mode
if dark_mode:
    plot_theme = st.sidebar.selectbox("📊 Plot Theme", ["plotly_dark", "plotly", "plotly_white"], index=0)
    default_color = "Plasma"
else:
    plot_theme = st.sidebar.selectbox("📊 Plot Theme", ["plotly_white", "plotly", "plotly_dark"], index=0)
    default_color = "Viridis"

color_scale = st.sidebar.selectbox("🌈 Color Scale", 
                                  ["Viridis", "Plasma", "Inferno", "Magma", "RdBu", "Blues", "Greens"], 
                                  index=["Viridis", "Plasma", "Inferno", "Magma", "RdBu", "Blues", "Greens"].index(default_color))

# Animation settings
show_animations = st.sidebar.toggle("✨ Show Animations", value=True)

# Function to apply consistent theming to plots
def apply_plot_theme(fig, plot_theme, dark_mode):
    """Apply consistent theming to plotly figures"""
    fig.update_layout(
        template=plot_theme,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="white" if dark_mode else "black"
        ),
        title_font_size=16,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified' if show_animations else 'x'
    )
    
    # Add animation settings if enabled
    if show_animations and hasattr(fig, 'update_traces'):
        fig.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>" +
                         "%{xaxis.title.text}: %{x}<br>" +
                         "%{yaxis.title.text}: %{y}<br>" +
                         "<extra></extra>"
        )
    
    return fig

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

# Enhanced info banner
if st.session_state.user_preferences['show_tips']:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.info("👆 **New to the app?** Try uploading the sample data file (`sample_data.csv`) included in this repository to see the dashboard in action!", icon="💡")
    with col2:
        if st.button("❌ Hide Tips"):
            st.session_state.user_preferences['show_tips'] = False
            st.rerun()

# File uploader with enhanced styling
st.markdown("### 📁 Data Upload")
uploaded_file = st.file_uploader(
    'Choose a CSV file', 
    type='csv', 
    help='Upload a CSV file containing your mental health data'
)

if uploaded_file is not None:
    try:
        # Load and process data with progress indicator
        with st.spinner('🔄 Loading and processing data... Please wait.'):
            progress_bar = st.progress(0)
            df = pd.read_csv(uploaded_file)
            progress_bar.progress(100)
        
        # Enhanced success message
        st.success(f"✅ Successfully loaded {len(df):,} rows and {len(df.columns)} columns!")
        
        # Create enhanced tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Overview", 
            "📊 Statistics", 
            "🔍 Analysis", 
            "📈 Advanced",
            "⚙️ Settings"
        ])
        
        # Tab 1: Enhanced Overview
        with tab1:
            st.markdown("### 📊 Dataset Overview")
            
            # Key metrics cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Rows", f"{df.shape[0]:,}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Columns", f"{df.shape[1]}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                st.metric("Data Quality", f"{100-missing_pct:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                st.metric("Numeric Columns", f"{numeric_cols}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Data quality check with enhanced styling
            with st.expander("🔍 Data Quality Report", expanded=True):
                quality_issues = []
                missing = df.isnull().sum()
                if missing.any():
                    quality_issues.append(f"Missing values found in columns: {missing[missing > 0].index.tolist()}")
                duplicates = df.duplicated().sum()
                if duplicates > 0:
                    quality_issues.append(f"Found {duplicates} duplicate rows")
                
                if quality_issues:
                    for issue in quality_issues:
                        st.warning(f"⚠️ {issue}")
                else:
                    st.success("✅ No major data quality issues found!")
            
            # Enhanced data preview
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("#### 📋 Sample Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
            with col2:
                st.markdown("#### 📝 Column Information")
                column_info = []
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    null_count = df[col].isnull().sum()
                    column_info.append({
                        'Column': col,
                        'Type': dtype,
                        'Missing': null_count
                    })
                st.dataframe(pd.DataFrame(column_info), use_container_width=True)
            
            # Enhanced missing values visualization
            st.markdown("#### 🔍 Missing Values Visualization")
            if df.isnull().sum().sum() > 0:
                fig = px.imshow(df.isnull(), 
                              aspect="auto",
                              color_continuous_scale=color_scale,
                              title="Missing Values Heatmap")
                fig = apply_plot_theme(fig, plot_theme, dark_mode)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("🎉 No missing values detected in your dataset!")
            
            # Enhanced automated insights
            st.markdown("### 🤖 Automated Insights")
            
            def generate_enhanced_insights(df):
                insights = []
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    # Dataset insights
                    insights.append(f"📊 **Dataset Size**: {len(df):,} rows with {len(df.columns)} columns")
                    
                    # Missing data insights
                    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                    if missing_pct > 0:
                        insights.append(f"⚠️ **Data Quality**: {missing_pct:.1f}% of data points are missing")
                    else:
                        insights.append(f"✅ **Data Quality**: No missing values detected")
                    
                    # Enhanced numeric column insights
                    for col in numeric_cols:
                        if any(keyword in col.lower() for keyword in ['anxiety', 'depression', 'stress', 'mood']):
                            mean_val = df[col].mean()
                            std_val = df[col].std()
                            median_val = df[col].median()
                            
                            # Range-based insights
                            if mean_val > 7:
                                insights.append(f"🔴 **{col}**: High average level ({mean_val:.1f}) - may need attention")
                            elif mean_val < 3:
                                insights.append(f"🟢 **{col}**: Low average level ({mean_val:.1f}) - positive indicator")
                            else:
                                insights.append(f"🟡 **{col}**: Moderate average level ({mean_val:.1f})")
                            
                            # Variability insights
                            if std_val > 2:
                                insights.append(f"📈 **{col} Variability**: High variation ({std_val:.1f}) - patterns may be inconsistent")
                            
                            # Skewness insights
                            skewness = df[col].skew()
                            if abs(skewness) > 1:
                                direction = "right" if skewness > 0 else "left"
                                insights.append(f"📊 **{col} Distribution**: Highly skewed {direction} ({skewness:.2f})")
                    
                    # Enhanced correlation insights
                    if len(numeric_cols) > 1:
                        corr_matrix = df[numeric_cols].corr()
                        high_corrs = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                corr_val = corr_matrix.iloc[i, j]
                                if abs(corr_val) > 0.7:
                                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                                    if corr_val > 0:
                                        insights.append(f"🔗 **Strong Positive Correlation**: {col1} and {col2} ({corr_val:.2f})")
                                    else:
                                        insights.append(f"🔀 **Strong Negative Correlation**: {col1} and {col2} ({corr_val:.2f})")
                        
                        # Most/least correlated pairs
                        if len(high_corrs) == 0:
                            # Find the highest correlation if no strong ones exist
                            max_corr = 0
                            max_pair = None
                            for i in range(len(corr_matrix.columns)):
                                for j in range(i+1, len(corr_matrix.columns)):
                                    corr_val = abs(corr_matrix.iloc[i, j])
                                    if corr_val > max_corr:
                                        max_corr = corr_val
                                        max_pair = (corr_matrix.columns[i], corr_matrix.columns[j])
                            
                            if max_pair:
                                insights.append(f"🔗 **Highest Correlation**: {max_pair[0]} and {max_pair[1]} ({max_corr:.2f})")
                    
                    # Temporal insights (enhanced)
                    date_cols = df.select_dtypes(include=['datetime64', 'object']).columns
                    for date_col in date_cols:
                        try:
                            df_temp = df.copy()
                            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
                            if not df_temp[date_col].isna().all():
                                date_range = (df_temp[date_col].max() - df_temp[date_col].min()).days
                                insights.append(f"📅 **Time Period**: Data spans {date_range} days")
                                
                                # Frequency insights
                                if date_range > 365:
                                    insights.append(f"📊 **Long-term Data**: Over {date_range//365} years of data available for trend analysis")
                                elif date_range > 30:
                                    insights.append(f"📊 **Medium-term Data**: {date_range//30} months of data available")
                                break
                        except:
                            continue
                
                return insights
            
            # Generate and display enhanced insights
            with st.spinner("🧠 Generating enhanced insights..."):
                insights = generate_enhanced_insights(df)
                
            if insights:
                # Display insights in columns for better layout
                insight_cols = st.columns(2)
                for i, insight in enumerate(insights):
                    with insight_cols[i % 2]:
                        st.markdown(f"• {insight}")
            else:
                st.info("Upload data to see automated insights!")
            
            # Enhanced download options
            st.markdown("### 📥 Download Options")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "📊 Download Sample Data",
                    df.head(100).to_csv(index=False).encode('utf-8'),
                    "sample_data.csv",
                    "text/csv",
                    key='download-sample'
                )
            with col2:
                st.download_button(
                    "📑 Download Insights Report",
                    "\n".join([insight.replace("*", "").replace("•", "-") for insight in insights]),
                    "insights_report.txt",
                    key='download-insights'
                )
            with col3:
                if st.button("🔖 Bookmark This View"):
                    st.session_state.bookmarked_views.append({
                        'name': f"Overview - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        'type': 'overview',
                        'data': df.head().to_dict()
                    })
                    st.success("✅ View bookmarked!")

        # Tab 2: Enhanced Statistics
        with tab2:
            st.markdown("### 📊 Statistical Analysis")
            
            # Enhanced basic statistics
            st.markdown("#### 📈 Descriptive Statistics")
            if len(df.select_dtypes(include=[np.number]).columns) > 0:
                stats_df = df.describe().round(2)
                
                # Add additional statistics
                numeric_df = df.select_dtypes(include=[np.number])
                additional_stats = pd.DataFrame({
                    'skewness': numeric_df.skew(),
                    'kurtosis': numeric_df.kurtosis(),
                    'variance': numeric_df.var()
                }).round(2)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Basic Statistics:**")
                    st.dataframe(stats_df, use_container_width=True)
                with col2:
                    st.markdown("**Distribution Metrics:**")
                    st.dataframe(additional_stats, use_container_width=True)
            
            # Enhanced correlation analysis
            if len(df.select_dtypes(include=[np.number]).columns) > 0:
                st.markdown("#### 🔗 Correlation Analysis")
                
                # Correlation threshold slider
                corr_threshold = st.slider(
                    "Correlation Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="Show correlations above this threshold"
                )
                
                corr_matrix = df.select_dtypes(include=[np.number]).corr()
                
                # Enhanced correlation heatmap
                fig = px.imshow(corr_matrix, 
                              color_continuous_scale=color_scale,
                              title='Correlation Matrix',
                              text_auto=True)
                fig = apply_plot_theme(fig, plot_theme, dark_mode)
                st.plotly_chart(fig, use_container_width=True)
                
                # Significant correlations table
                st.markdown("#### 📋 Significant Correlations")
                significant_corr_data = []
                for col1 in corr_matrix.columns:
                    for col2 in corr_matrix.columns:
                        if col1 != col2:
                            corr = corr_matrix.loc[col1, col2]
                            if abs(corr) > corr_threshold:
                                significant_corr_data.append({
                                    'Variable 1': col1,
                                    'Variable 2': col2,
                                    'Correlation': corr,
                                    'Strength': 'Strong' if abs(corr) > 0.7 else 'Moderate',
                                    'Direction': 'Positive' if corr > 0 else 'Negative'
                                })
                
                if significant_corr_data:
                    corr_df = pd.DataFrame(significant_corr_data)
                    corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
                    st.dataframe(corr_df, use_container_width=True)
                else:
                    st.info("No correlations found above the selected threshold.")

        # Tab 3: Enhanced Analysis
        with tab3:
            st.markdown("### 🔍 Data Analysis")
            
            # Enhanced analysis options
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Time Series", "Distribution", "Comparison", "Trend Analysis"]
            )
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            date_cols = df.select_dtypes(include=['datetime64', 'object']).columns.tolist()
            
            if analysis_type == "Time Series" and len(date_cols) > 0 and len(numeric_cols) > 0:
                st.markdown("#### 📈 Time Series Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    date_col = st.selectbox("Select Date Column", date_cols)
                with col2:
                    metric_col = st.selectbox("Select Metric", numeric_cols)
                
                try:
                    # Convert to datetime
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    
                    if df[date_col].isna().all():
                        st.warning(f"Could not convert column '{date_col}' to datetime format")
                    else:
                        # Remove rows with invalid dates
                        df_clean = df.dropna(subset=[date_col])
                        
                        # Enhanced time series plot
                        fig = px.line(df_clean, x=date_col, y=metric_col,
                                    title=f'{metric_col} Over Time',
                                    color_discrete_sequence=px.colors.qualitative.Set2)
                        fig = apply_plot_theme(fig, plot_theme, dark_mode)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Enhanced trend analysis
                        col1, col2 = st.columns(2)
                        with col1:
                            ma_window = st.select_slider(
                                "Moving Average Window",
                                options=[3, 7, 14, 30, 60],
                                value=7,
                                help="Select window size for moving average"
                            )
                        with col2:
                            show_trend = st.checkbox("Show Trend Line", value=True)
                        
                        # Calculate moving averages and trends
                        df_temp = df_clean.copy()
                        df_temp[f'MA{ma_window}'] = df_temp[metric_col].rolling(window=ma_window).mean()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_temp[date_col], 
                            y=df_temp[metric_col],
                            name='Raw Data', 
                            mode='lines',
                            line=dict(color='lightblue', width=1)
                        ))
                        fig.add_trace(go.Scatter(
                            x=df_temp[date_col], 
                            y=df_temp[f'MA{ma_window}'],
                            name=f'{ma_window}-day Moving Average', 
                            mode='lines',
                            line=dict(color='red', width=2)
                        ))
                        
                        if show_trend:
                            # Add trend line
                            x_numeric = pd.to_numeric(df_temp[date_col])
                            z = np.polyfit(x_numeric.dropna(), df_temp[metric_col].dropna(), 1)
                            trend_line = np.poly1d(z)(x_numeric)
                            fig.add_trace(go.Scatter(
                                x=df_temp[date_col],
                                y=trend_line,
                                name='Trend Line',
                                mode='lines',
                                line=dict(color='green', width=2, dash='dash')
                            ))
                        
                        fig.update_layout(
                            title='Enhanced Trend Analysis',
                            xaxis_title='Date',
                            yaxis_title='Value'
                        )
                        fig = apply_plot_theme(fig, plot_theme, dark_mode)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistical summary
                        st.markdown("#### 📊 Trend Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Value", f"{df_temp[metric_col].mean():.2f}")
                        with col2:
                            st.metric("Trend Slope", f"{z[0]:.4f}" if show_trend else "N/A")
                        with col3:
                            volatility = df_temp[metric_col].std() / df_temp[metric_col].mean() * 100
                            st.metric("Volatility", f"{volatility:.1f}%")
                        
                except Exception as date_error:
                    st.error(f"Could not process date column: {str(date_error)}")
            
            elif analysis_type == "Distribution" and len(numeric_cols) > 0:
                st.markdown("#### 📊 Distribution Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    selected_col = st.selectbox("Select Column for Distribution", numeric_cols)
                with col2:
                    show_normal = st.checkbox("Show Normal Distribution Overlay", value=True)
                
                # Enhanced distribution plot
                fig = px.histogram(df, x=selected_col, marginal="box", 
                                 title=f"Distribution of {selected_col}",
                                 color_discrete_sequence=px.colors.qualitative.Set2,
                                 nbins=30)
                
                if show_normal:
                    # Add normal distribution overlay
                    mean_val = df[selected_col].mean()
                    std_val = df[selected_col].std()
                    x_norm = np.linspace(df[selected_col].min(), df[selected_col].max(), 100)
                    y_norm = stats.norm.pdf(x_norm, mean_val, std_val)
                    # Scale to match histogram
                    y_norm = y_norm * len(df) * (df[selected_col].max() - df[selected_col].min()) / 30
                    
                    fig.add_trace(go.Scatter(
                        x=x_norm, y=y_norm,
                        name='Normal Distribution',
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                
                fig = apply_plot_theme(fig, plot_theme, dark_mode)
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribution statistics
                st.markdown("#### 📈 Distribution Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{df[selected_col].mean():.2f}")
                with col2:
                    st.metric("Median", f"{df[selected_col].median():.2f}")
                with col3:
                    st.metric("Skewness", f"{df[selected_col].skew():.2f}")
                with col4:
                    st.metric("Kurtosis", f"{df[selected_col].kurtosis():.2f}")

        # Tab 4: Enhanced Advanced Analysis
        with tab4:
            st.markdown("### 📈 Advanced Analysis")
            
            # Enhanced PCA Analysis
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                st.markdown("#### 🔍 Principal Component Analysis (PCA)")
                
                # PCA settings
                col1, col2 = st.columns(2)
                with col1:
                    n_components = st.slider(
                        "Number of Components to Analyze",
                        min_value=2,
                        max_value=min(10, len(numeric_df.columns)),
                        value=min(3, len(numeric_df.columns)),
                        help="Select number of principal components"
                    )
                with col2:
                    scaling_method = st.selectbox(
                        "Scaling Method",
                        ["StandardScaler", "MinMaxScaler", "RobustScaler"],
                        help="Choose how to scale the data before PCA"
                    )
                
                # Handle missing values
                imputer = SimpleImputer(strategy='mean')
                imputed_features = imputer.fit_transform(numeric_df)
                
                # Scale features based on selection
                if scaling_method == "StandardScaler":
                    scaler = StandardScaler()
                elif scaling_method == "MinMaxScaler":
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                else:
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                
                scaled_features = scaler.fit_transform(imputed_features)
                
                # Perform PCA
                pca = PCA(n_components=n_components)
                pca_features = pca.fit_transform(scaled_features)
                
                # Enhanced explained variance plot
                fig = px.line(
                    y=np.cumsum(pca.explained_variance_ratio_),
                    title='Cumulative Explained Variance Ratio',
                    labels={'index': 'Number of Components', 'y': 'Cumulative Explained Variance Ratio'},
                    markers=True
                )
                fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                             annotation_text="80% Variance Explained")
                fig = apply_plot_theme(fig, plot_theme, dark_mode)
                st.plotly_chart(fig, use_container_width=True)
                
                # Enhanced PCA scatter plot
                if pca_features.shape[1] >= 2:
                    fig = px.scatter(
                        x=pca_features[:, 0], 
                        y=pca_features[:, 1],
                        title='First Two Principal Components',
                        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                               'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'},
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig = apply_plot_theme(fig, plot_theme, dark_mode)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Enhanced component analysis
                    st.markdown("#### 🔬 Component Analysis")
                    
                    # Component loadings
                    loadings = pd.DataFrame(
                        pca.components_[:2].T,
                        columns=['PC1', 'PC2'],
                        index=numeric_df.columns
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Component Loadings:**")
                        st.dataframe(loadings.round(3), use_container_width=True)
                    
                    with col2:
                        # Biplot-style visualization
                        fig = go.Figure()
                        
                        # Add data points
                        fig.add_trace(go.Scatter(
                            x=pca_features[:, 0],
                            y=pca_features[:, 1],
                            mode='markers',
                            name='Data Points',
                            marker=dict(size=8, opacity=0.6)
                        ))
                        
                        # Add loading vectors
                        for i, feature in enumerate(numeric_df.columns):
                            fig.add_trace(go.Scatter(
                                x=[0, loadings.iloc[i, 0] * 3],
                                y=[0, loadings.iloc[i, 1] * 3],
                                mode='lines+text',
                                name=feature,
                                text=['', feature],
                                textposition='top center',
                                line=dict(color='red', width=2),
                                showlegend=False
                            ))
                        
                        fig.update_layout(
                            title='PCA Biplot',
                            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
                        )
                        fig = apply_plot_theme(fig, plot_theme, dark_mode)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Detailed variance table
                st.markdown("#### 📊 Explained Variance Details")
                explained_variance = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                    'Explained Variance Ratio': pca.explained_variance_ratio_,
                    'Cumulative Explained Variance': np.cumsum(pca.explained_variance_ratio_),
                    'Eigenvalue': pca.explained_variance_
                })
                st.dataframe(explained_variance.round(4), use_container_width=True)
            
            # Enhanced export options
            st.markdown("### 📤 Enhanced Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    "📊 Export Full Dataset",
                    df.to_csv(index=False).encode('utf-8'),
                    "full_dataset.csv",
                    "text/csv",
                    key='download-full'
                )
            
            with col2:
                # Enhanced analysis report
                report_content = f"""Mental Health Data Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW:
- Total Rows: {df.shape[0]:,}
- Total Columns: {df.shape[1]}
- Missing Values: {df.isnull().sum().sum():,}
- Data Quality: {100 - (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%

STATISTICAL SUMMARY:
{df.describe().to_string()}

CORRELATION ANALYSIS:
{df.select_dtypes(include=[np.number]).corr().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else 'No numeric columns found'}
"""
                st.download_button(
                    "📑 Export Detailed Report",
                    report_content,
                    "detailed_analysis_report.txt",
                    key='download-detailed-report'
                )
            
            with col3:
                if 'pca' in locals():
                    pca_report = f"""PCA Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PCA RESULTS:
- Components Analyzed: {n_components}
- Scaling Method: {scaling_method}
- Total Variance Explained: {np.sum(pca.explained_variance_ratio_):.1%}

COMPONENT DETAILS:
{explained_variance.to_string(index=False)}
"""
                    st.download_button(
                        "🔬 Export PCA Report",
                        pca_report,
                        "pca_analysis_report.txt",
                        key='download-pca-report'
                    )

        # Tab 5: Settings and Preferences
        with tab5:
            st.markdown("### ⚙️ Settings & Preferences")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎨 Display Preferences")
                
                # Theme preferences
                st.session_state.user_preferences['default_theme'] = st.selectbox(
                    "Default Plot Theme",
                    ["plotly_white", "plotly", "plotly_dark"],
                    index=["plotly_white", "plotly", "plotly_dark"].index(
                        st.session_state.user_preferences['default_theme']
                    )
                )
                
                st.session_state.user_preferences['default_color_scale'] = st.selectbox(
                    "Default Color Scale",
                    ["Viridis", "Plasma", "Inferno", "Magma", "RdBu", "Blues", "Greens"],
                    index=["Viridis", "Plasma", "Inferno", "Magma", "RdBu", "Blues", "Greens"].index(
                        st.session_state.user_preferences['default_color_scale']
                    )
                )
                
                st.session_state.user_preferences['show_tips'] = st.checkbox(
                    "Show Tips and Hints",
                    value=st.session_state.user_preferences['show_tips']
                )
                
            with col2:
                st.markdown("#### 🔖 Bookmarked Views")
                
                if st.session_state.bookmarked_views:
                    for i, bookmark in enumerate(st.session_state.bookmarked_views):
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"📌 {bookmark['name']}")
                        with col_b:
                            if st.button("🗑️", key=f"delete_bookmark_{i}"):
                                st.session_state.bookmarked_views.pop(i)
                                st.rerun()
                    
                    if st.button("🗑️ Clear All Bookmarks"):
                        st.session_state.bookmarked_views = []
                        st.rerun()
                else:
                    st.info("No bookmarked views yet. Use the bookmark button in other tabs!")
            
            # Analysis history
            st.markdown("#### 📊 Analysis History")
            if st.session_state.analysis_history:
                history_df = pd.DataFrame(st.session_state.analysis_history)
                st.dataframe(history_df, use_container_width=True)
                
                if st.button("🗑️ Clear History"):
                    st.session_state.analysis_history = []
                    st.rerun()
            else:
                st.info("No analysis history yet. Your analysis actions will appear here.")
            
            # Export settings
            st.markdown("#### 💾 Export Settings")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Export User Preferences"):
                    preferences_json = json.dumps(st.session_state.user_preferences, indent=2)
                    st.download_button(
                        "Download Preferences",
                        preferences_json,
                        "user_preferences.json",
                        "application/json"
                    )
            
            with col2:
                uploaded_prefs = st.file_uploader("📤 Import Preferences", type="json")
                if uploaded_prefs:
                    try:
                        prefs = json.load(uploaded_prefs)
                        st.session_state.user_preferences.update(prefs)
                        st.success("✅ Preferences imported successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error importing preferences: {e}")

    except Exception as e:
        st.error(f"""
        ### 😕 Oops! Something went wrong
        
        **Error Details:** {str(e)}
        
        **Possible Solutions:**
        - Check if your data format is correct
        - Ensure all required columns are present
        - Make sure numerical columns contain valid numbers
        - Try uploading a smaller file first
        
        Need help? Check the documentation below 👇
        """)
        
        with st.expander("📚 Documentation & Troubleshooting"):
            st.markdown("""
            ### Required Data Format
            - CSV file with headers in the first row
            - At least one date column (YYYY-MM-DD format preferred)
            - Numerical columns for analysis
            - UTF-8 encoding recommended
            
            ### Common Issues & Solutions
            1. **Missing values in numerical columns**
               - The app will handle missing values automatically
               - Consider cleaning your data before upload for best results
            
            2. **Incorrect date format**
               - Use YYYY-MM-DD format (e.g., 2024-01-15)
               - Ensure date column doesn't contain text values
            
            3. **Non-numeric data in numeric columns**
               - Remove text values from columns meant to be numerical
               - Use consistent decimal separators (dots, not commas)
            
            4. **File size too large**
               - Try uploading a smaller sample first
               - Consider splitting large files into smaller chunks
               
            5. **Encoding issues**
               - Save your CSV file with UTF-8 encoding
               - Avoid special characters in column names
            
            ### Sample Data Structure
            ```
            Date,Anxiety_Level,Depression_Level,Sleep_Hours,Mood_Rating
            2024-01-01,5,4,7.5,6
            2024-01-02,6,5,6.8,5
            2024-01-03,4,3,8.0,7
            ```
            """)
        st.stop()

else:
    # Enhanced welcome message
    st.markdown("""
    <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 1rem; margin: 2rem 0; color: white;'>
        <h1>👋 Welcome to the Mental Health Analysis Dashboard!</h1>
        <p style='font-size: 1.3rem; margin: 1rem 0;'>
            Upload your CSV file to start analyzing your mental health data
        </p>
        <p style='font-size: 1rem; opacity: 0.9;'>
            🚀 Enhanced with advanced analytics • 🎨 Beautiful visualizations • 📊 Automated insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced feature showcase
    st.markdown("### ✨ What's New in Version 2.0")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🌙 Enhanced UI/UX
        - **Dark/Light Mode Toggle**
        - **Mobile Responsive Design**
        - **Improved Navigation**
        - **Better Accessibility**
        """)
    
    with col2:
        st.markdown("""
        #### 🤖 Smart Analytics
        - **Automated Insights Generation**
        - **Enhanced Statistical Analysis**
        - **Advanced PCA Analysis**
        - **Trend Detection**
        """)
    
    with col3:
        st.markdown("""
        #### 🔧 Power User Features
        - **Customizable Themes**
        - **Bookmarked Views**
        - **Analysis History**
        - **Enhanced Export Options**
        """)
    
    # Sample data format with enhanced styling
    with st.expander("📋 Sample Data Format & Getting Started", expanded=True):
        st.markdown("""
        ### 🚀 Quick Start Guide
        
        1. **Prepare your data** in CSV format with these columns:
           - **Date**: Timestamp of the measurement (YYYY-MM-DD format)
           - **Mental Health Metrics**: Numerical scores (e.g., anxiety, depression levels)
           - **Additional Factors**: Sleep hours, mood ratings, stress levels, etc.
        
        2. **Upload your file** using the file uploader above
        
        3. **Explore the tabs**:
           - 📋 **Overview**: Get data quality insights and automated analysis
           - 📊 **Statistics**: Dive into statistical analysis and correlations
           - 🔍 **Analysis**: Perform time series and distribution analysis
           - 📈 **Advanced**: Use PCA and advanced analytics
           - ⚙️ **Settings**: Customize your experience
        
        ### 📊 Example Data Structure:
        ```csv
        Date,Anxiety_Level,Depression_Level,Sleep_Hours,Mood_Rating,Stress_Level
        2024-01-01,5,4,7.5,6,3
        2024-01-02,6,5,6.8,5,4
        2024-01-03,4,3,8.0,7,2
        2024-01-04,7,6,6.0,4,5
        2024-01-05,3,2,8.5,8,1
        ```
        
        ### 💡 Pro Tips:
        - **Use consistent scales** (e.g., 1-10 for all ratings)
        - **Include daily entries** for better trend analysis
        - **Add context columns** like weather, events, medications
        - **Keep column names simple** (no spaces or special characters work best)
        """)
    
    # Quick demo section
    st.markdown("### 🎯 Try the Demo")
    if st.button("🎮 Load Demo Data", type="primary"):
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
        np.random.seed(42)
        
        demo_data = pd.DataFrame({
            'Date': dates,
            'Anxiety_Level': np.random.normal(5, 2, len(dates)).clip(1, 10),
            'Depression_Level': np.random.normal(4, 1.5, len(dates)).clip(1, 10),
            'Sleep_Hours': np.random.normal(7.5, 1, len(dates)).clip(4, 12),
            'Mood_Rating': np.random.normal(6, 1.5, len(dates)).clip(1, 10),
            'Stress_Level': np.random.normal(4, 2, len(dates)).clip(1, 10)
        })
        
        # Add some correlations to make it more realistic
        demo_data['Depression_Level'] = demo_data['Depression_Level'] + 0.3 * demo_data['Anxiety_Level'] + np.random.normal(0, 0.5, len(dates))
        demo_data['Mood_Rating'] = 10 - 0.4 * demo_data['Anxiety_Level'] - 0.3 * demo_data['Depression_Level'] + np.random.normal(0, 0.8, len(dates))
        demo_data['Sleep_Hours'] = demo_data['Sleep_Hours'] - 0.1 * demo_data['Stress_Level'] + np.random.normal(0, 0.3, len(dates))
        
        # Clip values to reasonable ranges
        demo_data['Depression_Level'] = demo_data['Depression_Level'].clip(1, 10)
        demo_data['Mood_Rating'] = demo_data['Mood_Rating'].clip(1, 10)
        demo_data['Sleep_Hours'] = demo_data['Sleep_Hours'].clip(4, 12)
        
        # Round to reasonable precision
        demo_data = demo_data.round(1)
        
        st.success("🎉 Demo data loaded! The analysis will appear above.")
        st.rerun()
