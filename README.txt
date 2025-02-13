# Anxiety and Depression Analysis App

## Overview
This Streamlit application provides interactive analysis and visualization of anxiety and depression data. The app offers multiple analysis methods through an intuitive interface with four main tabs.

## Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd anxiety-depression-analysis

# Install required packages
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

## Features

### 1. Data Upload
- ✅ Accepts CSV files
- 📊 Displays upload progress
- 🔍 Validates data format
- ⚠️ Shows immediate feedback on data quality

### 2. Analysis Tabs

#### 📋 Overview Tab
- Dataset summary statistics
- Column information
- Data quality report
  - Missing values detection
  - Duplicate rows identification
  - Data preview
- Missing values visualization
- Sample data download option

#### 📊 Statistics Tab
- Numerical statistics
- Correlation analysis
  - Interactive correlation matrix
  - Significance testing
  - Adjustable correlation threshold
- Basic statistical summaries
- Downloadable statistical reports

#### 🔍 Analysis Tab
- Time Series Analysis
  - Date-based trending
  - Moving averages (7-day and 30-day)
  - Customizable window sizes
- Distribution Analysis
- Correlation Studies
- Custom Analysis Options

#### 📈 Advanced Tab
- Principal Component Analysis (PCA)
  - Variance explanation
  - Component visualization
  - Dimensionality reduction
- Export capabilities
  - Full dataset export
  - Analysis report generation

### 3. Customization Options
- Theme selection
  - Light/Dark modes
  - Plot themes
- Color scale selection
  - Viridis
  - Plasma
  - Inferno
  - Magma
  - RdBu

## Data Requirements

Your CSV file should include:
```csv
Date,Anxiety,Depression,Additional_Metrics
2024-01-01,5,4,7.5
2024-01-02,6,5,6.8
```

Required columns:
- Date column (any format convertible to datetime)
- Numerical columns for anxiety/depression scores
- Optional additional metrics

## Technical Requirements

### Dependencies
```txt
streamlit>=1.24.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.13.0
seaborn>=0.12.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
scipy>=1.10.0
```

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum
- Modern web browser

## Usage Guide

### Basic Analysis
1. Launch the app using `streamlit run streamlit_app.py`
2. Upload your CSV file using the file uploader
3. Review the data quality report
4. Navigate through the tabs for different analyses

### Time Series Analysis
1. Select the date column
2. Choose metrics to analyze
3. Adjust moving average windows
4. Export results if needed

### Correlation Analysis
1. Navigate to Statistics tab
2. Review the correlation matrix
3. Adjust significance thresholds
4. Download statistical reports

### Advanced Features
1. Use PCA for dimensionality reduction
2. Customize visualizations
3. Export detailed analysis reports

## Error Handling

The app includes comprehensive error handling for:
- ⚠️ Missing values
- ❌ Invalid data formats
- 📅 Date conversion issues
- 🔢 Non-numeric data in numeric columns

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Support

For support:
- Create an issue in the repository
- Contact: [Your Contact Information]
- Documentation: [Link to Documentation]

## Acknowledgments

- Built with Streamlit
- Powered by Python
- Data analysis tools: Pandas, NumPy, Scikit-learn
- Visualization: Plotly, Seaborn

## Version History

- 1.0.0
  - Initial Release
  - Basic features implemented
  - Four main analysis tabs

---
Created with ❤️ for mental health analysis
```

This `README.md` file provides comprehensive documentation for your project. To use it:

1. Save it as `README.md` in your project's root directory
2. Update the placeholder content (repository URL, contact information, etc.)
3. Add any specific instructions or requirements for your implementation

The Markdown format will render nicely on GitHub or any other platform that supports Markdown.
This `README.md` file provides comprehensive documentation for your project. To use it:

1. Save it as `README.md` in your project's root directory
2. Update the placeholder content (repository URL, contact information, etc.)
3. Add any specific instructions or requirements for your implementation
