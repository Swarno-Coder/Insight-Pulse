# 🎯 InsightPulse - Automated Data Storytelling & Forecasting Agent

An intelligent agent that reads any uploaded CSV file, performs full exploratory data analysis (EDA), builds predictive models, and narrates insights in human-readable language.

## 🌐 Streamlit Cloud Deployment

### Quick Deploy to Streamlit Cloud

1. **Fork/Clone this repository** to your GitHub account

2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**

3. **Click "New app"** and select this repository

4. **Configure Secrets** (Important!):
   - Click on "Advanced settings" → "Secrets"
   - Add your Gemini API key:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key_here"
   GEMINI_MODEL = "gemini-2.5-flash"
   ```
   - Get your API key from: [Google AI Studio](https://aistudio.google.com/app/apikey)

5. **Deploy!** - The app will automatically build and launch

### Configuration Files for Cloud

The repository includes:
- `.python-version` - Forces Python 3.11 (required for dependency compatibility)
- `.streamlit/config.toml` - Streamlit configuration with theme settings
- `packages.txt` - System dependencies
- `.streamlit/secrets.toml.example` - Example secrets file

**Note**: Never commit your actual API key to the repository! Always use Streamlit Cloud's Secrets management.

## 📋 Overview

InsightPulse is designed for data scientists who need quick, automated insights from their datasets. Simply upload a CSV file and get:

- **Automated EDA** with interactive visualizations
- **AI-Powered Forecasting** using AutoGluon
- **Natural Language Insights** via Google Gemini AI
- **Downloadable Reports** for sharing results

## 🚀 Features

### 1. Upload → Analyze → Storytell Pipeline
- Drag-and-drop CSV upload
- Automatic data type detection
- Missing value analysis
- Statistical summaries

### 2. Exploratory Data Analysis (EDA)
- Summary statistics for all numeric columns
- Data types and missing value percentages
- Correlation analysis with interactive heatmaps
- Distribution plots and trend analysis
- Time-series visualizations (when date column is available)

### 3. Predictive Modeling & Forecasting
- **AutoGluon Time-Series Forecasting**: Advanced ML models for 30-day predictions
- **Fallback Random Forest**: Simple regression when forecasting isn't applicable
- Feature importance analysis
- Growth/decline percentage calculations

### 4. LLM-Driven Narrative Insights
- AI-generated business insights using Google Gemini
- 3-5 key findings presented in natural language
- Actionable business recommendations
- Predictive summary with context

### 5. Export & Download
- Downloadable text reports with all key metrics
- Export processed data as CSV
- Timestamp for version tracking

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Interactive web interface)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, AutoGluon (Time-Series)
- **Visualization**: Plotly (Interactive charts)
- **AI Insights**: LangChain + Google Gemini API

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory**:
```bash
cd d:\codes\insight_pulse_datasc
```

2. **Install required packages**:
```bash
pip install -r requirements.txt
```

This will install:
- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- autogluon.timeseries >= 1.0.0
- plotly >= 5.17.0
- langchain >= 0.1.0
- langchain-google-genai >= 1.0.0
- langchain-core >= 0.1.0
- google-generativeai >= 0.3.0

## 🎮 Usage

### Running the Application

**Option 1: Quick Start (Recommended)**
```bash
# Windows PowerShell
.\run.ps1

# Windows Command Prompt
run.bat
```

**Option 2: Manual Start**
```bash
# PowerShell
$env:LOKY_MAX_CPU_COUNT="4"; python -m streamlit run app.py

# Command Prompt
set LOKY_MAX_CPU_COUNT=4 && python -m streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

> **Note**: The `LOKY_MAX_CPU_COUNT` environment variable suppresses a harmless CPU core detection warning on Windows.

### Using InsightPulse

1. **Upload CSV File**: Click "Browse files" and select your dataset
2. **Configure Settings** (Sidebar):
   - **Target Variable**: Select the numeric column you want to predict/analyze
   - **Date Column**: Choose the date/time column for trend analysis (optional)
3. **Review EDA**: Explore the automated visualizations and statistics
4. **View Predictions**: Check the forecast section for future predictions
5. **Get AI Insights**: Enter your Gemini API key to generate narrative insights
6. **Download Reports**: Export your analysis results

### Getting a Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and paste it in the InsightPulse interface

## 📊 Recommended Datasets

### 1. Global Superstore Sales Dataset
- Contains sales data with dates, regions, categories, and sales amounts
- Perfect for time-series forecasting and trend analysis

### 2. Retail Sales Forecasting
- Time-series data with historical sales
- Ideal for testing forecasting capabilities

### Sample CSV Structure
```csv
Date,Sales,Profit,Category,Region
2023-01-01,1500.00,300.00,Electronics,East
2023-01-02,2300.00,450.00,Furniture,West
...
```

## 🔧 Configuration

### Analysis Settings

- **Target Variable**: The numeric column to predict (required)
- **Date Column**: Time-series column for forecasting (optional but recommended)

### AutoGluon Settings

The forecasting model uses:
- **Prediction Length**: 30 days
- **Time Limit**: 5 minutes (300 seconds)
- **Evaluation Metric**: MASE (Mean Absolute Scaled Error)

You can modify these in `app.py`:
```python
predictor = TimeSeriesPredictor(
    prediction_length=30,  # Change forecast horizon
    target='target',
    eval_metric="MASE",
)
predictor.fit(ts_df, time_limit=300, random_seed=42)
```

## 📈 Output Examples

### Metrics Displayed
- Next month's forecast value
- Growth/decline percentage vs. last month
- Mean Absolute Error (MAE) for regression models
- Top feature importances

### Visualizations
- Line charts for trends over time
- Histograms for distribution analysis
- Correlation heatmaps
- Forecast plots with historical data

## 🐛 Troubleshooting

### Common Issues

**Issue**: "AutoGluon not available"
- **Solution**: Install AutoGluon: `pip install autogluon.timeseries`

**Issue**: "Not enough data points for forecasting"
- **Solution**: Ensure your dataset has at least 30 rows with valid dates

**Issue**: "LLM error: Check API key"
- **Solution**: Verify your Gemini API key is correct and active

**Issue**: "No numeric columns found"
- **Solution**: Ensure your CSV has at least one numeric column

### Error Handling

The application includes robust error handling:
- Invalid date formats are automatically cleaned
- Missing values are reported but don't crash the app
- Fallback models activate if AutoGluon fails
- Informative error messages guide troubleshooting

## 📝 Project Structure

```
insight_pulse_datasc/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎯 Timeline & Deliverables

**Development Time**: ~5 days

**Completed Deliverables**:
- ✅ "Upload → Analyze → Storytell" pipeline
- ✅ Text + chart-based narrative dashboard
- ✅ Predictive summary (e.g., "Next month's revenue expected to rise by 12%")
- ✅ Interactive Streamlit front-end
- ✅ AutoML forecasting with AutoGluon
- ✅ LLM integration with Gemini API
- ✅ Dynamic charts with Plotly
- ✅ Export functionality

## 🔮 Future Enhancements

Potential improvements:
- [ ] Support for multiple file formats (Excel, JSON)
- [ ] Custom model selection (Prophet, ARIMA, etc.)
- [ ] Advanced feature engineering options
- [ ] Automated hyperparameter tuning
- [ ] Multi-variate forecasting
- [ ] Anomaly detection
- [ ] Deployment on cloud platforms

## 📄 License

This project is open source and available for educational and commercial use.

## 👨‍💻 Developer Notes

### Code Quality
- All imports are properly handled with try-except blocks
- Comprehensive error messages for debugging
- Modular code structure for easy maintenance
- Type hints and documentation where needed

### Performance
- 5-minute time limit on model training for responsiveness
- Efficient data aggregation for time-series
- Cached operations where applicable (can add `@st.cache_data` for optimization)

## 🤝 Contributing

Feel free to fork this project and submit pull requests for:
- Bug fixes
- New features
- Documentation improvements
- Performance optimizations

---

**Built for Data Scientists** 📊 | **Powered by AI** 🤖 | **Made with Streamlit** ⚡
