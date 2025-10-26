"""
InsightPulse - Intelligent Data Science Agent
Minimalist UI for automated data analysis with LLM-powered code generation
"""
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Suppress joblib warning

import streamlit as st
import pandas as pd
import plotly.express as px
from agent import DataScienceAgent
from config import Config
import traceback

# Page configuration
st.set_page_config(
    page_title="InsightPulse AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimalist design
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1557b0;
    }
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Header
st.markdown('<div class="main-header">🤖 InsightPulse AI Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload data, describe what you need, get results instantly</div>', unsafe_allow_html=True)

# Check API key
try:
    Config.validate()
    api_key_status = "✅ Gemini API configured"
except ValueError as e:
    api_key_status = f"⚠️ {str(e)}"
    st.error(api_key_status)
    st.info("Create a `.env` file with your `GEMINI_API_KEY`. See `.env.example` for template.")
    st.stop()

st.sidebar.success(api_key_status)

# File Upload Section
st.markdown("### 📁 Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader(
    "Drop your CSV file here",
    type=['csv'],
    help="Upload a CSV file to begin analysis"
)

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        # Initialize agent if not already done
        if st.session_state.agent is None:
            st.session_state.agent = DataScienceAgent()
        
        st.session_state.agent.load_data(df)
        st.session_state.data_loaded = True
        
        st.success(f"✅ Dataset loaded: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
        
        # Data Preview Section
        st.markdown("### 📊 Step 2: Review Your Data")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Preview", "📈 Summary Statistics", "🔍 Data Info", "📉 Missing Values"])
        
        with tab1:
            st.markdown("**First 10 rows:**")
            st.dataframe(df.head(10), use_container_width=True)
        
        with tab2:
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                st.markdown("**Statistical Summary:**")
                st.dataframe(numeric_df.describe(), use_container_width=True)
            else:
                st.info("No numeric columns found in the dataset")
        
        with tab3:
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(info_df, use_container_width=True)
        
        with tab4:
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
            }).sort_values('Missing Count', ascending=False)
            
            if missing_df['Missing Count'].sum() > 0:
                st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
                
                fig = px.bar(
                    missing_df[missing_df['Missing Count'] > 0],
                    x='Column',
                    y='Missing %',
                    title='Missing Values by Column',
                    labels={'Missing %': 'Missing Percentage (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values in the dataset!")
        
        # Agent Interaction Section
        st.markdown("### 🎯 Step 3: Tell the AI What You Need")
        st.markdown("""
        <div class="info-box">
        <b>What can the AI do?</b><br>
        • Clean and preprocess your data<br>
        • Create new features from existing columns<br>
        • Build forecasting models with AutoGluon<br>
        • Generate visualizations (plots, charts, heatmaps)<br>
        • Provide statistical analysis and insights<br>
        • Create final reports with recommendations
        </div>
        """, unsafe_allow_html=True)
        
        # Chat interface
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_area(
                "Your instruction:",
                placeholder="Example: Remove rows with missing values, create a 'total_sales' column by multiplying price and quantity, build a time-series forecast for the next 30 days using the 'order_date' column, and visualize the results with a line chart.",
                height=100,
                key="user_instruction"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            execute_button = st.button("🚀 Execute", use_container_width=True, type="primary")
        
        # Quick action buttons
        st.markdown("**Quick Actions:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🧹 Clean Data", use_container_width=True):
                user_input = "Remove rows with missing values, handle duplicates, and fix data types appropriately"
                execute_button = True
        
        with col2:
            if st.button("📊 EDA Visualization", use_container_width=True):
                user_input = "Generate comprehensive exploratory data analysis with correlation heatmap, distribution plots for all numeric columns, and key statistical insights"
                execute_button = True
        
        with col3:
            if st.button("🔮 Build Forecast Model", use_container_width=True):
                user_input = f"Build an AutoGluon time-series forecasting model for the next 30 days. Use the most appropriate date column and target variable. Show predictions with visualization and forecast summary"
                execute_button = True
        
        with col4:
            if st.button("📝 Generate Report", use_container_width=True):
                user_input = "Analyze all key metrics, trends, and patterns in the data. Generate a comprehensive business report with actionable insights and recommendations"
                execute_button = True
        
        # Execute instruction
        if execute_button and user_input:
            with st.spinner("🤖 AI Agent is working on your request..."):
                result = st.session_state.agent.process_instruction(user_input)
                
                # Store in chat history
                st.session_state.chat_history.append({
                    'instruction': user_input,
                    'result': result
                })
        
        # Display Results
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 📋 Results")
            
            # Show latest result first
            for idx, item in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"📌 {item['instruction'][:80]}...", expanded=(idx == 0)):
                    result = item['result']
                    
                    if result['success']:
                        # Final Report
                        if result.get('report'):
                            st.markdown("""
                            <div class="success-box">
                            <b>📊 Final Report:</b><br>
                            """ + result['report'] + """
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Generated Code
                        with st.expander("💻 View Generated Code"):
                            st.code(result['code'], language='python')
                        
                        # Execution Output
                        if result.get('output'):
                            st.markdown("**📤 Output:**")
                            st.text(result['output'])
                        
                        # Visualizations
                        if result.get('figures'):
                            st.markdown("**📈 Visualizations:**")
                            for fig in result['figures']:
                                st.pyplot(fig)
                        
                        st.success("✅ Execution completed successfully!")
                        
                    else:
                        st.markdown(f"""
                        <div class="error-box">
                        <b>❌ Error occurred:</b><br>
                        {result.get('error', 'Unknown error')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if result.get('code'):
                            with st.expander("💻 View Attempted Code"):
                                st.code(result['code'], language='python')
        
        # Download section
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download processed data
                csv = st.session_state.df.to_csv(index=False)
                st.download_button(
                    "📥 Download Current Dataset",
                    csv,
                    file_name="processed_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Download analysis report
                report_text = "\n\n".join([
                    f"INSTRUCTION: {item['instruction']}\n\nRESULT: {item['result'].get('report', 'N/A')}\n\nOUTPUT: {item['result'].get('output', 'N/A')}"
                    for item in st.session_state.chat_history
                ])
                
                st.download_button(
                    "📥 Download Analysis Report",
                    report_text,
                    file_name="analysis_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.code(traceback.format_exc())

else:
    # Welcome screen
    st.markdown("""
    <div class="info-box">
    <h3>👋 Welcome to InsightPulse AI Agent!</h3>
    <p>This intelligent system automates your entire data science workflow:</p>
    <ul>
        <li>📤 <b>Upload</b> your CSV file</li>
        <li>📊 <b>Review</b> data summary and statistics</li>
        <li>💬 <b>Describe</b> what you need in plain English</li>
        <li>🤖 <b>AI Agent</b> generates and executes code automatically</li>
        <li>✨ <b>Get</b> results, visualizations, and insights instantly</li>
    </ul>
    <p><b>No coding required!</b> Just upload your data and tell the AI what you want to achieve.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎬 Example Instructions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Cleaning:**
        - "Remove all rows with missing values"
        - "Fill missing values with column means"
        - "Convert date column to datetime format"
        """)
        
        st.markdown("""
        **Feature Engineering:**
        - "Create a 'profit' column by subtracting cost from revenue"
        - "Extract month and year from the date column"
        - "Create age groups from the age column"
        """)
    
    with col2:
        st.markdown("""
        **Modeling:**
        - "Build a forecasting model for the next 30 days"
        - "Predict sales using AutoGluon with all features"
        - "Show feature importance for the model"
        """)
        
        st.markdown("""
        **Visualization:**
        - "Create a correlation heatmap for all numeric columns"
        - "Plot sales trends over time with a line chart"
        - "Show distribution of all numeric variables"
        """)

# Sidebar info
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    **InsightPulse AI Agent**
    
    Version: 2.0
    
    Powered by:
    - Google Gemini AI
    - AutoGluon
    - Streamlit
    """)
    
    if st.session_state.data_loaded:
        st.markdown("---")
        st.markdown("### 📊 Current Dataset")
        st.metric("Rows", f"{st.session_state.df.shape[0]:,}")
        st.metric("Columns", st.session_state.df.shape[1])
        st.metric("Memory", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        if st.button("🔄 Clear & Reload", use_container_width=True):
            st.session_state.agent = None
            st.session_state.df = None
            st.session_state.chat_history = []
            st.session_state.data_loaded = False
            st.rerun()
