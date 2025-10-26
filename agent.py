"""
Intelligent LLM Agent Module
Handles data analysis, code generation, and execution with AutoGluon
"""
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from config import Config
from code_executor import CodeExecutor
import json

class DataScienceAgent:
    """Intelligent agent for automated data science workflow"""
    
    def __init__(self):
        """Initialize the agent with LLM and code executor"""
        Config.validate()
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=Config.GEMINI_API_KEY,
            model=Config.GEMINI_MODEL,
            temperature=0.3
        )
        self.executor = None
        self.df = None
        self.conversation_history = []
        
    def load_data(self, df):
        """Load dataset into agent context"""
        self.df = df.copy()
        self.executor = CodeExecutor({
            'pd': pd,
            'np': __import__('numpy'),
            'plt': __import__('matplotlib.pyplot'),
            'px': __import__('plotly.express'),
            'go': __import__('plotly.graph_objects'),
            'df': self.df
        })
        
    def get_data_summary(self):
        """Generate comprehensive data summary"""
        if self.df is None:
            return "No data loaded"
        
        summary = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'missing': self.df.isnull().sum().to_dict(),
            'numeric_cols': list(self.df.select_dtypes(include=['number']).columns),
            'categorical_cols': list(self.df.select_dtypes(include=['object', 'category']).columns),
            'date_cols': [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()],
            'head': self.df.head().to_dict(),
            'describe': self.df.describe().to_dict() if len(self.df.select_dtypes(include=['number']).columns) > 0 else {}
        }
        return summary
    
    def analyze_and_generate_code(self, user_instruction):
        """
        Analyze user instruction and generate appropriate Python code
        
        Args:
            user_instruction: Natural language instruction from user
            
        Returns:
            Generated Python code
        """
        data_summary = self.get_data_summary()
        
        prompt = PromptTemplate(
            input_variables=["data_summary", "user_instruction", "history"],
            template="""You are an expert data scientist and Python programmer. You have access to a pandas DataFrame called 'df'.

DATA SUMMARY:
{data_summary}

CONVERSATION HISTORY:
{history}

USER INSTRUCTION:
{user_instruction}

IMPORTANT GUIDELINES:
1. Generate ONLY executable Python code, no explanations
2. The DataFrame is already loaded as 'df' - DO NOT reload it
3. Available imports: pandas as pd, numpy as np, matplotlib.pyplot as plt, plotly.express as px, plotly.graph_objects as go
4. For data cleaning: handle missing values, remove duplicates, fix data types
5. For feature engineering: create new features from existing ones as requested
6. For modeling: use AutoGluon (from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame) for time-series forecasting
7. For visualization: use plotly or matplotlib to create charts (they will be displayed automatically)
8. For analysis: generate summary statistics and insights
9. Store important results in variables like 'result', 'predictions', 'model_summary', etc.
10. Use print() to show important information
11. Handle errors gracefully with try-except blocks
12. For AutoGluon: aggregate data properly, ensure correct date format, minimum 30 rows required

Generate clean, production-ready Python code that accomplishes the user's request:
"""
        )
        
        history_text = "\n".join([
            f"User: {h['user']}\nAgent: {h['agent'][:200]}..." 
            for h in self.conversation_history[-3:]
        ]) if self.conversation_history else "No previous conversation"
        
        chain = prompt | self.llm
        response = chain.invoke({
            "data_summary": json.dumps(data_summary, indent=2, default=str),
            "user_instruction": user_instruction,
            "history": history_text
        })
        
        code = response.content if hasattr(response, 'content') else str(response)
        
        # Extract code from markdown if present
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0].strip()
        elif '```' in code:
            code = code.split('```')[1].split('```')[0].strip()
        
        return code
    
    def execute_with_retry(self, code, max_retries=3):
        """
        Execute code with automatic error fixing
        
        Args:
            code: Python code to execute
            max_retries: Maximum number of retry attempts
            
        Returns:
            Execution result dict
        """
        for attempt in range(max_retries):
            result = self.executor.execute(code)
            
            if result['success']:
                return result
            
            # Try to fix the error
            if attempt < max_retries - 1:
                fix_prompt = PromptTemplate(
                    input_variables=["code", "error"],
                    template="""The following Python code produced an error:

CODE:
{code}

ERROR:
{error}

Generate ONLY the FIXED Python code without any explanation. Fix the error while maintaining the original intent.
The DataFrame is available as 'df' and standard imports (pandas, numpy, matplotlib, plotly) are available.
"""
                )
                
                chain = fix_prompt | self.llm
                fix_response = chain.invoke({
                    "code": code,
                    "error": result['error']
                })
                
                code = fix_response.content if hasattr(fix_response, 'content') else str(fix_response)
                
                # Extract code from markdown if present
                if '```python' in code:
                    code = code.split('```python')[1].split('```')[0].strip()
                elif '```' in code:
                    code = code.split('```')[1].split('```')[0].strip()
        
        return result
    
    def generate_final_report(self, execution_results):
        """
        Generate human-readable final report from execution results
        
        Args:
            execution_results: List of execution result dicts
            
        Returns:
            Natural language report
        """
        context_vars = self.executor.context
        
        # Extract key metrics and results
        metrics = {}
        for key, value in context_vars.items():
            if not key.startswith('_') and key not in ['pd', 'np', 'plt', 'px', 'go', 'df']:
                if isinstance(value, (int, float, str, list, dict)):
                    metrics[key] = str(value)[:500]  # Limit length
        
        outputs = "\n\n".join([r['output'] for r in execution_results if r['output']])
        
        report_prompt = PromptTemplate(
            input_variables=["metrics", "outputs"],
            template="""You are a data storyteller. Based on the analysis results below, create a concise, engaging one-paragraph summary (3-5 sentences) highlighting the key insights and recommendations.

ANALYSIS METRICS:
{metrics}

EXECUTION OUTPUTS:
{outputs}

Generate a professional, business-focused summary that a stakeholder would understand:
"""
        )
        
        chain = report_prompt | self.llm
        response = chain.invoke({
            "metrics": json.dumps(metrics, indent=2),
            "outputs": outputs
        })
        
        report = response.content if hasattr(response, 'content') else str(response)
        return report
    
    def process_instruction(self, user_instruction):
        """
        Main method to process user instruction end-to-end
        
        Args:
            user_instruction: Natural language instruction
            
        Returns:
            dict with code, results, figures, and report
        """
        if self.df is None:
            return {
                'success': False,
                'error': 'No dataset loaded. Please upload a CSV file first.'
            }
        
        try:
            # Generate code
            code = self.analyze_and_generate_code(user_instruction)
            
            # Execute with retry
            result = self.execute_with_retry(code)
            
            # Generate report if successful
            report = ""
            if result['success']:
                report = self.generate_final_report([result])
            
            # Update conversation history
            self.conversation_history.append({
                'user': user_instruction,
                'agent': report or result.get('error', 'Execution failed')
            })
            
            return {
                'success': result['success'],
                'code': code,
                'output': result['output'],
                'error': result['error'],
                'figures': result['figures'],
                'report': report,
                'context': result['context']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Agent error: {str(e)}"
            }
