"""
Configuration module for InsightPulse
Handles environment variables and settings
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
    
    # AutoGluon settings
    PREDICTION_LENGTH = int(os.getenv('PREDICTION_LENGTH', 30))
    TIME_LIMIT = int(os.getenv('TIME_LIMIT', 300))  # 5 minutes
    
    # Code execution settings
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
    TIMEOUT = int(os.getenv('TIMEOUT', 60))
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.GEMINI_API_KEY or cls.GEMINI_API_KEY == 'your_gemini_api_key_here':
            raise ValueError(
                "GEMINI_API_KEY not found or invalid. "
                "Please copy .env.example to .env and add your API key from "
                "https://aistudio.google.com/app/apikey"
            )
        return True
