"""
===============================================================================
File Name: config.py
Project: AI-Enabled Real-Time Voice Transcription & Reflexive Question Generator
===============================================================================

Purpose of this File:
---------------------
This file is responsible for managing all application-level configuration
settings for the project.

It centralizes environment variables and reusable configuration values,
making the application:
- Cleaner
- Easier to maintain
- More secure
- Easier to deploy across environments

Why This File Exists:
---------------------
Instead of hardcoding sensitive values and application settings directly
inside business logic files, they are stored separately in:
- Environment variables
- .env file
- Config class

This improves:
--------------
1. Security
2. Scalability
3. Deployment flexibility
4. Environment management

Typical Configurations Stored Here:
-----------------------------------
- API keys
- Logging levels
- Timeout settings
- Model configurations
- Database URLs
- Server settings

Current Configurations:
-----------------------
1. LOG_LEVEL
   Controls application logging verbosity.

2. mic_timeout
   Defines microphone timeout duration for speech-related operations.

Important Notes:
----------------
- Environment variables are loaded using python-dotenv.
- Sensitive keys should NEVER be hardcoded.
- .env file should generally be added to .gitignore
  to avoid exposing secrets publicly on GitHub.

Example .env File:
------------------
LOG_LEVEL=INFO
OPENAI_API_KEY=your_api_key_here

===============================================================================
"""

# ------------------------------------------------------------------------------
# Import dotenv loader
#
# load_dotenv() loads environment variables from a .env file
# into the operating system environment.
#
# This allows secure management of sensitive credentials/configurations.
# ------------------------------------------------------------------------------
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# os module
#
# Used for accessing environment variables using os.getenv()
# ------------------------------------------------------------------------------
import os

# ------------------------------------------------------------------------------
# Load environment variables from .env file
# Existing comment retained intentionally.
#
# This reads all variables from .env and makes them accessible
# throughout the application.
#
# Example:
# --------
# LOG_LEVEL=INFO
# OPENAI_API_KEY=xyz
# ------------------------------------------------------------------------------
load_dotenv()

# ------------------------------------------------------------------------------
# Configuration Class
#
# Centralized configuration manager for the application.
#
# Why Use a Config Class?
# -----------------------
# 1. Keeps settings organized
# 2. Makes application scalable
# 3. Allows future extension easily
# 4. Prevents repeated environment variable fetching
#
# This class can later be extended with:
# - Database configs
# - HuggingFace model configs
# - API endpoints
# - Deployment settings
# - Authentication settings
# ------------------------------------------------------------------------------
class Config:

    """
    Manages all configuration for the AI MER Assistant application.

    Existing docstring retained intentionally.

    Note:
    -----
    "AI MER Assistant" appears to be the internal/project naming convention
    used during development.
    """

    def __init__(self):

        # ----------------------------------------------------------------------
        # --- REMOVED THE OPENAI KEY CHECK ---
        # Existing comment retained intentionally.
        #
        # This section previously validated whether OPENAI_API_KEY
        # existed inside environment variables.
        #
        # Why It May Have Been Removed:
        # -----------------------------
        # 1. Project may have shifted to HuggingFace models
        # 2. OpenAI integration may be optional
        # 3. Development/testing without API dependency
        # 4. Local/offline model support
        #
        # Previous Intended Logic:
        # ------------------------
        # - Read OpenAI API key
        # - Raise error if missing
        #
        # Keeping commented code is useful for:
        # -------------------------------------
        # - Future OpenAI reintegration
        # - Reference during deployment
        # - Understanding historical architecture
        # ----------------------------------------------------------------------

        # self.openai_api_key = os.getenv('OPENAI_API_KEY')

        # if not self.openai_api_key:
        #     raise ValueError(
        #         "OPENAI_API_KEY not found in environment variables. "
        #         "Please check your .env file."
        #     )

        # ----------------------------------------------------------------------
        # --- You can keep other configurations here ---
        # Existing comment retained intentionally.
        #
        # This section stores additional reusable application settings.
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # LOG_LEVEL Configuration
        #
        # Reads LOG_LEVEL from environment variables.
        #
        # Default Value:
        # --------------
        # INFO
        #
        # .upper() ensures consistency regardless of user input.
        #
        # Example:
        # --------
        # info -> INFO
        # debug -> DEBUG
        #
        # Common Logging Levels:
        # ----------------------
        # DEBUG
        # INFO
        # WARNING
        # ERROR
        # CRITICAL
        # ----------------------------------------------------------------------
        self.log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

        # ----------------------------------------------------------------------
        # Microphone Timeout Setting
        #
        # Defines maximum wait time (in seconds)
        # for microphone/speech input operations.
        #
        # Current Value:
        # --------------
        # 5 seconds
        #
        # Why This Is Important:
        # ----------------------
        # Prevents the system from waiting indefinitely
        # during real-time voice interaction.
        #
        # Future Enhancement:
        # -------------------
        # This can also be moved to environment variables:
        #
        # Example:
        # MIC_TIMEOUT=5
        # ----------------------------------------------------------------------
        self.mic_timeout = 5