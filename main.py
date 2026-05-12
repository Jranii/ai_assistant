"""
===============================================================================
File Name: main.py
Project: AI-Enabled Real-Time Voice Transcription & Reflexive Question Generator
===============================================================================

Purpose of this File:
---------------------
This file acts as the standalone console-based execution entry point
for the AI healthcare consultation system.

Unlike app.py (which provides a Flask web API),
this file runs the consultation directly in terminal/console mode.

Main Responsibilities:
----------------------
1. Initialize all AI/system modules
2. Run real-time doctor-patient consultation loop
3. Capture speech input from patient
4. Transcribe voice into text
5. Generate AI-based reflexive follow-up questions
6. Maintain conversation history
7. Detect conversation-ending commands
8. Generate final consultation summary
9. Save transcript + summary into JSON file

Overall Workflow:
-----------------
Doctor Question
        ↓
Patient Speaks
        ↓
Speech-to-Text Transcription
        ↓
Conversation History Update
        ↓
AI Reflexive Question Generation
        ↓
Dynamic Follow-up Question Queue
        ↓
Consultation Completion
        ↓
Summary Generation
        ↓
Save JSON Report

Difference Between main.py and app.py:
--------------------------------------
main.py:
- Console/terminal-based execution
- Direct interaction flow
- Primarily for local testing/development

app.py:
- Flask web server
- Frontend-integrated interaction
- API-based communication

Important Features:
-------------------
- Real-time speech transcription
- Context-aware AI questioning
- Dynamic follow-up generation
- Consultation memory/state management
- Final structured summary generation

Tech Stack Used:
----------------
- Python
- Speech Transcription
- HuggingFace/Transformer-based NLU
- JSON transcript storage
- Queue-based consultation management

===============================================================================
"""

# ------------------------------------------------------------------------------
# logging module
#
# Used for:
# - Debugging
# - Error tracking
# - Runtime monitoring
# ------------------------------------------------------------------------------
import logging

# ------------------------------------------------------------------------------
# os module
#
# Used for:
# - File system operations
# - Path handling
# - Environment interactions
# ------------------------------------------------------------------------------
import os

# ------------------------------------------------------------------------------
# collections module
#
# deque is used for efficient question queue management.
#
# Why deque?
# ----------
# Faster append/pop operations from both ends compared to lists.
# Ideal for conversational queue systems.
# ------------------------------------------------------------------------------
import collections

# ------------------------------------------------------------------------------
# json module
#
# Used for:
# - Saving final consultation summaries
# - Saving full transcripts
# - Structured output generation
# ------------------------------------------------------------------------------
import json

# ------------------------------------------------------------------------------
# warnings module
#
# Used to suppress unnecessary/non-critical warnings.
# Helps keep console output clean.
# ------------------------------------------------------------------------------
import warnings

# ------------------------------------------------------------------------------
# re module
#
# Used for regex-based text processing if needed.
# ------------------------------------------------------------------------------
import re

# ------------------------------------------------------------------------------
# string module
#
# Used for punctuation removal during text normalization.
# ------------------------------------------------------------------------------
import string


# ------------------------------------------------------------------------------
# --- Suppress Benign Warnings ---
# Existing comment retained intentionally.
#
# Benign warnings are harmless warnings that do not affect functionality.
#
# Suppressing them improves console readability during execution.
# ------------------------------------------------------------------------------

# Ignore UserWarning category warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ------------------------------------
# Existing separator comment retained intentionally
# ------------------------------------


# ------------------------------------------------------------------------------
# Import Project Configuration Module
#
# Loads application settings such as:
# - Logging level
# - Timeout settings
# - Future API configurations
# ------------------------------------------------------------------------------
from config import Config

# ------------------------------------------------------------------------------
# Real-Time Speech Transcription Module
#
# Responsible for:
# - Listening to microphone input
# - Converting speech into text
# ------------------------------------------------------------------------------
from modules.transcription import RealTimeTranscriber

# ------------------------------------------------------------------------------
# Natural Language Understanding (NLU) Module
#
# Responsible for:
# - AI-based reflexive question generation
# - Context-aware follow-up question creation
# - Understanding patient responses
# ------------------------------------------------------------------------------
from modules.nlu import NLUModule

# ------------------------------------------------------------------------------
# Consultation State Manager
#
# Responsible for:
# - Maintaining conversation history
# - Tracking doctor/patient dialogue
# - Managing consultation script state
# ------------------------------------------------------------------------------
from modules.state_manager import ConsultationStateManager

# ------------------------------------------------------------------------------
# Summary Generation Module
#
# Responsible for:
# - Generating final consultation summaries
# - Structuring transcript outputs
# ------------------------------------------------------------------------------
from modules.summary import SummaryModule


# ------------------------------------------------------------------------------
# --- Define Closing and User Stop Phrases ---
# Existing comment retained intentionally.
#
# These predefined phrases help detect:
# 1. Scripted consultation ending
# 2. User-requested conversation termination
# ------------------------------------------------------------------------------

# Script-defined consultation ending phrases
CLOSING_PHRASES = [

    # Final health-sharing question
    "Is there anything else about your health you’d like to share?",

    # Script completion message
    "We've completed your health evaluation.",

    # Thank-you ending message
    "Thank you for providing all the necessary information."
]

# User-triggered stop commands
USER_STOP_PHRASES = {

    # User manually stops consultation
    "stop",

    # Explicit stop instruction
    "stop conversation",

    # User indicates completion
    "im done",

    # Conversation ending phrase
    "goodbye"
}


def normalize_text(text: str) -> str:
    """
    Normalizes text for comparison.

    Existing docstring retained intentionally.

    Why Normalization Is Needed:
    ----------------------------
    User speech/transcription may contain:
    - Mixed casing
    - Punctuation
    - Extra spaces

    Normalization ensures reliable phrase matching.

    Example:
    --------
    Input  -> "GoodBye!"
    Output -> "goodbye"
    """

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation symbols
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove leading/trailing spaces
    return text.strip()


def is_closing_question(question: str) -> bool:
    """
    Checks if the current question is a closing phrase.

    Existing docstring retained intentionally.

    Purpose:
    --------
    Detect whether the consultation script
    has reached its predefined ending stage.
    """

    # Return True if any closing phrase exists inside question
    return any(phrase in question for phrase in CLOSING_PHRASES)


def run_consultation():
    """
    Initializes all modules and runs the main conversational loop.

    Existing docstring retained intentionally.

    This is the core execution function of the console-based system.

    Main Workflow:
    --------------
    1. Load configurations
    2. Initialize AI modules
    3. Load consultation script
    4. Start question-answer loop
    5. Generate AI follow-ups
    6. Generate final summary
    7. Save transcript/output
    """

    # ------------------------------------------------------------------------------
    # --- Initialization ---
    # Existing comment retained intentionally.
    #
    # Responsible for initializing configuration and logging system.
    # ------------------------------------------------------------------------------

    try:

        # Load application configuration
        config = Config()

        # Configure logging settings
        logging.basicConfig(
            level=config.log_level,

            format="%(asctime)s - %(levelname)s - %(message)s",

            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Log successful configuration initialization
        logging.info("Configuration loaded.")

    except Exception as e:

        # Handle configuration loading errors
        logging.error(f"Configuration Error: {e}")

        return

    # ------------------------------------------------------------------------------
    # Initialize Core System Modules
    # ------------------------------------------------------------------------------

    # Real-time voice transcription system
    transcriber = RealTimeTranscriber()

    # AI/NLU module for reflexive question generation
    nlu = NLUModule()

    # Consultation state/history manager
    state_manager = ConsultationStateManager()

    # Final summary generation module
    summarizer = SummaryModule()

    # ------------------------------------------------------------------------------
    # Create Question Queue
    #
    # Queue stores upcoming consultation questions sequentially.
    # ------------------------------------------------------------------------------

    question_queue = collections.deque()

    # Load all scripted consultation modules/questions
    for module in state_manager.script['modules']:

        # Each module contains multiple questions
        for question in module['questions']:

            # Add question into queue
            question_queue.append(question)

    # ------------------------------------------------------------------------------
    # --- Main Conversational Loop ---
    # Existing comment retained intentionally.
    #
    # This loop controls the full doctor-patient interaction lifecycle.
    # ------------------------------------------------------------------------------

    # Tracks consultation turn count
    turn_count = 0

    # Continue while questions exist in queue
    while question_queue:

        # Increment conversation turn counter
        turn_count += 1

        # Retrieve next question from queue
        current_question = question_queue.popleft()

        # Print visual separator for readability
        print("\n" + "="*50)

        # Display doctor's question
        print(f"Doctor Asks: {current_question}")

        # Print separator line
        print("="*50)

        # Save doctor's question into conversation history
        state_manager.add_to_history("doctor", current_question)

        # Check whether current question is consultation-ending phrase
        if is_closing_question(current_question):

            print("Scripted closing question detected. Ending conversation loop.")

            break

        # ------------------------------------------------------------------------------
        # Capture Patient Response
        # ------------------------------------------------------------------------------

        # Initialize empty response
        customer_response = ""

        # Keep listening until valid response is received
        while not customer_response:

            # Listen and transcribe user speech
            customer_response = transcriber.listen_and_transcribe()

            # Handle failed/empty transcription
            if not customer_response:

                print("No valid response detected, please speak again.")

        # ------------------------------------------------------------------------------
        # --- DEFINITIVE FIX: Use an IF/ELSE block for control flow ---
        # Existing comment retained intentionally.
        #
        # This fix ensures:
        # - Stop commands are handled BEFORE AI logic
        # - Conversation exits immediately when requested
        # ------------------------------------------------------------------------------

        # Normalize user response
        normalized_response = normalize_text(customer_response)

        # Check if user wants to end conversation
        if normalized_response in USER_STOP_PHRASES:

            print("User has ended the conversation. Proceeding to summary.")

            # Save final customer response into history
            state_manager.add_to_history(
                "customer",
                customer_response
            )

            # Immediately terminate consultation loop
            break

        else:

            # ----------------------------------------------------------------------
            # Only proceed with AI logic if the user did NOT say a stop phrase
            # Existing comment retained intentionally.
            #
            # This prevents unnecessary AI processing after conversation termination.
            # ----------------------------------------------------------------------

            # Save customer response into conversation history
            state_manager.add_to_history(
                "customer",
                customer_response
            )

            # Display transcribed patient response
            print(f"Patient Responded: {customer_response}")

            # ------------------------------------------------------------------------------
            # Reflexive AI Question Generation
            # ------------------------------------------------------------------------------

            # Skip AI generation during initial turn if required
            if turn_count > 0:

                # Generate context-aware reflexive follow-up questions
                reflexive_data = nlu.generate_reflexive_questions(
                    state_manager.get_history(),
                    current_question,
                    customer_response
                )

                # Extract generated questions from AI output
                suggested_questions = reflexive_data.get('questions', [])

                # Check whether AI generated valid follow-up questions
                if suggested_questions:

                    print(f"\n--- AI Suggests Follow-ups ---")

                    # Display each AI-generated question
                    for q in suggested_questions:
                        print(f" -> {q}")

                    # ------------------------------------------------------------------------------
                    # Add AI-generated questions to FRONT of queue
                    #
                    # reversed() preserves intended conversational order.
                    # extendleft() inserts efficiently at beginning of deque.
                    # ------------------------------------------------------------------------------

                    question_queue.extendleft(reversed(suggested_questions))

                else:

                    # AI generated no additional follow-ups
                    print("--- AI did not provide follow-ups. Proceeding with script. ---")

            else:

                # First-turn handling message
                print("--- First turn complete. Proceeding with scripted questions. ---")

        # -------------------------------------------------------------
        # Existing separator comment retained intentionally
        # -------------------------------------------------------------

    # ------------------------------------------------------------------------------
    # --- Post-Consultation Summary Generation ---
    # Existing comment retained intentionally.
    #
    # Once consultation ends:
    # 1. Generate final summary
    # 2. Save transcript + summary
    # ------------------------------------------------------------------------------

    print("\n" + "="*50)

    print("Consultation finished. Generating final summary...")

    print("="*50)

    # Generate AI consultation summary
    final_summary_text = summarizer.generate_summary(
        state_manager.get_history()
    )

    # ------------------------------------------------------------------------------
    # Create structured output dictionary
    #
    # Includes:
    # - AI-generated summary
    # - Complete conversation transcript
    # ------------------------------------------------------------------------------

    output_data = {

        "summary": final_summary_text,

        "full_transcript": state_manager.get_history()
    }

    # Output JSON filename
    output_filename = "consultation_summary.json"

    # ------------------------------------------------------------------------------
    # Save consultation output as JSON file
    # ------------------------------------------------------------------------------

    with open(output_filename, 'w') as f:

        json.dump(output_data, f, indent=4)

    # ------------------------------------------------------------------------------
    # Display Final Results
    # ------------------------------------------------------------------------------

    print(f"\n--- Final Consultation Summary ---")

    print(final_summary_text)

    print("--------------------------------")

    print(f"Full summary and transcript saved to '{output_filename}'")


# ------------------------------------------------------------------------------
# Main Execution Entry Point
#
# Ensures consultation starts only when:
# python main.py
#
# This prevents accidental execution during imports.
# ------------------------------------------------------------------------------

if __name__ == "__main__":

    # Start complete consultation workflow
    run_consultation()