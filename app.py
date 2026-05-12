"""
===============================================================================
File Name: app.py
Project: AI-Enabled Real-Time Voice Transcription & Reflexive Question Generator
===============================================================================

Purpose of this File:
---------------------
This is the main backend entry point of the application.

This Flask application is responsible for:
1. Starting the web server
2. Managing doctor-patient conversation flow
3. Maintaining real-time conversation state
4. Serving frontend pages
5. Handling API requests/responses
6. Generating speech audio from questions
7. Processing uploaded medical images
8. Extracting text from images using OCR
9. Generating summaries from extracted text

Main Functionalities:
---------------------
- Real-time consultation handling
- Question queue management
- Conversation history tracking
- Text-to-Speech (TTS)
- OCR-based image text extraction
- AI-based summary generation
- API communication with frontend

Architecture Flow:
------------------
Frontend UI (index.html)
        ↓
Flask Backend (app.py)
        ↓
Consultation State Manager
        ↓
Question Queue / AI Logic
        ↓
Response Generation
        ↓
TTS Audio Generation
        ↓
JSON Response to Frontend

Important Notes:
----------------
- This file acts as the central controller of the project.
- Global in-memory state is currently used.
- Some AI modules are commented out for future integration.
- Audio files are generated dynamically.
- OCR support allows reading text from uploaded images/documents.

Tech Stack Used:
----------------
- Flask              -> Backend framework
- gTTS               -> Text-to-Speech conversion
- pytesseract        -> OCR text extraction
- Pillow (PIL)       -> Image processing
- collections.deque  -> Efficient question queue
===============================================================================
"""

# ------------------------------------------------------------------------------
# Logging library
# Used for debugging, error tracking, and application monitoring
# ------------------------------------------------------------------------------
import logging

# ------------------------------------------------------------------------------
# collections module
# deque is used for efficient queue operations
# Faster than list for pop/append operations from both ends
# ------------------------------------------------------------------------------
import collections

# ------------------------------------------------------------------------------
# json module
# Used for JSON handling if required in future extensions
# ------------------------------------------------------------------------------
import json

# ------------------------------------------------------------------------------
# re module
# Used for regex/string pattern operations
# ------------------------------------------------------------------------------
import re

# ------------------------------------------------------------------------------
# string module
# Used for punctuation removal during text normalization
# ------------------------------------------------------------------------------
import string

# ------------------------------------------------------------------------------
# Flask-related imports
# Flask                  -> Main application framework
# request                -> Access incoming request data
# jsonify                -> Send JSON responses
# render_template        -> Render HTML pages
# send_from_directory    -> Serve generated audio files
# ------------------------------------------------------------------------------
from flask import Flask, request, jsonify, render_template, send_from_directory

# ------------------------------------------------------------------------------
# Google Text-to-Speech library
# Converts text into spoken audio
# ------------------------------------------------------------------------------
from gtts import gTTS

# ------------------------------------------------------------------------------
# os module
# Used for file system operations and path handling
# ------------------------------------------------------------------------------
import os

# ------------------------------------------------------------------------------
# time module
# Used for timestamp generation
# Helps create unique audio filenames
# ------------------------------------------------------------------------------
import time

# ------------------------------------------------------------------------------
# image upload necessary modules
# Existing comment retained intentionally
# These modules are required for secure image upload + OCR processing
# ------------------------------------------------------------------------------

# secure_filename prevents malicious file names during upload
from werkzeug.utils import secure_filename

# PIL.Image is used to open/process uploaded images
from PIL import Image

# pytesseract performs OCR (Optical Character Recognition)
# It extracts text from uploaded medical images/documents
import pytesseract


# ------------------------------------------------------------------------------
# --- Import Your Existing Modules ---
# Existing comment retained intentionally
# These are custom project modules developed separately
# ------------------------------------------------------------------------------

# Application configuration settings
from config import Config

# Future NLU module for reflexive AI question generation
# Currently commented out but retained for future integration
# from modules.nlu import NLUModule

# Handles conversation state/history management
from modules.state_manager import ConsultationStateManager

# Generates summaries from conversation or OCR extracted text
from modules.summary import SummaryModule

# ------------------------------------------------------------------------------
# --- Flask App Initialization ---
# Existing comment retained intentionally
# Creates Flask application instance
# ------------------------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------------------------
# --- Global In-Memory State ---
# Existing comment retained intentionally
#
# This dictionary stores runtime conversation data.
#
# Why Global State?
# -----------------
# Since this is a real-time conversational application,
# current consultation data must remain accessible across API calls.
#
# Current Stored Items:
# ---------------------
# state_manager   -> Tracks conversation history
# question_queue  -> Stores upcoming questions
# current_question-> Tracks currently active question
#
# NOTE:
# -----
# This approach is suitable for development/small deployments.
# Production systems should ideally use:
# - Redis
# - Database storage
# - Session management
# ------------------------------------------------------------------------------
STATE = {
    "state_manager": None,

    # Queue storing pending consultation questions
    "question_queue": None,

    # Future AI/NLU module placeholder
    # "nlu_module": None,

    # Future summary module placeholder
    # "summary_module": None,

    # Stores currently active question
    "current_question": ""
}

# ------------------------------------------------------------------------------
# --- Helper Functions and Stop Words ---
# Existing comment retained intentionally
#
# These phrases indicate that the user wants to terminate the conversation.
# User input is normalized before comparison.
# ------------------------------------------------------------------------------
USER_STOP_PHRASES = {
    "stop",
    "end conversation",
    "thats all",
    "im done",
    "goodbye",
    "thank you"
}


def normalize_text(text: str) -> str:
    """
    Normalizes user text for consistent comparison.

    Operations Performed:
    ---------------------
    1. Converts text to lowercase
    2. Removes punctuation
    3. Removes leading/trailing spaces

    Example:
    --------
    Input  -> "Thank You!"
    Output -> "thank you"

    This helps accurately detect stop phrases regardless of formatting.
    """

    # Return empty string if text is None or blank
    if not text:
        return ""

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation marks
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra spaces
    return text.strip()


def is_closing_question(question: str) -> bool:
    """
    Checks whether a question is a predefined closing statement.

    If a closing phrase is detected,
    the consultation is considered complete.
    """

    # List of scripted ending phrases
    closing_phrases = [
        "Is there anything else about your health you’d like to share?",
        "We've completed your health evaluation.",
        "Thank you for providing all the necessary information."
    ]

    # Returns True if any closing phrase exists in question
    return any(phrase in question for phrase in closing_phrases)


# ------------------------------------------------------------------------------
# filepath: d:\chhalang\AI_Assistant\AI_Assistant\app.py
#
# Existing comment retained intentionally.
# This likely indicates original local project path during development.
# ------------------------------------------------------------------------------
def text_to_speech(text, filename="question.mp3"):
    """
    Converts text into speech audio using Google Text-to-Speech (gTTS).

    Parameters:
    -----------
    text : str
        Text to convert into spoken audio

    filename : str
        Name of generated audio file

    Returns:
    --------
    filename -> if successful
    None     -> if conversion fails

    Why This Function Exists:
    -------------------------
    The project provides voice-enabled consultation,
    so every generated question is converted into speech.
    """

    try:

        # Create Text-to-Speech object
        tts = gTTS(text=text, lang='en')

        # Save generated audio as MP3
        tts.save(filename)

        # Return generated filename
        return filename

    except Exception as e:

        # Log TTS errors for debugging
        logging.error(f"TTS error: {e}")

        return None


# ------------------------------------------------------------------------------
# --- Flask Routes ---
# Existing comment retained intentionally
#
# Flask routes define API endpoints of the application.
# Each route handles a specific frontend/backend interaction.
# ------------------------------------------------------------------------------

@app.route("/")
def index():
    """
    Renders the main frontend page.

    This loads index.html,
    which contains the chatbot UI.
    """

    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start_conversation():
    """
    Initializes and starts a completely new consultation session.

    Main Workflow:
    --------------
    1. Reset consultation state
    2. Create question queue
    3. Load scripted questions
    4. Fetch first question
    5. Generate audio for first question
    6. Return question + audio URL
    """

    logging.info("--- NEW CONVERSATION STARTED ---")

    # Create new consultation state manager
    STATE["state_manager"] = ConsultationStateManager()

    # Create empty deque queue
    # deque provides efficient FIFO operations
    STATE["question_queue"] = collections.deque()

    # Future NLU module initialization
    # STATE["nlu_module"] = NLUModule()

    # Future summary module initialization
    #STATE["summary_module"] = SummaryModule()

    # Simple debug print
    print('Hi')

    # Load all scripted consultation questions into queue
    for module in STATE["state_manager"].script['modules']:

        # Each module contains multiple questions
        for question in module['questions']:

            # Append questions sequentially
            STATE["question_queue"].append(question)

    # Debug print queue contents
    print(STATE["question_queue"])

    # Retrieve first question from queue
    first_question = STATE["question_queue"].popleft()

    # Save doctor's question into conversation history
    STATE["state_manager"].add_to_history("doctor", first_question)

    # Track current active question
    STATE["current_question"] = first_question

    # Debug print
    print(first_question)

    # Existing commented code retained intentionally
    # This regex was likely used for cleaning special characters
    # cleaned_question = re.sub(r"[^a-zA-Z0-9\s.,?!]", "", first_question)

    # Generate speech audio for first question
    audio_file = text_to_speech(first_question)

    # Generate accessible frontend audio URL
    if audio_file:
        audio_url = f"/{audio_file}"
    else:
        audio_url = None

    # Debug print
    print(first_question)

    # Return question and audio URL to frontend
    return jsonify({
        "question": first_question,
        "audio_url": audio_url
    })


@app.route("/send_response", methods=["POST"])
def handle_response():
    """
    Handles patient response and generates next system action.

    Main Workflow:
    --------------
    1. Receive patient response
    2. Detect stop commands
    3. Store response in history
    4. Generate reflexive questions (future AI logic)
    5. Fetch next question
    6. Generate TTS audio
    7. Return response to frontend
    """

    # Extract JSON request body
    data = request.json

    # Extract user's response text
    customer_response = data.get("response", "")

    # ------------------------------------------------------------------------------
    # --- DEFINITIVE FIX: CHECK FOR STOP COMMAND FIRST ---
    # Existing comment retained intentionally
    #
    # This logic ensures conversation termination commands
    # are handled BEFORE normal processing.
    # ------------------------------------------------------------------------------

    # Normalize text before comparison
    normalized_response = normalize_text(customer_response)

    # Check if user wants to stop consultation
    if normalized_response in USER_STOP_PHRASES:

        logging.info(
            f"Stop phrase '{normalized_response}' detected. Ending conversation."
        )

        # Save user's final response into history
        STATE["state_manager"].add_to_history(
            "customer",
            customer_response
        )

        # Existing commented code retained intentionally
        # Intended future AI-generated summary functionality
        # summary = STATE["summary_module"].generate_summary(
        #     STATE["state_manager"].get_history()
        # )

        # Temporary placeholder summary
        summary = ""

        # Return final conversation response
        return jsonify({
            "final_summary":
            f"Conversation ended by user.\n\n--- SUMMARY ---\n{summary}"
        })

    # ------------------------------------------------------------------------------
    # --- If not a stop command, proceed with normal logic ---
    # Existing comment retained intentionally
    # ------------------------------------------------------------------------------

    # Save customer response into history
    STATE["state_manager"].add_to_history(
        "customer",
        customer_response
    )

    # ------------------------------------------------------------------------------
    # Future Reflexive AI Question Generation Logic
    #
    # Existing commented block retained intentionally.
    #
    # Planned Workflow:
    # -----------------
    # 1. Analyze conversation history
    # 2. Analyze current question
    # 3. Analyze customer response
    # 4. Generate context-aware follow-up questions
    #
    # This is where transformer/HuggingFace models
    # would dynamically generate reflexive questions.
    # ------------------------------------------------------------------------------

    # reflexive_data = STATE["nlu_module"].generate_reflexive_questions(
    #     STATE["state_manager"].get_history(),
    #     STATE["current_question"],
    #     customer_response
    # )

    '''
    Existing commented logic retained intentionally.

    This block would:
    -----------------
    1. Extract AI-generated questions
    2. Insert them dynamically at front of queue
    3. Allow adaptive/context-aware consultation flow
    '''

    '''
    suggested_questions = reflexive_data.get('questions', [])

    if suggested_questions:

        logging.info(f"AI suggests: {suggested_questions}")

        # Insert AI-generated questions at queue front
        # reversed() preserves intended order
        STATE["question_queue"].extendleft(reversed(suggested_questions))
    '''

    # Check whether more scripted questions exist
    if STATE["question_queue"]:

        # Get next question from queue
        next_question = STATE["question_queue"].popleft()

        # Update current active question
        STATE["current_question"] = next_question

        # Check if this is a consultation-ending question
        if is_closing_question(next_question):

            # Existing commented summary logic retained intentionally
            # summary = STATE["summary_module"].generate_summary(
            #     STATE["state_manager"].get_history()
            # )

            # Temporary empty summary
            summary = ""

            return jsonify({
                "final_summary":
                f"Conversation ended by completing the script.\n\n"
                f"--- SUMMARY ---\n{summary}"
            })

        else:

            # Save doctor's next question into history
            STATE["state_manager"].add_to_history(
                "doctor",
                next_question
            )

            # ----------------------------------------------------------------------
            # When generating audio file for a question
            # Existing comment retained intentionally
            #
            # Timestamp-based filenames prevent overwriting
            # multiple generated audio files.
            # ----------------------------------------------------------------------

            # unique filename by timestamp
            # Existing inline comment retained intentionally
            filename = f"question_{int(time.time()*1000)}.mp3"

            # Generate TTS audio
            audio_file = text_to_speech(next_question, filename)

            # Create frontend-accessible audio URL
            if audio_file:
                audio_url = f"/{audio_file}"
            else:
                audio_url = None

            # Debug print
            print(next_question)

            # Return next question and audio URL
            return jsonify({
                "question": next_question,
                "audio_url": audio_url
            })

    else:

        # ------------------------------------------------------------------------------
        # This case is a fallback if the queue becomes empty unexpectedly.
        # Existing comment retained intentionally
        #
        # Ideally this condition should rarely occur.
        # Acts as a safety mechanism.
        # ------------------------------------------------------------------------------

        # Existing future summary generation logic retained
        # summary = STATE["summary_module"].generate_summary(
        #     STATE["state_manager"].get_history()
        # )

        # Temporary empty summary
        summary = ""

        # Generate audio for summary
        audio_file = text_to_speech(summary)

        # Create audio URL
        if audio_file:
            audio_url = f"/{audio_file}"
        else:
            audio_url = None

        # Debug print
        print(summary)

        # Return fallback response
        return jsonify({
            "final_summary":
            f"End of script reached.\n\n--- SUMMARY ---\n{summary}"
        })


@app.route("/<filename>")
def serve_audio(filename):
    """
    Serves generated MP3 files to frontend/browser.

    Example:
    --------
    Browser requests:
    /question_123456.mp3

    Flask returns corresponding audio file.
    """

    return send_from_directory(os.getcwd(), filename)


# ------------------------------------------------------------------------------
# Allow uploads of only images (jpeg/png)
# Existing comment retained intentionally
#
# Defines supported image formats for OCR processing.
# ------------------------------------------------------------------------------

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Directory used to store uploaded images
UPLOAD_FOLDER = "uploads"

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Save upload folder path in Flask configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    """
    Validates uploaded file extension.

    Returns:
    --------
    True  -> if file type is allowed
    False -> otherwise
    """

    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """
    Handles uploaded medical image/documents.

    Workflow:
    ---------
    1. Validate uploaded image
    2. Save image securely
    3. Extract text using OCR
    4. Generate summary from extracted text
    5. Return summary to frontend
    """

    # Check whether image exists in request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # Extract uploaded file
    file = request.files['image']

    # Check empty filename
    if file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400

    # Validate uploaded image type
    if file and allowed_file(file.filename):

        # Sanitize filename for security
        filename = secure_filename(file.filename)

        # Create complete file path
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save uploaded image locally
        file.save(filepath)

        try:

            # Open uploaded image
            img = Image.open(filepath)

            # Extract text using OCR
            ocr_text = pytesseract.image_to_string(img).strip()

            # Check if meaningful text was extracted
            if ocr_text and len(ocr_text) > 10:

                # Initialize summary generation module
                summary_module = SummaryModule()

                # Format OCR text into conversation-style structure
                conversation_history = [
                    {
                        "role": "customer",
                        "content": ocr_text
                    }
                ]

                # Generate AI summary
                summary_text = summary_module.generate_summary(
                    conversation_history
                )

                # Handle empty summary output
                if not summary_text.strip():
                    summary_text = (
                        "Could not generate summary from the text."
                    )

            else:

                # OCR failed or insufficient text found
                summary_text = (
                    "Image received. No significant text detected."
                )

        except Exception as e:

            # Log OCR processing errors
            logging.error(f"OCR processing error: {e}")

            return jsonify({
                'error': f'Failed to process image: {e}'
            }), 500

        # Return generated summary
        return jsonify({'summary': summary_text})

    else:

        # Return validation error for unsupported image types
        return jsonify({
            'error': 'Allowed image types: png, jpg, jpeg'
        }), 400

'''
------------------------------------------------------------------------------
Existing commented-out backup implementation retained intentionally.

Purpose:
--------
This older version directly returned OCR text instead of AI-generated summary.

Difference from current version:
--------------------------------
Old Version:
- Only extracted raw OCR text

Current Version:
- Extracts OCR text
- Passes extracted text into SummaryModule
- Returns AI-generated summary

Keeping this code is useful for:
--------------------------------
- Future rollback
- Debugging OCR issues
- Comparing raw OCR vs summarized output
------------------------------------------------------------------------------
'''

'''
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            img = Image.open(filepath)
            
            # 1. Extract text with OCR
            ocr_text = pytesseract.image_to_string(img).strip()
            
            # 2. If text is found and above a reasonable minimum length, return it
            if ocr_text and len(ocr_text) > 10:
                summary_text = ocr_text
            else:
                # 3. If no or too little text found,
                # Return a fallback message indicating an image was uploaded
                summary_text = "Image received. No significant text detected."
                
                # Optional: add lightweight image description or analysis here if needed
                
        except Exception as e:
            logging.error(f"OCR processing error: {e}")
            return jsonify({'error': f'Failed to process image: {e}'}), 500
        return jsonify({'summary': summary_text})
    else:
        return jsonify({'error': 'Allowed image types: png, jpg, jpeg'}), 400
'''

# ------------------------------------------------------------------------------
# Main Application Entry Point
#
# This block ensures the Flask server starts
# only when this file is executed directly.
# ------------------------------------------------------------------------------

if __name__ == "__main__":

    # Configure application logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Start Flask development server
    app.run(debug=True, port=5000)