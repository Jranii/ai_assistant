"""
===============================================================================
File Name: transcription.py
Project: AI-Enabled Real-Time Voice Transcription & Reflexive Question Generator
===============================================================================

Purpose of this File:
---------------------
This file handles real-time voice transcription functionality for the
AI healthcare consultation system.

It converts spoken patient audio into text using OpenAI Whisper.

This module acts as the "Speech-to-Text Engine" of the application.

Main Responsibilities:
----------------------
1. Capture live microphone audio
2. Detect spoken language dynamically
3. Convert speech into text
4. Support multilingual transcription
5. Handle real-time patient voice input
6. Return clean transcribed text to the system

Why This Module Is Important:
-----------------------------
The consultation system is voice-enabled.

Without this module:
- Patients would need to type manually
- Real-time voice interaction would not exist
- Conversational flow would feel unnatural

This module enables:
--------------------
- Natural speech interaction
- Hands-free communication
- Faster consultation workflow
- Better accessibility

Model Used:
------------
openai/whisper-base

About Whisper:
---------------
Whisper is OpenAI’s automatic speech recognition (ASR) model.

Capabilities:
--------------
- Multilingual speech recognition
- Language detection
- Noise robustness
- Real-time transcription support

Why whisper-base?
-----------------
- Lightweight enough for local execution
- Faster inference compared to larger Whisper models
- Good balance between:
  - speed
  - accuracy
  - memory usage

Core Workflow:
---------------
Microphone Audio
        ↓
SpeechRecognition captures audio
        ↓
Whisper detects language
        ↓
Whisper transcribes speech
        ↓
Clean text returned to system

Libraries Used:
----------------
- torch
- transformers.pipeline
- WhisperProcessor
- speech_recognition
- logging

Important Features:
-------------------
- Automatic language detection
- English/Hindi support
- Real-time microphone listening
- Ambient noise adjustment
- GPU acceleration support
- Error-safe transcription pipeline

===============================================================================
"""

# ------------------------------------------------------------------------------
# logging module
#
# Used for:
# - Runtime logs
# - Error tracking
# - Debugging
# - Monitoring transcription flow
# ------------------------------------------------------------------------------
import logging

# ------------------------------------------------------------------------------
# PyTorch library
#
# Used for:
# - Running Whisper model
# - GPU acceleration
# - Tensor computations
# ------------------------------------------------------------------------------
import torch

# ------------------------------------------------------------------------------
# HuggingFace Transformers Components
#
# pipeline:
# ---------
# Simplifies AI model inference.
#
# WhisperProcessor:
# -----------------
# Handles:
# - audio preprocessing
# - tokenization
# - feature extraction
# - language token handling
# ------------------------------------------------------------------------------
from transformers import pipeline, WhisperProcessor


# ------------------------------------------------------------------------------
# RealTimeTranscriber Class
#
# Responsible for:
# - Initializing speech transcription model
# - Capturing microphone audio
# - Detecting language
# - Performing speech-to-text conversion
# ------------------------------------------------------------------------------
class RealTimeTranscriber:

    def __init__(self):

        """
        Initializes transcription pipeline and microphone system.

        Main Initialization Steps:
        --------------------------
        1. Load Whisper model
        2. Detect GPU/CPU
        3. Initialize processor
        4. Create ASR pipeline
        5. Configure microphone
        6. Configure speech recognizer
        """

        # ----------------------------------------------------------------------
        # Use a multilingual model like 'base' or 'small'
        # Existing comment retained intentionally.
        #
        # Whisper models:
        # ----------------
        # tiny   -> very fast, lower accuracy
        # base   -> balanced performance
        # small  -> better accuracy, more memory
        # medium -> higher accuracy, slower
        # large  -> best accuracy, highest memory
        #
        # Current Choice:
        # ----------------
        # openai/whisper-base
        #
        # Good balance for real-time healthcare conversations.
        # ----------------------------------------------------------------------
        model_name = "openai/whisper-base"

        # ----------------------------------------------------------------------
        # Device Selection Logic
        #
        # If CUDA-enabled GPU exists:
        #     use GPU
        #
        # Otherwise:
        #     use CPU
        #
        # GPU significantly improves transcription speed.
        # ----------------------------------------------------------------------
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Log model loading status
        logging.info(
            f"Loading Whisper model '{model_name}' for transcription."
        )

        # ----------------------------------------------------------------------
        # We need the processor to access language tokens
        # Existing comment retained intentionally.
        #
        # WhisperProcessor handles:
        # - audio preprocessing
        # - tokenizer setup
        # - feature extraction
        # - language token mapping
        #
        # Language tokens are important for:
        # - automatic language detection
        # - multilingual transcription
        # ----------------------------------------------------------------------
        self.processor = WhisperProcessor.from_pretrained(model_name)

        # ----------------------------------------------------------------------
        # Initialize HuggingFace ASR Pipeline
        #
        # Task:
        # -----
        # automatic-speech-recognition
        #
        # Components:
        # -----------
        # model               -> Whisper model
        # tokenizer           -> text tokenizer
        # feature_extractor   -> audio preprocessing
        # device              -> CPU/GPU selection
        # ----------------------------------------------------------------------
        self.transcription_pipe = pipeline(

            # Speech recognition task
            "automatic-speech-recognition",

            # Whisper model name
            model=model_name,

            # Tokenizer from processor
            tokenizer=self.processor.tokenizer,

            # Audio feature extractor
            feature_extractor=self.processor.feature_extractor,

            # Execution device
            device=self.device
        )

        # ----------------------------------------------------------------------
        # --- For SpeechRecognition library ---
        # Existing comment retained intentionally.
        #
        # speech_recognition library is used for:
        # - microphone access
        # - live audio recording
        # - capturing speech input
        # ----------------------------------------------------------------------

        # Import speech_recognition locally
        import speech_recognition as sr

        # Initialize speech recognizer object
        self.recognizer = sr.Recognizer()

        # Initialize default system microphone
        self.microphone = sr.Microphone()

        # ----------------------------------------------------------------------
        # Energy Threshold Configuration
        #
        # Determines minimum audio energy level
        # required to consider input as speech.
        #
        # Higher value:
        # --------------
        # Less background noise sensitivity
        #
        # Lower value:
        # -------------
        # More sensitive to quieter voices
        # ----------------------------------------------------------------------
        self.recognizer.energy_threshold = 1000

        # Existing comment retained intentionally.
        #
        # Stricter energy threshold reduces accidental noise detection.
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Disable dynamic energy adjustment
        #
        # Why?
        # ----
        # Prevents automatic fluctuation of threshold during runtime.
        #
        # Useful for:
        # - controlled environments
        # - consistent speech detection
        # ----------------------------------------------------------------------
        self.recognizer.dynamic_energy_threshold = False

        # Log successful initialization
        logging.info(
            "Transcriber initialized for dynamic language detection."
        )

    def listen_and_transcribe(self, timeout=7) -> str:

        """
        Listens for audio, detects the language, and then transcribes.

        Existing docstring retained intentionally.

        Parameters:
        -----------
        timeout : int
            Maximum waiting time before timeout occurs.

        Default:
        --------
        7 seconds

        Returns:
        --------
        str
            Transcribed speech text.

        Workflow:
        ---------
        1. Listen to microphone
        2. Capture audio
        3. Detect language
        4. Transcribe speech
        5. Return text
        """

        # Local import for speech recognition
        import speech_recognition as sr

        # ----------------------------------------------------------------------
        # Open microphone stream
        #
        # "source" represents active microphone audio input.
        # ----------------------------------------------------------------------
        with self.microphone as source:

            # ------------------------------------------------------------------
            # Adjust recognizer for surrounding/background noise
            #
            # duration=1:
            # Listen for 1 second to calibrate ambient sound levels.
            #
            # Improves transcription accuracy.
            # ------------------------------------------------------------------
            self.recognizer.adjust_for_ambient_noise(
                source,
                duration=1
            )

            # Log listening start
            logging.info("Listening for response...")

            try:

                # ------------------------------------------------------------------
                # Capture live speech from microphone
                #
                # timeout:
                # --------
                # Max wait time for user to begin speaking.
                #
                # phrase_time_limit:
                # ------------------
                # Max allowed speaking duration.
                # ------------------------------------------------------------------
                audio_data = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=10
                )

                # ------------------------------------------------------------------
                # Convert captured audio into WAV byte format
                #
                # Whisper pipeline expects raw audio bytes.
                # ------------------------------------------------------------------
                wav_bytes = audio_data.get_wav_data()

                # ------------------------------------------------------------------
                # --- 1. Detect Language ---
                # Existing comment retained intentionally.
                #
                # Whisper can automatically identify spoken language.
                #
                # Example:
                # --------
                # English -> "en"
                # Hindi   -> "hi"
                # ------------------------------------------------------------------

                # The pipeline can return language probabilities
                # Existing comment retained intentionally.
                #
                # return_language=True enables language prediction.
                # ------------------------------------------------------------------
                outputs = self.transcription_pipe(
                    wav_bytes,
                    return_language=True
                )

                # Extract detected language
                detected_language = outputs.get("language", "en")

                # Log detected language
                logging.info(
                    f"Detected language: {detected_language}"
                )

                # ------------------------------------------------------------------
                # Constrain to only English or Hindi for stability
                # Existing comment retained intentionally.
                #
                # Why Restrict Languages?
                # -----------------------
                # Prevent:
                # - incorrect multilingual detection
                # - unstable transcription
                # - unsupported workflows
                #
                # Current Supported Languages:
                # -----------------------------
                # en -> English
                # hi -> Hindi
                # ------------------------------------------------------------------

                if detected_language not in ["en", "hi"]:

                    # ------------------------------------------------------------------
                    # Fallback language handling
                    #
                    # If unsupported language detected,
                    # default back to English.
                    # ------------------------------------------------------------------
                    logging.warning(
                        f"Detected '{detected_language}', "
                        f"forcing English as a fallback."
                    )

                    detected_language = "en"

                # ------------------------------------------------------------------
                # --- 2. Transcribe with the detected language ---
                # Existing comment retained intentionally.
                #
                # Perform actual speech-to-text conversion.
                # ------------------------------------------------------------------
                result = self.transcription_pipe(

                    # Input audio
                    wav_bytes,

                    # Whisper generation settings
                    generate_kwargs={

                        # Force detected language
                        "language": detected_language,

                        # Transcription task
                        "task": "transcribe"
                    }
                )

                # ------------------------------------------------------------------
                # Extract final transcribed text
                #
                # .strip() removes extra spaces/newlines.
                # ------------------------------------------------------------------
                transcription = result.get('text', '').strip()

                # Log transcription result
                logging.info(
                    f"Transcription ({detected_language}): {transcription}"
                )

                # Return clean transcription text
                return transcription

            except sr.WaitTimeoutError:

                # ------------------------------------------------------------------
                # Handle user silence / no speech detected
                #
                # Returns empty string instead of crashing.
                # ------------------------------------------------------------------
                return ""

            except Exception as e:

                # ------------------------------------------------------------------
                # General transcription error handling
                #
                # Handles:
                # - microphone issues
                # - model inference failures
                # - audio processing errors
                # - device problems
                # ------------------------------------------------------------------
                logging.error(f"Transcription error: {e}")

                # Safe fallback response
                return ""