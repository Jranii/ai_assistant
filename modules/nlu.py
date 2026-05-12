"""
===============================================================================
File Name: nlu.py
Project: AI-Enabled Real-Time Voice Transcription & Reflexive Question Generator
===============================================================================

Purpose of this File:
---------------------
This file contains the Natural Language Understanding (NLU) module
responsible for generating intelligent reflexive follow-up questions
during the doctor-patient consultation.

This module acts as the "AI Brain" of the system.

Main Responsibilities:
----------------------
1. Load and initialize transformer-based language model
2. Analyze patient responses
3. Detect meaningful medical responses
4. Generate context-aware follow-up questions
5. Avoid generating unnecessary questions for simple replies
6. Clean and structure AI-generated output

Why This Module Is Important:
-----------------------------
Traditional scripted healthcare bots only ask predefined questions.

This NLU module makes the consultation:
- Dynamic
- Adaptive
- Context-aware
- More human-like

Example:
--------
Doctor:
"Do you have headaches?"

Patient:
"Yes, especially during the night."

AI Follow-up:
"How long have you been experiencing nighttime headaches?"

This creates a more intelligent consultation experience.

Model Used:
------------
microsoft/phi-2

About Phi-2:
-------------
- Lightweight transformer-based language model
- Developed by Microsoft
- Efficient for local inference
- Smaller than large enterprise LLMs
- Suitable for local/offline AI applications

Key AI Features:
----------------
- Transformer-based text generation
- Local inference (no external API dependency)
- Prompt engineering
- Dynamic follow-up generation
- Medical conversational context understanding

Libraries Used:
----------------
- torch
- transformers.pipeline
- regex processing
- logging

Important Notes:
----------------
- Model runs locally using HuggingFace Transformers.
- GPU acceleration is automatically enabled if CUDA is available.
- AI-generated output is aggressively cleaned before usage.
- Non-informative responses are filtered to avoid unnecessary AI generation.

===============================================================================
"""

# ------------------------------------------------------------------------------
# logging module
#
# Used for:
# - AI model loading logs
# - Error tracking
# - Runtime monitoring
# ------------------------------------------------------------------------------
import logging

# ------------------------------------------------------------------------------
# PyTorch library
#
# Used for:
# - Deep learning operations
# - GPU detection
# - Running transformer models
# ------------------------------------------------------------------------------
import torch

# ------------------------------------------------------------------------------
# HuggingFace Transformers Pipeline
#
# pipeline() simplifies transformer model usage.
#
# Here it is used for:
# - Text generation
# - AI follow-up question generation
# ------------------------------------------------------------------------------
from transformers import pipeline

# ------------------------------------------------------------------------------
# re module
#
# Used for:
# - Regex-based text cleanup
# - Removing unwanted AI-generated artifacts
# ------------------------------------------------------------------------------
import re


# ------------------------------------------------------------------------------
# NLU Module Class
#
# This class handles:
# - AI model loading
# - AI inference
# - Reflexive question generation
# - Output cleanup
# ------------------------------------------------------------------------------
class NLUModule:

    def __init__(self):

        # ----------------------------------------------------------------------
        # HuggingFace model name
        #
        # microsoft/phi-2 is a lightweight transformer model.
        #
        # Why Phi-2?
        # -----------
        # - Good performance for smaller hardware
        # - Efficient local inference
        # - Suitable for conversational tasks
        # - Lower memory usage compared to larger LLMs
        # ----------------------------------------------------------------------
        self.model_name = "microsoft/phi-2"

        # ----------------------------------------------------------------------
        # Device Selection Logic
        #
        # If CUDA-enabled GPU is available:
        #     device = 0 (GPU)
        #
        # Otherwise:
        #     device = -1 (CPU)
        #
        # This automatically optimizes inference performance.
        # ----------------------------------------------------------------------
        self.device = 0 if torch.cuda.is_available() else -1

        # Log model loading information
        logging.info(
            f"Loading local model '{self.model_name}'... "
            f"This may take some time and memory."
        )

        # ----------------------------------------------------------------------
        # Phi-2 requires trusting remote code to load its specific architecture
        # Existing comment retained intentionally.
        #
        # Explanation:
        # ------------
        # Some HuggingFace models include custom architecture code.
        #
        # trust_remote_code=True allows:
        # - Loading custom model implementation
        # - Using model-specific logic
        #
        # Security Note:
        # --------------
        # This should only be enabled for trusted model sources.
        # ----------------------------------------------------------------------

        # Initialize HuggingFace text-generation pipeline
        self.question_generator = pipeline(

            # Pipeline task type
            'text-generation',

            # Transformer model name
            model=self.model_name,

            # CPU/GPU device
            device=self.device,

            # Allow model-specific remote architecture loading
            trust_remote_code=True
        )

        # Log successful initialization
        logging.info(
            "Local NLU module with Phi-2 initialized successfully."
        )

    def generate_reflexive_questions(
        self,
        conversation_history: list,
        current_question: str,
        customer_response: str
    ) -> dict:

        """
        Generates a follow-up question only if the patient's response
        is medically significant.

        Existing docstring retained intentionally.

        Parameters:
        -----------
        conversation_history : list
            Full consultation history

        current_question : str
            Current doctor question

        customer_response : str
            Latest patient response

        Returns:
        --------
        dict containing:
        - questions
        - sentiment

        Example Return:
        ---------------
        {
            "questions": ["How long have you had chest pain?"],
            "sentiment": "unknown"
        }
        """

        # ----------------------------------------------------------------------
        # --- 1. THE "GATE": CHECK FOR NON-INFORMATIVE RESPONSES ---
        # Existing comment retained intentionally.
        #
        # Purpose:
        # --------
        # Prevent unnecessary AI generation for very simple replies.
        #
        # This improves:
        # ---------------
        # - Performance
        # - Cost efficiency
        # - Conversation quality
        # - AI relevance
        #
        # Example Non-Informative Responses:
        # ----------------------------------
        # "yes"
        # "no"
        # "okay"
        # "fine"
        # ----------------------------------------------------------------------

        non_informative_phrases = {

            "yes",
            "no",
            "ok",
            "okay",
            "alright",
            "fine",
            "correct",
            "thank you",
            "thanks",
            "i don't know",
            "not sure",
            "maybe"
        }

        # ----------------------------------------------------------------------
        # Also check for simple, short answers that are just names or numbers
        # Existing comment retained intentionally.
        #
        # Logic:
        # ------
        # If response contains <= 2 words,
        # it is likely too short for meaningful AI follow-up generation.
        #
        # Examples:
        # ----------
        # "John Doe"
        # "2 days"
        # "No pain"
        # ----------------------------------------------------------------------

        if (
            customer_response.strip().lower() in non_informative_phrases
            or
            len(customer_response.split()) <= 2
        ):

            # ------------------------------------------------------------------
            # If the response is simple, do not generate a question.
            # Existing comment retained intentionally.
            #
            # Avoids unnecessary/reflexive AI generation.
            # ------------------------------------------------------------------

            logging.info(
                "Patient response is non-informative. "
                "Skipping AI question generation."
            )

            return {
                "questions": [],
                "sentiment": "neutral"
            }

        # -----------------------------------------------------------
        # Existing separator comment retained intentionally
        # -----------------------------------------------------------

        # ----------------------------------------------------------------------
        # --- 2. THE REFINED PROMPT ---
        # Existing comment retained intentionally.
        #
        # Prompt Engineering Section
        #
        # This prompt guides the LLM to:
        # - Understand medical context
        # - Focus on patient response
        # - Generate concise medical follow-up
        #
        # Why Prompt Engineering Matters:
        # --------------------------------
        # LLM output quality heavily depends on prompt clarity.
        # ----------------------------------------------------------------------

        prompt = f"""
        This is a medical interview. The doctor's goal is to understand the patient's health.
        Patient's last response: "{customer_response}"
        Generate a single, concise, and medically relevant follow-up question based on this response.
        Follow-up Question:
        """

        try:

            # Log inference start
            logging.info("Generating AI follow-up question...")

            # ------------------------------------------------------------------
            # Generate text using transformer model
            # ------------------------------------------------------------------

            outputs = self.question_generator(

                # Input prompt
                prompt,

                # Maximum number of new tokens generated
                #
                # Prevents excessively long responses.
                max_new_tokens=30,

                # Generate only one response
                num_return_sequences=1,

                # Prevent tokenizer padding issues
                #
                # eos_token_id = End-of-sequence token
                pad_token_id=self.question_generator.tokenizer.eos_token_id
            )

            # ------------------------------------------------------------------
            # --- 3. BULLETPROOF PARSING AND CLEANUP ---
            # Existing comment retained intentionally.
            #
            # AI-generated output can contain:
            # - Extra formatting
            # - Prompt repetition
            # - Unwanted prefixes
            # - Broken formatting
            #
            # This section cleans generated output aggressively.
            # ------------------------------------------------------------------

            # Initialize empty response
            full_text = ""

            # Validate pipeline output structure
            if isinstance(outputs, list) and outputs:

                # Extract first generated result
                first_item = outputs[0]

                # Ensure dictionary structure exists
                if (
                    isinstance(first_item, dict)
                    and
                    'generated_text' in first_item
                ):

                    # Extract generated text
                    full_text = first_item['generated_text']

            # ------------------------------------------------------------------
            # Extract only the text generated after the prompt
            # Existing comment retained intentionally.
            #
            # Since generated output includes original prompt,
            # remove prompt portion.
            # ------------------------------------------------------------------

            generated_text = full_text[len(prompt):].strip()

            # ------------------------------------------------------------------
            # Find the first complete line of text
            # Existing comment retained intentionally.
            #
            # Prevents multi-line/unstructured outputs.
            # ------------------------------------------------------------------

            first_question = generated_text.split('\n')[0].strip()

            # ------------------------------------------------------------------
            # Final aggressive cleanup of any unwanted characters or phrases
            # Existing comment retained intentionally.
            #
            # Removes:
            # - Quotes
            # - Role labels
            # - Prompt artifacts
            # ------------------------------------------------------------------

            # Remove quotes
            first_question = re.sub(r"['\"]", "", first_question)

            # Remove unwanted prefixes
            first_question = re.sub(
                r"^(Doctor:|Patient:|Follow-up Question:)\s*",
                "",
                first_question,
                flags=re.IGNORECASE
            )

            # Convert into list format if valid question exists
            questions = [first_question] if first_question else []

            # Return structured output
            return {
                "questions": questions,
                "sentiment": "unknown"
            }

        except Exception as e:

            # ------------------------------------------------------------------
            # Error handling during model inference
            # ------------------------------------------------------------------

            logging.error(
                f"Error during local model inference with Phi-2: {e}"
            )

            # Return safe fallback response
            return {
                "questions": [],
                "sentiment": "unknown"
            }