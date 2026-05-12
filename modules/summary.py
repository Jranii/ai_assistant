"""
===============================================================================
File Name: summary.py
Project: AI-Enabled Real-Time Voice Transcription & Reflexive Question Generator
===============================================================================

Purpose of this File:
---------------------
This file contains the AI-powered medical summarization module.

The module is responsible for generating a concise medical summary
from the complete doctor-patient consultation transcript.

It acts as the "Medical Report Generator" of the system.

Main Responsibilities:
----------------------
1. Load transformer-based summarization model
2. Convert conversation history into AI-readable prompt
3. Generate structured medical summaries
4. Highlight key medical information
5. Detect possible medical risks/flags
6. Produce concise consultation reports

Why This Module Is Important:
-----------------------------
During long consultations, manually reviewing transcripts can be difficult.

This module automatically converts lengthy conversations into:
- Short summaries
- Important medical highlights
- Underwriter-relevant insights
- Easy-to-review consultation outputs

Example:
--------
Conversation:
-------------
Doctor: Do you smoke?
Patient: Yes, occasionally.

Doctor: Do you have diabetes?
Patient: Yes, for 5 years.

Generated Summary:
------------------
Patient disclosed diabetes history and occasional smoking habits.
Potential risk factors include chronic diabetes and smoking behavior.

Model Used:
------------
google/flan-t5-base

About FLAN-T5:
--------------
- Instruction-tuned transformer model by Google
- Excellent for summarization and text generation
- Lightweight compared to large enterprise LLMs
- Efficient for local inference

Key AI Features:
----------------
- Transformer-based summarization
- Prompt-engineered medical report generation
- Context-aware summary extraction
- GPU acceleration support

Libraries Used:
----------------
- transformers
- torch
- logging
- json

Important Notes:
----------------
- The model runs locally using HuggingFace Transformers.
- GPU acceleration is automatically enabled if CUDA is available.
- Summary generation is prompt-driven.
- Final summaries are used in:
  - consultation reports
  - transcript exports
  - JSON outputs
  - frontend responses

===============================================================================
"""

# ------------------------------------------------------------------------------
# logging module
#
# Used for:
# - Model loading logs
# - Error tracking
# - Runtime monitoring
# ------------------------------------------------------------------------------
import logging

# ------------------------------------------------------------------------------
# HuggingFace Transformer Components
#
# AutoTokenizer:
# --------------
# Converts human-readable text into model tokens.
#
# AutoModelForSeq2SeqLM:
# ----------------------
# Loads sequence-to-sequence language model used for summarization.
# ------------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------------------------------------------------------------
# PyTorch library
#
# Used for:
# - Running transformer models
# - GPU acceleration
# - Tensor operations
# ------------------------------------------------------------------------------
import torch

# ------------------------------------------------------------------------------
# json module
#
# Imported for future structured output handling if required.
#
# Possible future uses:
# ---------------------
# - Saving summary outputs
# - Structured report exports
# - JSON API responses
# ------------------------------------------------------------------------------
import json


# ------------------------------------------------------------------------------
# SummaryModule Class
#
# Responsible for:
# - Loading summarization model
# - Generating consultation summaries
# - Processing conversation history
# ------------------------------------------------------------------------------
class SummaryModule:

    def __init__(self):

        """
        Initializes summarization model and tokenizer.

        Workflow:
        ---------
        1. Define model name
        2. Detect GPU/CPU
        3. Load tokenizer
        4. Load transformer model
        5. Move model to execution device
        """

        # ----------------------------------------------------------------------
        # HuggingFace model name
        #
        # google/flan-t5-base:
        # - Instruction-tuned
        # - Good summarization performance
        # - Moderate resource usage
        # ----------------------------------------------------------------------
        self.model_name = "google/flan-t5-base"

        # ----------------------------------------------------------------------
        # Device Selection Logic
        #
        # If CUDA-enabled GPU exists:
        #     use GPU
        #
        # Otherwise:
        #     use CPU
        #
        # GPU greatly improves inference speed.
        # ----------------------------------------------------------------------
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Log device/model loading information
        logging.info(
            f"Loading Flan-T5 model {self.model_name} onto device: {device}"
        )

        # ----------------------------------------------------------------------
        # Debug/Progress Logging
        # Existing log retained intentionally.
        #
        # "Done1" indicates tokenizer loading phase is starting/completing.
        #
        # These logs are useful during:
        # - debugging
        # - long model loading
        # - deployment monitoring
        # ----------------------------------------------------------------------
        logging.info("Done1")

        # ----------------------------------------------------------------------
        # Load Tokenizer
        #
        # Tokenizer converts text into model-readable token IDs.
        #
        # Example:
        # --------
        # "Hello"
        #     ↓
        # [8774, 123]
        # ----------------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Additional progress log
        logging.info("Done2")

        # ----------------------------------------------------------------------
        # Load Transformer Summarization Model
        #
        # AutoModelForSeq2SeqLM:
        # - Sequence-to-sequence architecture
        # - Suitable for summarization
        # - Used heavily in text generation tasks
        #
        # .to(device):
        # Moves model to:
        # - GPU if available
        # - otherwise CPU
        # ----------------------------------------------------------------------
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name
        ).to(device)

        # Log successful model loading
        logging.info("Model loaded successfully.")

    def generate_summary(self, conversation_history: list) -> str:

        """
        Generates AI-based medical summary from consultation history.

        Parameters:
        -----------
        conversation_history : list
            Full doctor-patient transcript history.

        Returns:
        --------
        str
            Generated medical consultation summary.

        Example Input:
        --------------
        [
            {"role": "doctor", "content": "Do you smoke?"},
            {"role": "customer", "content": "Yes occasionally"}
        ]

        Example Output:
        ---------------
        "Patient reports occasional smoking habits."
        """

        # ----------------------------------------------------------------------
        # System Prompt
        #
        # This acts as instruction/context for the language model.
        #
        # Prompt Engineering:
        # -------------------
        # The better the prompt,
        # the better the generated summary quality.
        #
        # This prompt specifically instructs the model to:
        # - behave like medical summarization AI
        # - extract key medical insights
        # - identify risk factors
        # ----------------------------------------------------------------------
        system_prompt = """
        You are a medical summarization AI. Based on the provided conversation transcript
        between a doctor and a customer, generate a concise summary. Include:
        - Key medical conditions disclosed.
        - Important lifestyle habits.
        - Any potential risks or flags that require underwriter attention.
        """

        # ----------------------------------------------------------------------
        # Convert conversation history into plain text
        #
        # Why Needed?
        # ------------
        # Transformer models require text prompts,
        # not raw Python objects.
        #
        # Example Output:
        # ---------------
        # doctor: Do you smoke?
        # customer: Yes occasionally
        # ----------------------------------------------------------------------
        history_str = "".join([
            f"{msg['role']}: {msg['content']}"
            for msg in conversation_history
        ])

        # ----------------------------------------------------------------------
        # Create final prompt for model inference
        #
        # Structure:
        # ----------
        # [System Instructions]
        # +
        # [Conversation Transcript]
        # ----------------------------------------------------------------------
        prompt = f"{system_prompt}Conversation:{history_str}"

        try:

            # ------------------------------------------------------------------
            # Convert text prompt into tensors
            #
            # return_tensors="pt":
            # Returns PyTorch tensors
            #
            # .to(self.model.device):
            # Moves tensors to same device as model
            # ------------------------------------------------------------------
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.model.device)

            # ------------------------------------------------------------------
            # Generate AI summary
            #
            # max_new_tokens=200:
            # Limits generated output length.
            #
            # Prevents excessively long summaries.
            # ------------------------------------------------------------------
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200
            )

            # ------------------------------------------------------------------
            # Decode generated token IDs back into readable text
            #
            # skip_special_tokens=True:
            # Removes unnecessary transformer tokens.
            # ------------------------------------------------------------------
            result_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Return final generated summary
            return result_text

        except Exception as e:

            # ------------------------------------------------------------------
            # Error Handling
            #
            # Handles:
            # - Model inference failures
            # - Memory issues
            # - Tokenization problems
            # - Device mismatch errors
            # ------------------------------------------------------------------

            logging.error(f"Summary generation error: {e}")

            # Safe fallback response
            return "Failed to generate summary."