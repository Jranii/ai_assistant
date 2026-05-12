"""
===============================================================================
File Name: state_manager.py
Project: AI-Enabled Real-Time Voice Transcription & Reflexive Question Generator
===============================================================================

Purpose of this File:
---------------------
This file manages the complete consultation state and conversation flow
for the AI healthcare assistant.

It acts as the "memory manager" of the system.

Main Responsibilities:
----------------------
1. Load predefined medical consultation script
2. Track current consultation progress
3. Store full conversation history
4. Manage doctor-patient dialogue sequence
5. Provide next consultation questions
6. Maintain conversational context for AI modules

Why This Module Is Important:
-----------------------------
The AI system needs memory/context to:
- Understand ongoing consultation flow
- Generate context-aware follow-up questions
- Produce accurate final summaries
- Track doctor-patient interactions

Without state management:
-------------------------
- Conversation history would be lost
- AI would not know previous responses
- Summary generation would fail
- Consultation flow would become inconsistent

Core Concept:
--------------
This module maintains:
1. Script State
2. Question Position
3. Conversation History

Example Workflow:
-----------------
Load Script
     ↓
Ask Question
     ↓
Store Patient Response
     ↓
Update Conversation History
     ↓
Move to Next Question
     ↓
Repeat Until Consultation Ends

Data Managed:
--------------
1. Script Modules
2. Question Indexes
3. Conversation Transcript
4. Current Consultation State

Example Conversation History:
-----------------------------
[
    {"role": "doctor", "content": "Do you have fever?"},
    {"role": "customer", "content": "Yes, since yesterday"}
]

Why JSON Is Used:
-----------------
The consultation script is stored in JSON because:
- Easy to edit
- Human-readable
- Structured
- Scalable
- Easily portable

===============================================================================
"""

# ------------------------------------------------------------------------------
# json module
#
# Used for:
# - Loading medical consultation script
# - Parsing JSON files into Python dictionaries
# ------------------------------------------------------------------------------
import json

# ------------------------------------------------------------------------------
# logging module
#
# Used for:
# - System logs
# - Initialization tracking
# - Debugging
# ------------------------------------------------------------------------------
import logging


# ------------------------------------------------------------------------------
# ConsultationStateManager Class
#
# This class is responsible for:
# - Managing consultation flow
# - Tracking current question/module
# - Storing conversation history
# - Providing contextual memory
# ------------------------------------------------------------------------------
class ConsultationStateManager:

    def __init__(self, script_path="medical_script.json"):

        """
        Constructor for initializing consultation state manager.

        Parameters:
        -----------
        script_path : str
            Path to the predefined medical consultation script JSON file.

        Default:
        --------
        medical_script.json
        """

        # ----------------------------------------------------------------------
        # Load the predefined question script
        # Existing comment retained intentionally.
        #
        # The consultation script contains:
        # - Medical modules
        # - Question sequences
        # - Structured consultation flow
        #
        # Example JSON Structure:
        # -----------------------
        # {
        #     "modules": [
        #         {
        #             "name": "General Symptoms",
        #             "questions": [
        #                 "Do you have fever?",
        #                 "Do you have cough?"
        #             ]
        #         }
        #     ]
        # }
        # ----------------------------------------------------------------------

        # Open JSON script file in read mode
        with open(script_path, 'r') as f:

            # Load JSON content into Python dictionary
            self.script = json.load(f)

        # ----------------------------------------------------------------------
        # Track Current Module Index
        #
        # Represents which consultation module is active.
        #
        # Example:
        # --------
        # 0 -> General Symptoms
        # 1 -> Medical History
        # 2 -> Lifestyle Questions
        # ----------------------------------------------------------------------
        self.current_module_index = 0

        # ----------------------------------------------------------------------
        # Track Current Question Index
        #
        # Represents current question position inside module.
        #
        # Example:
        # --------
        # Module:
        #   "General Symptoms"
        #
        # Questions:
        #   0 -> Fever?
        #   1 -> Cough?
        # ----------------------------------------------------------------------
        self.current_question_index = 0

        # ----------------------------------------------------------------------
        # Conversation History Storage
        #
        # Stores full doctor-patient interaction history.
        #
        # Why Important?
        # ---------------
        # Used for:
        # - AI context understanding
        # - Reflexive question generation
        # - Final summary generation
        # - Conversation continuity
        #
        # Data Structure:
        # ----------------
        # [
        #     {"role": "doctor", "content": "..."},
        #     {"role": "customer", "content": "..."}
        # ]
        # ----------------------------------------------------------------------
        self.conversation_history = []

        # Log successful initialization
        logging.info("State Manager initialized with script.")

    def get_next_question(self):

        """
        Gets the next question from the predefined script.

        Existing docstring retained intentionally.

        Purpose:
        --------
        Retrieves:
        - Current module name
        - Current question

        Returns:
        --------
        dict:
        {
            "module": "...",
            "question": "..."
        }
        """

        # ----------------------------------------------------------------------
        # Retrieve Current Module Name
        #
        # Access structure:
        # self.script
        #     -> modules
        #         -> current module
        #             -> name
        # ----------------------------------------------------------------------
        module_name = self.script['modules'][self.current_module_index]['name']

        # ----------------------------------------------------------------------
        # Retrieve Current Question
        #
        # Access structure:
        # self.script
        #     -> modules
        #         -> current module
        #             -> questions
        #                 -> current question
        # ----------------------------------------------------------------------
        question = self.script['modules'][self.current_module_index]['questions'][self.current_question_index]

        # ----------------------------------------------------------------------
        # Logic to advance to the next question/module would go here
        # Existing comment retained intentionally.
        #
        # Current Status:
        # ---------------
        # Placeholder implementation.
        #
        # Future Enhancement Possibilities:
        # ---------------------------------
        # 1. Auto-increment question index
        # 2. Module transition logic
        # 3. Dynamic branching conversations
        # 4. Conditional medical pathways
        # 5. AI-prioritized question ordering
        #
        # Example Future Logic:
        # ---------------------
        # self.current_question_index += 1
        #
        # if end_of_module:
        #     self.current_module_index += 1
        # ----------------------------------------------------------------------

        # For simplicity, this is just a placeholder
        # Existing comment retained intentionally.
        #
        # Current implementation only fetches current question
        # without automatic progression logic.
        # ----------------------------------------------------------------------

        # Return structured module/question response
        return {
            "module": module_name,
            "question": question
        }

    def add_to_history(self, role: str, content: str):

        """
        Adds a turn to the conversation history.

        Existing docstring retained intentionally.

        Parameters:
        -----------
        role : str
            Speaker role
            Example:
            - doctor
            - customer

        content : str
            Actual spoken/generated text

        Purpose:
        --------
        Maintains conversational memory for:
        - AI context understanding
        - Summary generation
        - Transcript storage
        """

        # ----------------------------------------------------------------------
        # Append new conversation entry into history
        #
        # Example Stored Format:
        # ----------------------
        # {
        #     "role": "doctor",
        #     "content": "Do you have fever?"
        # }
        # ----------------------------------------------------------------------
        self.conversation_history.append({
            "role": role,
            "content": content
        })

    def get_history(self):

        """
        Returns the complete conversation history.

        Purpose:
        --------
        Used by:
        - NLU module
        - Summary generator
        - Transcript exporter
        - AI context processing

        Returns:
        --------
        list of conversation turns
        """

        # Return entire stored conversation history
        return self.conversation_history