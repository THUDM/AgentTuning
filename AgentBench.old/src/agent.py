# Import necessary modules and libraries
from typing import List
import os
import json
import sys
import time
import re
import math
import random
import datetime
import argparse
import requests

# Define a custom exception for sessions
class SessionExeption(Exception):
    pass

# Define a class for managing conversation sessions
class Session:
    def __init__(self, model_inference, history=None) -> None:
        # Initialize the session with a history of messages or an empty list
        self.history: list[dict] = history or []
        self.exception_raised = False
        # Wrap the model inference function for error handling
        self.model_inference = self.wrap_inference(model_inference)

    # Add a message to the conversation history
    def inject(self, message: dict) -> None:
        # Ensure the message is in the expected format
        assert isinstance(message, dict)
        assert "role" in message and "content" in message
        assert isinstance(message["role"], str)
        assert isinstance(message["content"], str)
        assert message["role"] in ["user", "agent"]
        # Append the message to the conversation history
        self.history.append(message)

    # Generate a response based on the conversation history
    def action(self, extend_messages: List[dict] = None) -> str:
        # If extend_messages is provided, add those messages to the history
        extend = []
        if extend_messages:
            if isinstance(extend_messages, list):
                extend.extend(extend_messages)
            elif isinstance(extend_messages, dict):
                extend.append(extend_messages)
            else:
                raise Exception("Invalid extend_messages")
        # Get the model's response based on the conversation history
        result = self.model_inference(self.history + extend)
        self.history.extend(extend)
        self.history.append({"role": "agent", "content": result})
        return result
    
    # Calculate the number of segments in a message
    def _calc_segments(self, msg: str):
        segments = 0
        current_segment = ""
        inside_word = False

        for char in msg:
            if char.isalpha():
                current_segment += char
                if not inside_word:
                    inside_word = True
                if len(current_segment) >= 7:
                    segments += 1
                    current_segment = ""
                    inside_word = False
            else:
                if inside_word:
                    segments += 1
                    current_segment = ""
                    inside_word = False
                if char not in [" ", "\n"]:
                    segments += 1

        if len(current_segment) > 0:
            segments += 1

        return segments
    
    # Wrap the model inference function for error handling
    def wrap_inference(self, inference_function: Callable[[List[dict]], str]) -> Callable[[List[dict]], str]:
        def _func(history: List[dict]) -> str:
            if self.exception_raised:
                return ""
            messages = self.filter_messages(history)
            try:
                result = inference_function(messages)
            except Exception as e:
                print(e)
                import traceback
                traceback.print_exc()
                print("Warning: Exception raised during inference.")
                self.exception_raised = True
                result = ""
            return result
        return _func

    # Filter and process messages in the conversation history
    def filter_messages(self, messages: List[Dict]) -> List[Dict]:
        try:
            assert len(messages) % 2 == 1
            for idx, message in enumerate(messages):
                assert isinstance(message, dict)
                assert "role" in message and "content" in message
                assert isinstance(message["role"], str)
                assert isinstance(message["content"], str)
                assert message["role"] in ["user", "agent"]
                if idx % 2 == 0:
                    assert message["role"] == "user"
                else:
                    assert message["role"] == "agent"
        except:
            raise SessionExeption("Invalid messages")
        threashold_segments = 3500
        return_messages = []
        # Only include the latest {threashold_segments} segments
        
        segments = self._calc_segments(messages[0]["content"])
        
        for message in messages[:0:-1]:
            segments += self._calc_segments(message["content"])
            if segments >= threashold_segments:
                break
            return_messages.append(message)
            
        if len(return_messages) > 0 and return_messages[-1]["role"] == "user":
            return_messages.pop()
        
        instruction = messages[0]["content"]

        omit = len(messages) - len(return_messages) - 1
        
        if omit > 0:
            instruction += f"\n\n[NOTICE] {omit} messages are omitted."
            print(f"Warning: {omit} messages are omitted.")
        
        return_messages.append({
            "role": "user",
            "content": instruction
        })
        
        return_messages.reverse()
        return return_messages

# Define a base class for conversational agents
class Agent:
    def __init__(self, **configs) -> None:
        self.name = configs.pop("name", None)
        self.src = configs.pop("src", None)
        # For any remaining config keys, print a warning
        # for key in configs:
        #     print(f"Warning: Unknown argument '{key}' for the agent.")
        pass

    # Create a new conversation session
    def create_session(self) -> Session:
        return Session(self.inference)

    # Define the model's inference function (to be implemented by subclasses)
    def inference(self, history: List[dict]) -> str:
        raise NotImplementedError

