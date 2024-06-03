"""
AI Hub interaction library
"""

import base64
import os
import requests


class AIHub:
    PERSONALITY_DEFAULT = "You are a helpful AI assistant."
    PERSONALITY_COMMAND_ONLY = "You can only output commands, no explanations."
    PERSONALITY_JSON_ONLY = "You can only output JSON, no explanations."
    PERSONALITY_YAML_ONLY = "You can only output YAML, no explanations."

    MODEL_LLAMA3_8B = "llama3:instruct" # Default and recommended model
    MODEL_LLAMA3_LLAVA = "llava-llama3:latest" # For image generation
    MODEL_AYA = "aya:8b"
    MODEL_PHI3 = "phi3:instruct"
    MODEL_MISTRAL = "mistral:instruct"
    MODEL_CODESTRAL = "codegemma:22b" # For coding use cases
    MODEL_GEMMA = "gemma:instruct"
    MODEL_CODEGEMMA = "codegemma:instruct" # For coding use cases
    MODEL_MIXTRAL_7b = "mixtral:latest"
    MODEL_MIXTRAL_22b = "mixtral:8x22b" # Can fail depending on load.
    MODEL_WIZARDLM2 = "wizardlm2:7b"

    def __init__(self):
        self.url = "https://ai.create.kcl.ac.uk/"
        self.token = open(os.path.expanduser("~/.config/aihub/token")).read().strip()
        self.headers = {"Authorization": "Bearer " + self.token}
        self.set_system(AIHub.PERSONALITY_DEFAULT)

    def set_system(self, personality):
        self.personality = personality
        self.chat_history = [{"role": "system", "content": self.personality}]

    def check_model(self, model, images=False):
        if model == "auto":
            if images:
                model = AIHub.MODEL_LLAVA
            else:
                model = AIHub.MODEL_LLAMA3_8B

        if images and model != AIHub.MODEL_LLAVA:
            raise Exception("Images are only supported with the LLAMA3-LLAVA model")

        return model

    def prepare_image(self, image):
        return base64.b64encode(image).decode("utf-8")

    def ask(self, prompt, images=[], model="auto"):
        model = self.check_model(model, len(images) > 0)
        data = {"model": model, "prompt": prompt, "stream": False}

        if self.personality != AIHub.PERSONALITY_DEFAULT:
            data["system"] = self.personality

        if len(images) > 0:
            data["images"] = [self.prepare_image(image) for image in images]

        url = self.url + "ollama/api/generate"
        response = requests.post(url, json=data, headers=self.headers)
        response = response.json()
        if "response" in response:
            return response["response"]

        if "detail" in response:
            raise Exception(response["detail"])
        raise Exception("Unknown error")

    def chat(self, prompt, images=[], model="auto"):
        model = self.check_model(model, len(images) > 0)

        message = {"role": "user", "content": prompt}
        if len(images) > 0:
            message["images"] = [self.prepare_image(image) for image in images]
        self.chat_history.append(message)

        data = {
            "model": model,
            "stream": False,
            "messages": self.chat_history,
        }

        url = self.url + "ollama/api/chat"
        response = requests.post(url, json=data, headers=self.headers)
        response = response.json()
        if "message" in response:
            message = response["message"]
            self.chat_history.append(message)
            return message["content"]

        if "detail" in response:
            raise Exception(response["detail"])
        raise Exception("Unknown error")
