"""
AI Hub interaction library
"""

import base64
import os
import requests


class AIHub:
    def __init__(self):
        self.url = "https://ai.create.kcl.ac.uk/"
        self.token = open(os.path.expanduser("~/.config/aihub/token")).read().strip()
        self.headers = {"Authorization": "Bearer " + self.token}
        self.set_system("You are a helpful AI assistant.")

    def set_system(self, personality):
        self.personality = personality
        self.chat_history = [{"role": "system", "content": self.personality}]

    def check_model(self, model, images=False):
        if model == "auto":
            model = "gemma3"

        if images and model != "gemma3":
            raise Exception("Images are only supported with the gemma3 model")

        return model

    def prepare_image(self, image):
        return base64.b64encode(image).decode("utf-8")

    def chat(self, prompt, images=[], model="auto"):
        model = self.check_model(model, len(images) > 0)

        message = {"role": "user", "content": prompt}
        if len(images) > 0:
            message["images"] = [self.prepare_image(image) for image in images]
        self.chat_history.append(message)

        data = {
            "model": model,
            "messages": self.chat_history,
        }

        url = self.url + "api/chat/completions"
        response = requests.post(url, json=data, headers=self.headers)
        if response.status_code != 200:
            raise Exception("Error: " + str(response.status_code))

        response = response.json()

        if "choices" in response:
            choices = response["choices"]
            if len(choices) == 0:
                raise Exception("No choices returned")
            message = choices[0]
            if "message" in message:
                message = message["message"]
                self.chat_history.append(message)
                return message["content"]

        raise Exception("Unknown error")
