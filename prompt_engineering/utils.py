"""Utility functions."""
import openai
from typing import Optional, Any


class Chat:
    def __init__(self, persona: Optional[str] = None, model: str = "gpt-3.5-turbo") -> None:
        self.persona = persona
        self.model = model

    def __str__(self):
        return f"Get chat completions from model {self.model} using persona: {self.persona}"

    def _make_messages(self, prompt: str) -> list[dict[str, str]]:
        messages = []

        if self.persona:
            messages.append({"role": "system", "content": self.persona})

        messages.append({"role": "user", "content": prompt})

        return messages

    def _get_completion(self, prompt: str, temperature: float) -> dict[str, Any]:
        response = openai.ChatCompletion.create(
            model=self.model, messages=self._make_messages(prompt=prompt), temperature=temperature
        )
        return response  # pyright: ignore

    def chat(self, prompt: str, temperature: float = 0.0):
        response = self._get_completion(prompt=prompt, temperature=temperature)

        print(response.choices[0].message.content)  # pyright: ignore
