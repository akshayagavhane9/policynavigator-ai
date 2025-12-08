import os
import openai
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def run(self, prompt):
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
