import os

class PromptBuilder:
    def __init__(self):
        base = os.path.dirname(__file__)
        self.system_prompt = self._load(os.path.join(base, "prompts/system_prompt.txt"))
        self.answer_prompt = self._load(os.path.join(base, "prompts/answer_prompt.txt"))
        self.rewrite_prompt = self._load(os.path.join(base, "prompts/rewrite_prompt.txt"))
        self.summarizer_prompt = self._load(os.path.join(base, "prompts/summarizer_prompt.txt"))

    def _load(self, path):
        with open(path, "r") as f:
            return f.read()

    def build_answer_prompt(self, question, context):
        return self.answer_prompt.replace("{{question}}", question)\
                                .replace("{{context}}", context)

    def build_rewrite_prompt(self, question):
        return self.rewrite_prompt.replace("{{question}}", question)

    def build_summarizer_prompt(self, history):
        return self.summarizer_prompt.replace("{{history}}", history)
