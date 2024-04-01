import os

import requests


class APIClient:
    def generate_response(self, model, system_prompt, user_prompt, *args):
        api_provider = os.getenv("API_PROVIDER")
        if api_provider == "OpenAI":
            api_key = os.getenv("OPENAI_API_KEY")
            headers = {"Authorization": f"Bearer {api_key}"}
            data = {
                "model": model,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}] + [{"role": "assistant" if i % 2 == 0 else "user", "content": arg} for i, arg in enumerate(args)],
                "max_tokens": os.getenv("DEFAULT_MAX_TOKENS", 150),
                "temperature": 0,
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
            return response.json()["choices"][0]["message"]["content"]
        elif api_provider == "HuggingFace":
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            headers = {"Authorization": f"Bearer {api_key}"}
            payload = {
                "inputs": {
                    "text": [system_prompt, user_prompt] + list(args),
                    "parameters": {"model": model, "max_length": int(os.getenv("DEFAULT_MAX_TOKENS", 150))}
                }
            }
            response = requests.post("https://api-inference.huggingface.co/models/gpt", json=payload, headers=headers)
            return response.json()["generated_text"]
