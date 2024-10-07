from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI


class Agent:
    def __init__(self, name, client, prompt, model="gpt-4o", temperature=0.2):
        self.name = name
        self.client = client
        self.system_prompt = prompt
        self.gen_kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": 1000,
        }
        self.message_history = [{"role": "system", "content": self.system_prompt}]

    @observe
    async def generate_response(self, user_message):
        if isinstance(user_message, dict):
            self.message_history.append(user_message)
        else:
            self.message_history.append({"role": "user", "content": user_message})

        stream = await self.client.chat.completions.create(
            messages=self.message_history, stream=True, **self.gen_kwargs
        )

        response_message = ""
        async for part in stream:
            if token := part.choices[0].delta.content or "":
                response_message += token
                yield token

        self.message_history.append({"role": "assistant", "content": response_message})

    def get_message_history(self):
        return self.message_history

    def clear_message_history(self):
        self.message_history = [{"role": "system", "content": self.system_prompt}]
