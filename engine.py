import openai
from api_secrets import get_api_key
from time import sleep
import tiktoken


openai.api_key = get_api_key()

length_limit = {
    'text-davinci-003': 4096,
    'text-curie-001': 2048,
    'text-babbage-001': 2048,
    'text-ada-001': 2048,
}

class Engine:
    def __init__(self, engine='text-davinci-003'):
        self.engine = engine
        self.tokenizer = tiktoken.encoding_for_model(engine)

    def check_prompt_length(self, prompt, max_tokens=64):
        prompt_length = len(self.tokenizer.encode(prompt))
        if prompt_length + max_tokens >= length_limit[self.engine]:  # Prompt is too long
            return True
        return False

    def complete(self, prompt, max_tokens=64):
        num_retry = 0
        while True:
            try:
                response = openai.Completion.create(
                    engine=self.engine,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                print(e)
                if num_retry >= 5:  # Retried too many times
                    print('Retried too many times, skip this instance.')
                    return None
                sleep(2)
                num_retry += 1
                continue
            break
        answer = response.choices[0].text
        return answer

    def get_prob(self, prompt, num_tokens):
        num_retry = 0
        while True:
            try:
                response = openai.Completion.create(
                    engine=self.engine,
                    prompt=prompt,
                    max_tokens=0,
                    logprobs=1,
                    echo=True,
                )
                token_logprobs = response.choices[0].logprobs.token_logprobs[-num_tokens:]
                seq_prob = sum(token_logprobs)
            except Exception as e:
                print(e)
                if num_retry >= 5:  # Retried too many times
                    print('Retried too many times, skip this instance.')
                    return None
                sleep(2)
                num_retry += 1
                continue
            break
        return seq_prob
