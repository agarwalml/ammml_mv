import os
import openai
import tqdm


openai.api_key = os.getenv("OPENAI_API_KEY")


def prompt_chatgpt(prompt):
    response = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=200)
    text = response["choices"][0]["text"].strip()
    return text


def lyrics_to_prompts(lyrics):
    print("Prompting LLM")
    prefix = "Convert to a detailed image description: "
    prompts = [(start, end, prompt_chatgpt(prefix + line)) for start, end, line in tqdm.tqdm(lyrics)]
    return prompts
