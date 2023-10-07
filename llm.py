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
    prefix = "In one English sentence, generate a visually descriptive image caption based on this line: \"{}\""
    prompts = [(start, end, prompt_chatgpt(prefix.format(line)).split("\n")[0]) for start, end, line in tqdm.tqdm(lyrics)]
    print(prompts)
    return prompts


def mux_to_prompts(mux):
    print("Prompting LLM")
    prefix = "In one English sentence, generate a visually descriptive image caption with moderately {} arousal and moderately {} valence based on this line: \"{}\""
    prompts = [(start, end, prompt_chatgpt(prefix.format(arousal, valence, lyric)).split("\n")[0]) for start, end, lyric, (arousal, valence) in tqdm.tqdm(mux)]
    # if lyric is not None else (start, end, default)
    print(prompts)
    return prompts
