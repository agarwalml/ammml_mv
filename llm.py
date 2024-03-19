import os
import openai
import tqdm


openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_quotes(text):
    quotes = []
    lines = text.split('\n')
    for line in lines:
        quotes.append(line.split('.')[1])
    return quotes

def prompt_chatgpt(prompt):
    completion = openai.ChatCompletion.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    )
    text = completion.choices[0].message.content.strip()
    return text
    # response = openai.Completion.create(model="gpt-4-turbo-preview", prompt=prompt, max_tokens=200)
    # text = response["choices"][0]["text"].strip()
    # return text


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

def mux_to_narrative_prompts(mux):
    print("Prompting LLM")
    prefix = "Create a narrative for a music video based on the following lyrics with the given arousal and given valence. For each line, generate a visually descriptive image caption based on the story. The image caption should be self contained and not reference any other caption but follow the same story. Display just the line numbers and the generated caption:"
    # generate prompts with line number
    prompts = [(lyric, valence, arousal) for start, end, lyric, (arousal, valence) in mux]
    # generate prompt string with line number for each prompt
    prompt_str = "\n".join([f"{i+1}.\"{prompts[i][0]}\" with moderately {prompts[i][1]} arousal and moderately {prompts[i][2]} valence" for i in range(len(prompts))])

    response = prompt_chatgpt(prefix + prompt_str)

    # split response into lines
    responses = extract_quotes(response)

    if not (len(responses) == len(mux)):
        print ("Blasphemy")
        
    final_list = [(mux[i][0], mux[i][1], responses[i]) for i in range(len(mux))]

    print(final_list)

