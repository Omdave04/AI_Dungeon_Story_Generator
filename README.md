from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import random
import os
from datetime import datetime

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

genre_prompts = {
    'Fantasy': "Once upon a time in a distant kingdom,",
    'Mystery': "It was a dark and stormy night when the detective arrived,",
    'Sci-Fi': "In the year 3021, humans colonized Mars and",
    'Adventure': "The jungle was thick and full of hidden dangers as",
    'Horror': "The door creaked open, revealing a shadow in the hallway."
}

def generate_story(prompt, max_length=100, num_return_sequences=3):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

import streamlit as st

st.title("AI Dungeon Story Generator")
genre = st.selectbox("Choose a Genre", list(genre_prompts.keys()))
user_input = st.text_input("Enter your story beginning (or leave empty to use genre default)")

if user_input.strip() == "":
    prompt = genre_prompts[genre]
else:
    prompt = user_input.strip()

if st.button("Generate Story"):
    stories = generate_story(prompt)
    for i, story in enumerate(stories):
        st.subheader(f"Story {i+1}")
        st.write(story)

    if st.button("Save Stories to File"):
        folder = "generated_stories"
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{folder}/story_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for i, story in enumerate(stories):
                f.write(f"Story {i+1}\n{story}\n\n")
        st.success(f"Stories saved to {filename}")
