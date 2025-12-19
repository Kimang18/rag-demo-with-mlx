import streamlit as st
from mlx_lm import load, generate
from pandasai.llm import LLM
import pandasai as pdai


class LLMWrapper:
    def __init__(self):
        # self.model, self.tokenizer = load('mlx-community/Qwen2.5-Coder-14B-Instruct-4bit')
        self.model, self.tokenizer = load("mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510")

    def __call__(self, prompt: str):
        messages = [{'role': 'user', 'content': prompt}]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        text = generate(self.model, self.tokenizer, prompt)
        return text


class LocalLLM(LLM):
    def call(self, instruction, context):
        return st.session_state.local_llm(instruction.to_string())

    @property
    def type(self):
        return 'local'

llm = LocalLLM()

pdai.config.set({'llm': llm})

st.title("Chat with your Dataframe")

csv_file = st.file_uploader("Upload your csv file here", type=['csv'])


if 'local_llm' not in st.session_state:
    st.session_state['local_llm'] = LLMWrapper()

if csv_file:
    df = pdai.read_csv(csv_file.name)
    st.write(df.head(3))
    prompt = st.text_area("Enter your command here")
    if st.button("Generate"):
        if prompt:
            # st.write("Your prompt is entered")
            with st.spinner("Generating response"):
                response = df.chat(prompt)
                st.write(response)
        else:
            st.write("Please enter your command!")
