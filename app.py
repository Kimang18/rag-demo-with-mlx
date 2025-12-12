from mlx_embeddings import load as em_load
from mlx_embeddings import generate as em_generate
from mlx_lm import load as llm_load
from mlx_lm import generate as llm_generate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import streamlit as st
import tempfile
import os


class LLMWrapper(object):
    def __init__(self):
        self.model, self.tokenizer = llm_load("mlx-community/Llama-3.2-3B-Instruct-4bit")

    def __call__(self, prompt: str) -> str:
        messages = [{'role': 'user', 'content': prompt}]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return llm_generate(self.model, self.tokenizer, prompt=prompt)


class LocalEmbeddings(object):
    def __init__(self):
        self.model, self.tokenizer = em_load("mlx-community/all-MiniLM-L6-v2-bf16")
    def embed_query(self, query: str) -> list[float]:
        outputs = em_generate(self.model, self.tokenizer, query)
        return outputs.text_embeds[0].tolist()
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        outputs = em_generate(self.model, self.tokenizer, texts)
        return [text_embeds.tolist() for text_embeds in outputs.text_embeds]


def get_chunks(file_path):
    documents = TextLoader(file_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_documents(documents)


def get_retriever(chunks):
    vector_store = Chroma(embedding_function=LocalEmbeddings())
    ids = vector_store.add_documents(chunks)
    return vector_store.as_retriever(search_kwargs={'k': 3})


def save_tmp_file(md_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.md') as tmp_file:
        tmp_file.write(md_file.getvalue())
        return tmp_file.name


def handle_md_upload(md_file):
    if st.session_state.md_file != md_file.name:
        with st.spinner("Processing your markdown file..."):
            st.session_state.md_file = md_file.name
            file_path = save_tmp_file(md_file)
            chunks = get_chunks(file_path)
            st.session_state.retriever = get_retriever(chunks)
            st.session_state.llm = LLMWrapper()
            st.session_state.messages = []
            os.unlink(file_path)
            st.success("Successfully processing markdown file")
    display_messages()
    handle_user_input()

def display_messages():
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['content'])

def handle_user_input():
    if query := st.chat_input("Ask questions about your markdown file"):
        with st.chat_message('user'):
            st.write(query)
            st.session_state.messages.append({'role': 'user', 'content': query})
        with st.chat_message('assistant'):
            results = st.session_state.retriever.invoke(query)
            context = "\n\n---\n\n".join([doc.page_content for doc in results])
            prompt = f"""Answer the question based only on the following context:

{context}

Answer the following question based only on the above context:{query}
"""
            answer = st.session_state.llm(prompt)
            sources = [doc.metadata['source'] for doc in results]
            st.write(f"Answer: {answer}\nSources: {sources}")
            st.session_state.messages.append({'role': 'assistant', 'content': answer})

st.title("Chat with your markdown file")

md_file = st.file_uploader("Please upload your markdown file here", type=['md', 'MD', 'markdown', 'rmd'])

if 'md_file' not in st.session_state:
    st.session_state['md_file'] = ''
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None
if 'llm' not in st.session_state:
    st.session_state['llm'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []



if md_file:
    #st.info("Uploaded file")
    handle_md_upload(md_file)
else:
    st.info("Please upload a markdown file to start chatting...")
