from mlx_embeddings import load as em_load
from mlx_embeddings import generate as em_generate
from mlx_lm import load as llm_load
from mlx_lm import generate as llm_generate
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import streamlit as st
import tempfile
import os
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler


class LLMWrapper(object):
    def __init__(self):
        self.model, self.tokenizer = llm_load("mlx-community/Llama-3.2-3B-Instruct-4bit")
        self.prompt_cache = make_prompt_cache(self.model, None)
        self.sampler = make_sampler(
            0.0, 1.0, xtc_threshold=0.0, xtc_probability=0.0,
            xtc_special_tokens=(
                self.tokenizer.encode("\n")+list(self.tokenizer.eos_token_ids)
            )
        )

    def __call__(self, prompt: str) -> str:
        messages = [{'role': 'user', 'content': prompt}]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return llm_generate(self.model, self.tokenizer, prompt=prompt, prompt_cache=self.prompt_cache, sampler=self.sampler)


class LocalEmbeddings(object):
    def __init__(self):
        self.model, self.tokenizer = em_load("mlx-community/all-MiniLM-L6-v2-bf16")
    def embed_query(self, query: str) -> list[float]:
        outputs = em_generate(self.model, self.tokenizer, query)
        return outputs.text_embeds[0].tolist()
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        outputs = em_generate(self.model, self.tokenizer, texts)
        return [text_embeds.tolist() for text_embeds in outputs.text_embeds]


def get_chunks(file_path, ext):
    if ext == '.md':
        documents = TextLoader(file_path).load()
    elif ext == '.pdf':
        documents = PyPDFLoader(file_path).load()
    if ext == '.csv':
        documents = CSVLoader(file_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_documents(documents)


def get_retriever(chunks, persist_dir):
    vector_store = Chroma(embedding_function=st.session_state.embedder, persist_directory=persist_dir)
    ids = vector_store.add_documents(chunks)
    return vector_store.as_retriever(search_kwargs={'k': 3})


def save_tmp_file(md_file):
    ext = os.path.splitext(md_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(md_file.getvalue())
        return tmp_file.name


def handle_md_upload(md_file):
    if st.session_state.md_file != md_file.name:
        with st.spinner("Processing your markdown file..."):
            st.session_state.md_file = md_file.name
            file_path = save_tmp_file(md_file)
            root, ext = os.path.splitext(file_path)
            chunks = get_chunks(file_path, ext)
            st.session_state.retriever = get_retriever(chunks, root)
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
            context = "\n\n".join([f"<context>\n{doc.page_content}\n</context>" for doc in results])
            prompt = f"""Use only the information enclosed in <context> tags to answer the question enclosed in <question> tags:

{context}

<question>
{query}
</question>
"""
            print(prompt)
            answer = st.session_state.llm(prompt)
            sources = [doc.metadata['source'] for doc in results]
            st.write(f"Answer: {answer}\nSources: {sources}")
            st.session_state.messages.append({'role': 'assistant', 'content': answer})

st.title("Chat with your markdown file")

md_file = st.file_uploader("Please upload your markdown file here", type=['md', 'pdf', 'csv'])

if 'md_file' not in st.session_state:
    st.session_state['md_file'] = ''
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None
if 'llm' not in st.session_state:
    st.session_state['llm'] = LLMWrapper()
if 'embedder' not in st.session_state:
    st.session_state['embedder'] = LocalEmbeddings()
if 'messages' not in st.session_state:
    st.session_state['messages'] = []



if md_file:
    #st.info("Uploaded file")
    handle_md_upload(md_file)
else:
    st.info("Please upload a markdown file to start chatting...")
