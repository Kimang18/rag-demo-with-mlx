from mlx_embeddings import load as em_load
from mlx_embeddings import generate as em_generate
import mlx.core as mx


class LocalEmbeddings(object):
    def __init__(self):
        self.model, self.tokenizer = em_load("mlx-community/all-MiniLM-L6-v2-bf16")
    def embed_query(self, query: str) -> list[float]:
        outputs = em_generate(self.model, self.tokenizer, query)
        return outputs.text_embeds[0].tolist()
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        outputs = em_generate(self.model, self.tokenizer, texts)
        return [text_embeds.tolist() for text_embeds in outputs.text_embeds]

#%%
from langchain_community.document_loaders import TextLoader
file_path = "./example.md"
documents = TextLoader(file_path).load()

#%%
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=len
)
chunks = splitter.split_documents(documents)
#%%
from langchain_chroma import Chroma

vector_store = Chroma(embedding_function=LocalEmbeddings())
ids = vector_store.add_documents(chunks)
#%%
vector_store = vector_store.as_retriever(search_kwargs={'k': 3})

query = "How does Alice meet Mad Hatter?"
results = vector_store.invoke(query)
#%%
context = "\n\n---\n\n".join([doc.page_content for doc in results])
prompt = f"""Answer the question based only on the following context:
---
{context}
---
Carefully study the above context to answer the following question: {query}
"""

#%%
from mlx_lm import load as llm_load
from mlx_lm import generate as llm_generate


model, tokenizer = llm_load("mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510")  #"mlx-community/Llama-3.2-3B-Instruct-4bit")

messages = [{'role': 'user', 'content': prompt}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
text = llm_generate(model, tokenizer, prompt=prompt)
sources = [doc.metadata['source'] for doc in results]

print(f'Answer: {text}\nSources: {sources}')
#%%
