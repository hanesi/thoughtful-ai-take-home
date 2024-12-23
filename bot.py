import os
import warnings
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import JSONLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.llms import HuggingFaceTextGenInference
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import HuggingFaceTextGenInference
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat


warnings.filterwarnings("ignore")
load_dotenv()
chat_history = []

def chat_bot():
    loader = JSONLoader(
        "data/questions.json",
        # jq_schema='.questions[] | {page_content: (.question + " Answer: " + .answer), metadata: {}}'
        jq_schema=".questions[]",
        content_key=".answer",
        is_content_key_jq_parsable=True,
        # text_content=False
    )
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    template_messages = [
        SystemMessage(content="You are a customer support AI Agent to assist users with basic questions about Thoughtful AI."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(template_messages)

    llm = HuggingFaceTextGenInference(
        inference_server_url="http://127.0.0.1:8080/",
        max_new_tokens=512,
        top_k=50,
        temperature=0.1,
        repetition_penalty=1.03,
    )

    # embeddings = OllamaEmbeddings(
    #     model="llama3",
    # )
    # db = Chroma.from_documents(texts, embeddings)

    # vectorstore = db.as_retriever()

    model = Llama2Chat(llm=llm)

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        # retriever=vectorstore
    )
    chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)


    question = input("Hi! How can I help you today? ")
    

    # chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")

    # qa = ConversationalRetrievalChain.from_llm(
    #     llm=model, chain_type="stuff", retriever=vectorstore
    # )

    qa = chain.run(text=question)
    print("\n", qa, "\n")

    # while question != "quit":

    #     res = qa
    #     print("\n", res, "\n")
    #     question = input("Is there anything else I can help with? ")
    #     qa = chain.run(text=question)

if __name__ == "__main__":
    chat_bot()