import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
load_dotenv()

if __name__=="__main__":
    print("Retrieving")
    query="What is pinecone?"
    embeddings=OpenAIEmbeddings()
    llm=ChatOpenAI()

    vectorstore= PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"],embedding=embeddings
    )

    retrival_qa_chat_prompt=hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain=create_stuff_documents_chain(llm,retrival_qa_chat_prompt)
    retreival_chain=create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    result=retreival_chain.invoke(input={"input":query})

    print(result)