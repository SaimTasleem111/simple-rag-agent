from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

#load and read from pdf
load_dotenv()
loader = PyPDFLoader('Bella Vista Restaurant Menu.pdf')
docs=loader.load()
print(len(docs))

#split the text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_data=splitter.split_documents(docs)
print(len(splitted_data))

#embeddings model
embeddings=OpenAIEmbeddings(model="text-embedding-3-large")

#store to vector database
vector_store = Chroma.from_documents(
    documents=splitted_data,
    embedding=embeddings
)

#llm model
llm = ChatOpenAI(model="gpt-4o-mini")

# get context for question provided by user
def get_context(query):
    # searching the data by similarity
    data = vector_store.similarity_search(query=query)
    context=""
    for doc in data:
        context+=doc.page_content + "\n"

    return {
        "context": context,
        "question": query,
    }

#prompt
prompt=PromptTemplate.from_template(""" 
You are a helpful assistant and provide answers based on the context for us 
if you don't know the answer then you can say 'I don't know.'

Context: {context}
Question: {question}
    """
)

rag_chain= get_context | prompt | llm

res= rag_chain.invoke('whats your opening hours')

print(res.content)
