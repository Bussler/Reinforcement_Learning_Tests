from langchain import HuggingFaceHub
import huggingface_hub
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader

from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.llms import HuggingFacePipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import ConversationalRetrievalChain


hugging_face_token = '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'


# M: Hugging face setup
#llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 0, "max_length":64},
#                     huggingfacehub_api_token=hugging_face_token)  # can use gpt2 also as repo_id, google/flan-t5-xl, flan-t5-small, base

# M: download and run llm locally
#huggingface_hub.login(token=hugging_face_token)
llm = HuggingFacePipeline.from_model_id(
        model_id="bigscience/bloom-560m", # Small model; stupid but works for testing
        # model_id="meta-llama/Llama-2-7b-chat-hf", # Big model; only works on TERA
        # model_id="google/flan-t5-small", # Small model, is not that good. Also need to change task to "text-generation", if we use this!
        task="text-generation",
        #task="text2text-generation",
        model_kwargs={"temperature": 0, "max_length": 300},#, "use_auth_token": True},
        device=-1, # CPU
        # device=0, # GPU
    )

#print(llm("translate to German: How are you"))

#template =  "Question: {question}"
#prompt = PromptTemplate(template=template, input_variables=["question"])
#question = "Who started ecomomic liberliation in India and what was the imnpact of it on India economically?"
#llm_chain=LLMChain(prompt=prompt,llm=llm)
#print(llm_chain.run(question))

# M: document loaders: parse in the docs
print("parse in text documents")
loader = DirectoryLoader("txt_data/", glob="**/*.txt" )
documents = loader.load()

# M: split the parsed documents into smaller chunks
print("doing text splitting")
text_splitter = CharacterTextSplitter(chunk_size = 100, chunk_overlap=0)
docs = text_splitter.split_documents(documents=documents)
#docs = text_splitter.create_documents(docs) # M: use this for string input, e.g. from file.read()

# M: Embedder for the vector store to transform the text into vector space
print("embeddings")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") # M: equivalent!

#query = "send me full list of conacts from contact list"
#embedding_vector = embeddings.embed_query(query)
#print(embedding_vector)


# create a vector store for these texts using the 'embeddings'
print("vector store")
doc_search = Chroma.from_documents(docs, embeddings, persist_directory='chroma_db')

# M: Using langchain chains to build qna: Incoming message is embedded, passed to the llm, which uses the vectorspace to find closely related words to the message
#qna = RetrievalQA.from_chain_type(llm= llm, chain_type="stuff", retriever=doc_search.as_retriever())
#query = "send me full list of conacts from contact list"
#print("Query: ", query)
#print(qna.run(query))

# M: build chat history to take into account
qna = ConversationalRetrievalChain.from_llm(
        llm, 
        doc_search.as_retriever(), 
        return_source_documents=True
    )

query = "send me full list of conacts from contact list"
chat_history = []
print("Query: ", query)
result = qna({"question": query, "chat_history": chat_history})
print(result['answer'])
pass
