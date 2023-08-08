from langchain import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader

from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


hugging_face_token = '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'


# M: Hugging face setup
llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0, "max_length":64},
                     huggingfacehub_api_token=hugging_face_token)  # can use gpt2 also as repo_id, google/flan-t5-xl, flan-t5-small /base


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
# TODO: try Sentence_Transformers for this too!
print("embeddings")
embeddings = HuggingFaceEmbeddings()

#query = "send me full list of conacts from contact list"
#embedding_vector = embeddings.embed_query(query)
#print(embedding_vector)


# create a vector store for these texts using the 'embeddings'
print("vector store")
doc_search = Chroma.from_documents(docs, embeddings)

# M: Using langchain chains to build qna: Incoming message is embedded, passed to the llm, which uses the vectorspace to find closely related words to the message
qna = RetrievalQA.from_chain_type(llm= llm, chain_type="stuff", retriever=doc_search.as_retriever())


query = "send me full list of conacts from contact list"
print(f"Query: {query}")
print("Answer:\n", qna.run(query))
pass