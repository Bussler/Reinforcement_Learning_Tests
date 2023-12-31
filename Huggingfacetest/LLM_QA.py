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
import transformers
import torch
from langchain.vectorstores import FAISS
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
hugging_face_token = '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'


#1) M: Hugging face setup: run on their server
#llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 0, "max_length":64},
#                     huggingfacehub_api_token=hugging_face_token)  # can use gpt2 also as repo_id, google/flan-t5-xl, flan-t5-small, base

#2) M: download and run llm locally with langchain wrapper. Watch out: is deprecated
#huggingface_hub.login(token=hugging_face_token)
#llm = HuggingFacePipeline.from_model_id(
#        #model_id="bigscience/bloom-560m", # Small model; stupid but works for testing
#        model_id="bigscience/bloom-1b7",
#        # model_id="meta-llama/Llama-2-7b-chat-hf", # Big model; only works on TERA
#        # model_id="google/flan-t5-small", # Small model, is not that good. Also need to change task to "text-generation", if we use this!
#        task="text-generation",
#        #task="text2text-generation",
#        model_kwargs={"temperature": 0, "max_length": 723},#, "use_auth_token": True},
#        #device=-1, # CPU
#        device=0, # GPU
#    )

#3) M: download and run llm locally with transformers pipeline
def setup_model_transformers():
    model_id="bigscience/bloom-560m"
    #model_id="bigscience/bloom-1b7"

    huggingface_hub.login(token=hugging_face_token)

    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        #use_auth_token=hugging_face_token
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        #trust_remote_code=True,
        config=model_config,
        #quantization_config=bnb_config,
        device_map=device,
        #use_auth_token=hugging_face_token
    )

    #tokenizer = transformers.AutoTokenizer.from_pretrained("bigscience/bloom-1b7", use_auth_token=hugging_face_token)
    #model = transformers.AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b7", device_map=device, use_auth_token=hugging_face_token)

    # enable evaluation mode to allow model inference
    model.eval()
    print(f"Model loaded on {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        #use_auth_token=hugging_face_token
    )

    stop_list = ['\nHuman:', '\n```\n']
    stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

    # define custom stopping criteria object
    class StopOnTokens(transformers.StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False

    stopping_criteria = transformers.StoppingCriteriaList([StopOnTokens()])

    generate_text = transformers.pipeline(
        model=model, 
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        stopping_criteria=stopping_criteria,  # without this model rambles during chat
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=64,  # max number of tokens to generate in the output
        #repetition_penalty=1.1  # without this output begins repeating -> M: for some reason, this doesn't work well with vector db!
    )

    #from transformers import pipeline
    #generate_text = pipeline("text-generation", model="bigscience/bloom-560m", max_new_tokens=723, 
    #                          device=0,)

    llm = HuggingFacePipeline(pipeline=generate_text)
    return llm

llm = setup_model_transformers()

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

#web_links = ["https://www.databricks.com/","https://help.databricks.com","https://databricks.com/try-databricks","https://help.databricks.com/s/","https://docs.databricks.com","https://kb.databricks.com/","http://docs.databricks.com/getting-started/index.html","http://docs.databricks.com/introduction/index.html","http://docs.databricks.com/getting-started/tutorials/index.html","http://docs.databricks.com/release-notes/index.html","http://docs.databricks.com/ingestion/index.html","http://docs.databricks.com/exploratory-data-analysis/index.html","http://docs.databricks.com/data-preparation/index.html","http://docs.databricks.com/data-sharing/index.html","http://docs.databricks.com/marketplace/index.html","http://docs.databricks.com/workspace-index.html","http://docs.databricks.com/machine-learning/index.html","http://docs.databricks.com/sql/index.html","http://docs.databricks.com/delta/index.html","http://docs.databricks.com/dev-tools/index.html","http://docs.databricks.com/integrations/index.html","http://docs.databricks.com/administration-guide/index.html","http://docs.databricks.com/security/index.html","http://docs.databricks.com/data-governance/index.html","http://docs.databricks.com/lakehouse-architecture/index.html","http://docs.databricks.com/reference/api.html","http://docs.databricks.com/resources/index.html","http://docs.databricks.com/whats-coming.html","http://docs.databricks.com/archive/index.html","http://docs.databricks.com/lakehouse/index.html","http://docs.databricks.com/getting-started/quick-start.html","http://docs.databricks.com/getting-started/etl-quick-start.html","http://docs.databricks.com/getting-started/lakehouse-e2e.html","http://docs.databricks.com/getting-started/free-training.html","http://docs.databricks.com/sql/language-manual/index.html","http://docs.databricks.com/error-messages/index.html","http://www.apache.org/","https://databricks.com/privacy-policy","https://databricks.com/terms-of-use"] 
#loader = WebBaseLoader(web_links)
#documents = loader.load()

# M: split the parsed documents into smaller chunks
print("doing text splitting")
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=20)
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
#doc_search = Chroma.from_documents(docs, embeddings, persist_directory='chroma_db')

doc_search = FAISS.from_documents(docs, embeddings)

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

#query = "what is gratuity?"
query = "give me all phone numbers from contacts"
#query = "What is Data lakehouse architecture in Databricks?"
chat_history = []
print("Query: ", query)
result = qna({"question": query, "chat_history": chat_history})
print(result['answer'])
pass
