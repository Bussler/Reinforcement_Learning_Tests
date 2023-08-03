from langchain.embeddings import HuggingFaceHubEmbeddings

hugging_face_token = "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"

repo_id = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceHubEmbeddings(
    repo_id=repo_id,
    task="text-generation", # "text-generation", "feature-extraction"
    huggingfacehub_api_token=hugging_face_token,
)