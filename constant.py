from chromadb.config import Settings
PERSIST_DIRECTORY="db"
MODEL_TYPE="GPT4All"
MODEL_TYPE_VM="huggingface"
MODEL_PATH="models/ggml-gpt4all-j-v1.3-groovy.bin"
# MODEL_PATH_VM="./models/mpt-7b/"
MODEL_PATH_VM="ai-forever/mGPT"
MODEL_NAME_VM= "gpt2" #"ai-forever/mGPT"  #
EMBEDDINGS_MODEL_NAME="all-MiniLM-L6-v2"
EMBEDDINGS_MODEL_NAME_VM="hkunlp/instructor-xl"
MODEL_N_CTX="1000"
DOC_DIR="docs"

CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)


#TheBloke/vicuna-7B-1.1-HF