#!/usr/bin/env python3
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from transformers import LlamaTokenizer, T5Tokenizer, AutoTokenizer,AutoModelForQuestionAnswering,BloomModel, T5ForConditionalGeneration, LlamaForCausalLM, GenerationConfig, pipeline, AutoModel, AutoModelForCausalLM

import os
import argparse
from constant import (MODEL_TYPE, MODEL_TYPE_VM, PERSIST_DIRECTORY, DOC_DIR, MODEL_PATH, MODEL_NAME_VM,
                      MODEL_PATH_VM, EMBEDDINGS_MODEL_NAME, EMBEDDINGS_MODEL_NAME_VM, MODEL_N_CTX, CHROMA_SETTINGS)
import pdb
from langchain import HuggingFaceHub
from langchain.llms import HuggingFacePipeline
# mosaicml/mpt-7b
embeddings_model_name =EMBEDDINGS_MODEL_NAME_VM
# embeddings_model_name = EMBEDDINGS_MODEL_NAME
persist_directory = PERSIST_DIRECTORY

# model_type = MODEL_TYPE
model_type = MODEL_TYPE_VM
model_path = MODEL_PATH
# model_path = MODEL_PATH_VM
model_n_ctx = MODEL_N_CTX

model_name = MODEL_NAME_VM

MODEL_PARAMS_MAPPING = {

    "google/flan-t5-small": {"task":"text2text-generation", "token": T5Tokenizer, "llm": T5ForConditionalGeneration #LlamaForCausalLM  # T5ForConditionalGeneration
                             },
    "deepset/roberta-base-squad2" :{"task":"question-answering", 
                                    "token": AutoTokenizer, 
                                    "llm": AutoModelForQuestionAnswering 
                                    #LlamaForCausalLM  # T5ForConditionalGeneration
                             },
    "bigscience/bloom-560m" :{"task":"text-generation", 
                                    "token": AutoTokenizer, 
                                    "llm": BloomModel 
                                    #LlamaForCausalLM  # T5ForConditionalGeneration
                             },

}


def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model_name, cache_folder="./models")
    db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx,
                           callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx,
                          backend='gptj', callbacks=callbacks, verbose=False)
        case "huggingface":
            # llm = AutoModel.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
            llm = load_model()
            print("Model downloaded")
        case _default:
            print(f"Model {model_type} not supported!")
            exit
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        res = qa(query)
        answer, docs = res['result'], [
        ] if args.hide_source else res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)


def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


def load_model():
    '''
    Select a model on huggingface. 
    If you are running this for the first time, it will download a model for you. 
    subsequent runs will use the model from the disk. 
    '''
    model_id = model_name
    tokenizer = MODEL_PARAMS_MAPPING[model_name]["token"].from_pretrained(model_id)

    model = MODEL_PARAMS_MAPPING[model_name]["llm"].from_pretrained(model_id,
                                                             #   load_in_8bit=True, # set these options if your GPU supports them!
                                                             #   device_map=1#'auto',
                                                             #   torch_dtype=torch.float16,
                                                             #   low_cpu_mem_usage=True
                                                             )

    pipe = pipeline(
       MODEL_PARAMS_MAPPING[model_name]["task"],
        model=model,
        tokenizer=tokenizer,
        # max_length=2048,
        # temperature=0,
        # top_p=0.95,
        # repetition_penalty=1.15
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm


if __name__ == "__main__":
    main()
