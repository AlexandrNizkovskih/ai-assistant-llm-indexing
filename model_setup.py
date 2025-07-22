"""Model and index configuration helpers."""

import os
import torch
from huggingface_hub import login
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from llama_index.core import SimpleDirectoryReader, Document, GPTVectorStoreIndex, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings

from preprocess import preprocess_document


def authenticate(hf_token: str, openai_key: str) -> None:
    """Authenticate with HuggingFace and set the OpenAI key."""
    login(hf_token, add_to_git_credential=True)
    os.environ["OPENAI_API_KEY"] = openai_key


def load_llm(model_name: str = "IlyaGusev/saiga_mistral_7b") -> HuggingFaceLLM:
    """Load Saiga model with quantization and LoRA weights."""
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    config = PeftConfig.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, model_name, torch_dtype=torch.float16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    generation_config = GenerationConfig.from_pretrained(model_name)

    return HuggingFaceLLM(
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        max_new_tokens=generation_config.max_new_tokens,
        model_kwargs={"quantization_config": quant_config},
        generate_kwargs={
            "bos_token_id": generation_config.bos_token_id,
            "eos_token_id": generation_config.eos_token_id,
            "pad_token_id": generation_config.pad_token_id,
            "no_repeat_ngram_size": generation_config.no_repeat_ngram_size,
            "repetition_penalty": generation_config.repetition_penalty,
            "temperature": generation_config.temperature,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
        },
        device_map="auto",
    )


def setup_service_context(llm: HuggingFaceLLM) -> None:
    """Configure LlamaIndex global settings for the provided LLM."""
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512


def build_index(data_path: str) -> GPTVectorStoreIndex:
    """Load and preprocess documents, then build a vector index."""
    documents = [
        preprocess_document(doc)
        for doc in SimpleDirectoryReader(data_path).load_data()
        if doc
    ]
    documents = [doc for doc in documents if doc is not None]
    return GPTVectorStoreIndex.from_documents(documents)
