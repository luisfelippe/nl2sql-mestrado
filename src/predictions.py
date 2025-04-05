"""Script responsável por gerar as predições do modelo."""

import json
import os
import time
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
import torch
from tqdm import tqdm

from unsloth import FastLanguageModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from dotenv import load_dotenv

# from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.dataset_loader import DatasetLoader
from src.env import OUTPUT_COMMERCIAL_MODEL_PREDICTIONS, OUTPUT_MODEL
from src.schemas import PredictionModel

# Load environment variables from the .env file (if present)
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODELS = {
    "gemini-2.0-flash": {
        "model": "gemini-2.0-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": GEMINI_API_KEY,
        "temperature": 0,
        "max_completition_tokens": 512,
        "local": False,
        "zero_shot": True,
        "model_type": "llm",
    },
    "gemini-1.5-flash": {
        "model": "gemini-1.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": GEMINI_API_KEY,
        "temperature": 0,
        "max_completition_tokens": 512,
        "local": False,
        "zero_shot": True,
        "model_type": "llm",
    },
    "gemini-1.5-pro": {
        "model": "gemini-1.5-pro",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": GEMINI_API_KEY,
        "temperature": 0,
        "max_completition_tokens": 512,
        "local": False,
        "zero_shot": True,
        "model_type": "llm",
    },
    "gemma-3-27b-it": {
        "model": "gemma-3-27b-it",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": GEMINI_API_KEY,
        "temperature": 0,
        "max_completition_tokens": 512,
        "local": False,
        "zero_shot": True,
        "model_type": "llm",
    },
    # TODO modelos da openai
    # TODO modelos lhama

    ## huggingface
    # "ibm-granite/granite-3.2-8b-instruct": {
    "unsloth/granite-3.2-8b-instruct-bnb-4bit": {
        "model": "unsloth/granite-3.2-8b-instruct-bnb-4bit",
        "base_url": None,
        "api_key": None,
        "temperature": 0,
        "max_completition_tokens": 512,
        "local": True,
        "zero_shot": True,
        "model_type": "llm",
    },
    "unsloth/gemma-3-1b-it-bnb-4bit": {
        "model": "unsloth/gemma-3-1b-it-bnb-4bit",
        "base_url": None,
        "api_key": None,
        "temperature": 0,
        "max_completition_tokens": 512,
        "local": True,
        "zero_shot": True,
        "model_type": "llm",
    },
    "unsloth/gemma-3-4b-it-bnb-4bit": {
        "model": "unsloth/gemma-3-4b-it-bnb-4bit",
        "base_url": None,
        "api_key": None,
        "temperature": 0,
        "max_completition_tokens": 512,
        "local": True,
        "zero_shot": True,
        "model_type": "llm",
    },
    "unsloth/gemma-3-12b-it-bnb-4bit": {
        "model": "unsloth/gemma-3-12b-it-bnb-4bit",
        "base_url": None,
        "api_key": None,
        "temperature": 0,
        "max_completition_tokens": 512,
        "local": True,
        "zero_shot": True,
        "model_type": "llm",
    },
    "unsloth/gemma-3-27b-it-bnb-4bit": {
        "model": "unsloth/gemma-3-27b-it-bnb-4bit",
        "base_url": None,
        "api_key": None,
        "temperature": 0,
        "max_completition_tokens": 512,
        "local": True,
        "zero_shot": True,
        "model_type": "llm",
    },
    "unsloth/gemma-2-2b-it-bnb-4bit": {
        "model": "unsloth/gemma-2-2b-it-bnb-4bit",
        "base_url": None,
        "api_key": None,
        "temperature": 0,
        "max_completition_tokens": 512,
        "local": True,
        "zero_shot": True,
        "model_type": "llm",
    },
    "unsloth/gemma-2-9b-it-bnb-4bit": {
        "model": "unsloth/gemma-2-9b-it-bnb-4bit",
        "base_url": None,
        "api_key": None,
        "temperature": 0,
        "max_completition_tokens": 512,
        "local": True,
        "zero_shot": True,
        "model_type": "llm",
    },
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit": {
        "model": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "base_url": None,
        "api_key": None,
        "temperature": 0,
        "max_completition_tokens": 512,
        "local": True,
        "zero_shot": True,
        "model_type": "llm",
    },
    "ft-seq2seq": {
        "model": "ft-seq2seq",
        "base_url": None,
        "api_key": None,
        "temperature": 0,
        "max_completition_tokens": 512,
        "local": True,
        "zero_shot": False,
        "model_type": "seq2seq",
    },
    "ft-causal": {
        "model": None,
        "base_url": None,
        "api_key": None,
        "temperature": 0,
        "max_completition_tokens": 512,
        "local": True,
        "zero_shot": False,
        "model_type": "causal",
    },
}

# TODO Gemma não aceita sistem prompt
# reacall alto signinifca pouca precisão
# TODO usar pelo langchain os modelos gemma 2 e 9b e lhama 1, 3 e 8b pelo huggingface


# class Predictions:
#     """Classe responsável por gerar as predições do modelo."""

#     def __init__(self, model: dict, model_path=None):
#         # Definir dispositivo: GPU se disponível, caso contrário CPU
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model_data = model
#         self.model_path = model_path

#         self.dataset_loader = DatasetLoader()

#     def _load_model(self):
#         # Carregar o tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)

#         if self.model_data["model_type"] == "seq2seq" and model_path:
#             # Carregar o modelo
#             self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
#         else:
#             self.model = AutoModelForCausalLM.from_pretrained(model_path)

#         self.model = self.model.to(self.device)

#         pipe = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             max_new_tokens=self.model_data["max_completition_tokens"],
#             device_map="auto",
#         )

#         self.model = HuggingFacePipeline(pipeline=pipe)

#     def use_ajusted_model(self, text):
#         """Gerar predições do modelo."""

#         prompt = ChatPromptTemplate.from_template(text)

#         # Criar o pipeline de predição
#         chain = prompt | self.model | StrOutputParser()

#         saida = chain.invoke({})

#         return saida


class Predicions:
    """Classe responsável por gerar as predições com modelos comerciais."""

    def __init__(self, model: dict, model_path=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(
            f"Dispositivo utilizado: {self.device}.\nModelo: {model['model']}.\nTipo de modelo: {model['model_type']}."
        )

        self.dataset_loader = DatasetLoader()

        self.model_data = model
        self.model = None
        self.model_path = model_path
        self.tokenizer = None

    def _load_model(self):
        if not self.model_data["local"]:
            # subistituir pala classe OpenAI do langchain só alterando a url da API e nome do modelo
            self.model = ChatOpenAI(
                base_url=self.model_data["base_url"],
                api_key=self.model_data["api_key"],
                model=self.model_data["model"],
                temperature=self.model_data["temperature"],
                verbose=True,
                streaming=False,
                disable_streaming=True,
            )

        elif self.model_data["zero_shot"]:
            # subistituir pela combinação do langchain com huggingface para baixar e usar os modelos sem ft
            # Carregamento direto de um modelo
            self.model = HuggingFacePipeline.from_model_id(
                model_id=self.model_data["model"],
                task="text-generation",
                device_map="auto",
                verbose=True,
            )
        else:
            # subistituir pela combinação do langchain com huggingface para usar os modelos locais com ft
            self.model = None
            self.tokenizer = None

            if self.model_data["model_type"] == "seq2seq" and self.model_path:
                # Carregar tokenizer e modelo
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            else:
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_path,
                    max_seq_length=2048,  # Ajuste conforme necessário
                    max_new_tokens=self.model_data['max_completition_tokens'],
                    # dtype=None,           # O Unsloth detecta automaticamente o tipo de dado
                    load_in_4bit=True,     # Ativa a quantização de 4 bits para economia de memória
                    dtype=torch.float32,
                    device_map="auto",    # Distribui automaticamente o modelo entre as GPUs disponíveis
                    trust_remote_code=True,  # Permite o uso de código remoto confiável
                    use_exact_model_name=True,  # Usa o nome exato do modelo
                    fast_inference=False,  # Ativa a inferência rápida
                )
                self.model = self.model.to(self.device)
                FastLanguageModel.for_inference(self.model)  # Habilita a inferência otimizada

            # self.model = self.model.to(self.device)

            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.model_data["max_completition_tokens"],
                device_map=0,
                framework="pt",  # Especifica o uso do PyTorch
                device=self.device,  # Define o dispositivo (CPU ou GPU)
                torch_dtype=torch.float32  # Define explicitamente o tipo de dado
            )

            self.model = HuggingFacePipeline(pipeline=pipe)

    def __get_prompt(self):
        if not self.model_data["local"] or "gemma" not in self.model_data['model'] or self.model_data["model_type"] in ["llm", "causal"]:
            msgs = [
                (
                    "system",
                    "You must only to return the sql command, nothing more. Don't use markdown to write answer.",
                ),
                ("human", "{input}"),
            ]
            return ChatPromptTemplate.from_messages(msgs)

        return ChatPromptTemplate.from_messages([("human", "{input}")])

    def use_ajusted_model(self, text):
        """Gerar predições do modelo."""

        self._load_model()

        prompt = self.__get_prompt()

        chain = prompt | self.model | StrOutputParser()

        return chain.invoke({"input": text})


if __name__ == "__main__":
    predictions = []

    model = MODELS["unsloth/gemma-3-1b-it-bnb-4bit"]
    DELAY = 30
    DIR = (
        OUTPUT_MODEL
        if model["local"]
        else f"{OUTPUT_COMMERCIAL_MODEL_PREDICTIONS}{model['model'].replace(':', '-').replace('/', '-').replace('.', '-')}"
    )

    model_path = None
    if model["local"]:
        model_path = os.path.join(OUTPUT_MODEL, "")

    print(f"Realizando predições com modelo \"{model['model']}\".")

    pred = Predicions(model, model_path=model_path)

    dataset = pred.dataset_loader.load()

    progress_bar = tqdm(
        total=len(dataset["test"]), desc="Gerando Predições...", colour="red"
    )

    for data in dataset["test"]:
        if model["local"] and model["model_type"] == "seq2seq":
            instruction = "###Context:\n" + data["context"] + "\n\n###Response: "
        else:
            instruction = data["instruction"]

        X = instruction + data["input"]

        predictions.append(
            PredictionModel(
                db_id=data["db_id"],
                difficulty=data["difficulty"],
                instruction=instruction,
                nl=data["input"],
                sql_expected=data["output"],
                sql_predicted=pred.use_ajusted_model(X),
            ).model_dump()
        )

        progress_bar.update(1)
        time.sleep(DELAY)
    progress_bar.close()

    # Criar o diretório para salvar as predições
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    # Escrever o JSON em um arquivo
    with open(
        os.path.join(
            DIR,
            "predictions.json",
        ),
        "w",
        encoding="utf8",
    ) as f:
        f.write(json.dumps(predictions, indent=4))

    print(
        f"Predictions saved on {os.path.join(DIR, 'predictions.json',)}! "
    )
