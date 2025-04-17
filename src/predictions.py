"""Script responsável por gerar as predições do modelo."""

import gc
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

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


from src.dataset_loader import DatasetLoader
from src.env import OUTPUT_COMMERCIAL_MODEL_PREDICTIONS, OUTPUT_MODEL, MODELS
from src.schemas import PredictionModel
from src.format import format as format_sql

# TODO Gemma não aceita sistem prompt
# reacall alto signinifca pouca precisão
# TODO usar pelo langchain os modelos gemma 2 e 9b e lhama 1, 3 e 8b pelo huggingface


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
        self.model_path = model_path if model['model_type'] != "seq2seq" else model['model']
        self.tokenizer = None

        # define se a casse utilizada será a do unsloth ou a do huggingface
        self.use_unsloth = model['model_type'] != "seq2seq"

    def _load_model(self):
        if self.model:
            return

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
            # subistituir pela combinação do langchain com huggingface para usar os modelos locais com ft
            self.model = None
            self.tokenizer = None

            if self.use_unsloth:
                print("Usando o modelo com unsloth")
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_data["model"],
                    max_seq_length=2048,  # Ajuste conforme necessário
                    # max_new_tokens=self.model_data['max_completition_tokens'],
                    # dtype=None,           # O Unsloth detecta automaticamente o tipo de dado
                    load_in_4bit=True,  # Ativa a quantização de 4 bits para economia de memória
                    # dtype=torch.float32,
                    device_map="auto",  # Distribui automaticamente o modelo entre as GPUs disponíveis
                    # trust_remote_code=True,  # Permite o uso de código remoto confiável
                    use_exact_model_name=True,  # Usa o nome exato do modelo
                    fast_inference=False,  # Ativa a inferência rápida
                )
                # self.model = self.model.to(self.device)
                FastLanguageModel.for_inference(
                    self.model
                )  # Habilita a inferência otimizada
            else:
                print("Usando o modelo com huggingface")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

                if self.model_data["model_type"] == "seq2seq" and self.model_path:
                    print("Usando o modelo seq2seq")
                    # Carregar tokenizer e modelo
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
                else:
                    print("Usando o modelo causal")
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

            # self.model = self.model.to(self.device)

            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.model_data["max_completition_tokens"],
                device_map="auto",
                framework="pt",  # Especifica o uso do PyTorch
                # device=self.device,  # Define o dispositivo (CPU ou GPU)
                # torch_dtype=torch.float32  # Define explicitamente o tipo de dado
            )

            self.model = HuggingFacePipeline(pipeline=pipe)

    def __get_prompt(self):
        if (
            not self.model_data["local"]
            or "gemma" not in self.model_data["model"]
            or self.model_data["model_type"] in ["llm", "causal"]
        ):
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
    
    def use_ajusted_model_batch(self, list_of_text):
        """Gerar predições do modelo em lote."""

        self._load_model()

        prompt = self.__get_prompt()

        chain = prompt | self.model | StrOutputParser()

        return chain.batch([{"input": text} for text in list_of_text])

# def format(text):
#     # remove o markdown do sql
#     text = text.replace("```sql", " ").replace("```", " ").replace("\n", " ")
#     # remove os espaços em branco duplicados
#     text = " ".join(text.split())
#     # remove os espaços em branco no início e no final da string
#     text = text.strip()

#     return text


if __name__ == "__main__":
    predictions = []

    model = MODELS["google/flan-t5-base"]
    DELAY = 0
    DIR = (
        OUTPUT_MODEL
        if model["local"] and not model["zero_shot"]
        else f"{OUTPUT_COMMERCIAL_MODEL_PREDICTIONS}{model['model'].replace(':', '-').replace('/', '-').replace('.', '-')}"
    )

    model_path = None
    if model["local"]:
        model_path = os.path.join(OUTPUT_MODEL, "")

    # Criar o diretório para salvar as predições
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    predictions_file = os.path.join(
        DIR,
        "predictions.json",
    )

    # Tenta carregar as predições existentes, se o arquivo existir
    if os.path.exists(predictions_file):
        with open(predictions_file, "r", encoding="utf8") as f:
            try:
                predictions = json.load(f)
            except json.JSONDecodeError:
                predictions = []

    print(f"Realizando predições com modelo \"{model['model']}\".")

    pred = Predicions(model, model_path=model_path)

    dataset = pred.dataset_loader.load()

    progress_bar = tqdm(
        total=len(dataset["test"]), desc="Gerando Predições...", colour="red"
    )

    # Converte a lista de predições existente para facilitar consulta
    existing_ids = {item["id"] for item in predictions if "id" in item}

    for data in dataset["test"]:
        # Se o ID já está nas predições, pula esta amostra
        if data["id"] in existing_ids:
            progress_bar.update(1)
            continue

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
                sql_predicted=format_sql(pred.use_ajusted_model(X)),
                id=data["id"]
            ).model_dump()
        )
        existing_ids.add(data["id"])

        with open(os.path.join(DIR, "predictions.json"), "w", encoding="utf8") as f:
            f.write(json.dumps(predictions, indent=4))

        progress_bar.update(1)

        if DELAY > 0:
            # aguarda o tempo definido para evitar sobrecarga no modelo
            time.sleep(DELAY)

        # invoca garbage collection
        # torch.cuda.empty_cache()
        gc.collect()

    progress_bar.close()

    # Criar o diretório para salvar as predições
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    print(f"Predictions saved on {os.path.join(DIR, 'predictions.json',)}! ")
