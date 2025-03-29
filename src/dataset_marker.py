"""Módulo responsável por marcar o dataset."""

import json
import os

from src.env import DATA_OUTPUT_PATH, DATA_PATH, PREFIX_ANNOTATED
from src.evaluation.evaluator import Evaluator


class DatasetMarker:
    """Classe responsável por marcar o dataset."""

    def __init__(self, dataset):
        self.dataset = dataset

    def process(self):
        """Processa o dataset."""

        if self.dataset is not None:
            for file in [
                self.dataset["train_file"],
                self.dataset["evaluate_file"],
                self.dataset["dev_file"],
            ]:
                self.__process_file(file)

    def __process_file(self, file_name):
        with open(
            os.path.join(DATA_PATH, self.dataset["name"], file_name),
            "r",
            encoding="utf-8",
        ) as file:
            dataset = json.load(file)

            print(f"Arquivo a ser tratado: {file_name}\n\n")
            print("Quantidade de amostras no arquivo: ", len(dataset))

            qtd_by_language = int(len(dataset) / len(self.dataset["languages"]))

            print(f"Quantidade de amostras por idioma: {qtd_by_language}\n\n")

            language_indicator = 0
            data_indicator = 0

            for _, data in enumerate(dataset):
                # descobre a complexidade da amostra
                data["complexity"] = Evaluator.eval_hardness(data["query"])

                # descobre o idioma da amostra
                if data_indicator < qtd_by_language:
                    data["language"] = self.dataset["languages"][language_indicator]
                else:
                    language_indicator += 1
                    data_indicator = 0
                    data["language"] = self.dataset["languages"][language_indicator]

                data_indicator += 1

            if len(self.dataset["languages"]) > 1:
                print("Amostras da fronteira:")
                print(
                    f"* {dataset[qtd_by_language - 1]['language']}: {dataset[qtd_by_language - 1]['question']}"
                )
                print(
                    f"* {dataset[qtd_by_language]['language']}: {dataset[qtd_by_language]['question']}"
                )

            out_file_name = PREFIX_ANNOTATED + file_name

            self.__write_dataset_in_file(out_file_name, dataset)

            print(f'\n\nArquivo "{out_file_name}" tratado e salvo com sucesso!\n')
            print("=============================================================\n")

    def __write_dataset_in_file(self, file_name, data):
        """Escreve o dataset tratado em um arquivo"""

        # cria o diretorio se não exitir
        os.makedirs(f"{DATA_OUTPUT_PATH}", exist_ok=True)

        # reescreve o arquivo com as devidas alterações
        with open(f"{DATA_OUTPUT_PATH}/{file_name}", "w", encoding="utf-8") as file:
            json.dump(data, file)
