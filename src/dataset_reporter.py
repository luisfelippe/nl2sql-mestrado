"""Módulo responsável por gerar um relatório do dataset."""

import json
import numpy as np


class DatasetReport:
    """Classe responsável por gerar um relatório do dataset."""

    def __init__(self, dataset, files):
        self.dataset = dataset
        self.files = files

    def __report(self, file_name):
        words_in = []
        words_out = []

        with open(file_name, "r") as file:
            dataset = json.load(file)

            print(f"Quantidade de amostras: {len(dataset)}")

            qtd_easy = 0
            qtd_medium = 0
            qtd_hard = 0
            qtd_extra_hard = 0

            for _, data in enumerate(dataset):
                if data["difficulty"] == "easy":
                    qtd_easy += 1
                elif data["difficulty"] == "medium":
                    qtd_medium += 1
                elif data["difficulty"] == "hard":
                    qtd_hard += 1
                elif data["difficulty"] == "extra hard":
                    qtd_extra_hard += 1

                words_in.append(data["count_words_in"])
                words_out.append(data["count_words_out"])

        print(
            f"\nQuantidade de amostras fáceis: {qtd_easy} = {round((qtd_easy/len(dataset)) * 100, 2)}%"
        )
        print(
            f"Quantidade de amostras médias: {qtd_medium} = {round((qtd_medium/len(dataset)) * 100, 2)}%"
        )
        print(
            f"Quantidade de amostras difíceis: {qtd_hard} = {round((qtd_hard/len(dataset)) * 100, 2)}%"
        )
        print(
            f"Quantidade de amostras extra difíceis: {qtd_extra_hard} = {round((qtd_extra_hard/len(dataset)) * 100, 2)}%"
        )

        print(f"\nQuantidade mínima de palavras no input: {np.min(words_in)}")
        print(f"Quantidade máxima de palavras no input: {np.max(words_in)}")
        print(f"Quantidade média de palavras no input: {np.mean(words_in)}")
        print(f"Quantidade total de palavras no input: {np.sum(words_in)}")
        print(f"Desvio padrão: {np.std(words_in)}")

        print(f"\nQuantidade mínima de palavras no output: {np.min(words_out)}")
        print(f"Quantidade máxima de palavras no output: {np.max(words_out)}")
        print(f"Quantidade média de palavras no output: {np.mean(words_out)}")
        print(f"Quantidade total de palavras no output: {np.sum(words_out)}")
        print(f"Desvio padrão: {np.std(words_out)}")
        print("=======================================\n\n")

    def report(self):
        """Gera um relatório do dataset."""

        print("\n\n=======================================")
        print("Relatório de processamento do dataset.")
        print(f"Dataset: {self.dataset['name']}")

        for file in self.files:
            print(f"\nArquivo: {file}")
            self.__report(file)
