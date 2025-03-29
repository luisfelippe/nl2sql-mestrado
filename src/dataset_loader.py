"""Módulo responsável por carregar o dataset processado."""

import os
import pandas as pd
from datasets import DatasetDict, Dataset
from src.dataset_processor import ProcessDataset
from src.env import (
    DATA_OUTPUT_PATH,
    LANGUAGE_TARGET,
    PREFIX_PROCESSED,
)


class DatasetLoader:
    """Classe responsável por carregar o dataset processado."""

    def __load_dataset(self, path, language=None):
        with open(path, "r", encoding="utf-8") as arquivo:
            data = pd.read_json(arquivo)
            if language is None:
                return data
            
            # Filtra os dados onde 'language' 
            # é igual ao parâmetro fornecido
            filtered_data = data[data["language"] == language]
            return filtered_data

    def load(self):
        """Carrega o dataset processado e retorna um DatasetDict \
            com os datasets de treino, validação e teste."""
        
        # Verifica se o dataset foi processado
        # if not self.processor.verifica_processado():
        #     print(
        #         "Dataset ainda não foi processado. \
        #             Efetue o processamento antes de carregar o dataset."
        #     )
        #     return None

        # transforma os dataframes em Dataset para a devida utilização no treinamento
        dataset_dict = {
            "train": Dataset.from_pandas(
                self.__load_dataset(
                    os.path.join(
                        DATA_OUTPUT_PATH, f"{PREFIX_PROCESSED}train_spider.json"
                    ),
                    language=LANGUAGE_TARGET,
                )
            ),
            "validation": Dataset.from_pandas(
                self.__load_dataset(
                    os.path.join(
                        DATA_OUTPUT_PATH, f"{PREFIX_PROCESSED}train_others.json"
                    ),
                    language=LANGUAGE_TARGET,
                )
            ),
            "test": Dataset.from_pandas(
                self.__load_dataset(
                    os.path.join(DATA_OUTPUT_PATH, f"{PREFIX_PROCESSED}dev.json"),
                    language=LANGUAGE_TARGET,
                )
            ),
        }

        return DatasetDict(dataset_dict)
