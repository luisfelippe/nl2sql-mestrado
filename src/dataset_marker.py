"""Módulo responsável por marcar o dataset."""

import json
import re
import os

from src.env import DATA_OUTPUT_PATH, DATA_PATH, PREFIX_ANNOTATED


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

    def __complexity_discover_of_query(self, sql_query):
        """Classifica o nível de dificuldade do SQL com base nos critérios do Spider."""

        # Contar o número de colunas no SELECT
        select_match = re.search(
            r"\bSELECT\b\s+(.*?)(\bFROM\b)", sql_query, re.IGNORECASE | re.DOTALL
        )
        if select_match:
            select_columns = select_match.group(1).split(",")
            num_select = len([col.strip() for col in select_columns if col.strip()])
        else:
            num_select = 0

        # Contar o número de condições no WHERE
        where_conditions = re.findall(
            r"\bWHERE\b(.*?)(\bGROUP BY\b|\bORDER BY\b|$)",
            sql_query,
            re.IGNORECASE | re.DOTALL,
        )
        num_where = 0
        if where_conditions:
            where_clause = where_conditions[0][0]
            num_where = (
                len(re.findall(r"AND|OR", where_clause, re.IGNORECASE))
                if where_clause.strip()
                else 0
            )

        # Contar o número de colunas no GROUP BY
        group_by_match = re.search(
            r"\bGROUP BY\b\s+(.*?)(\bORDER BY\b|$)",
            sql_query,
            re.IGNORECASE | re.DOTALL,
        )
        if group_by_match:
            group_by_columns = group_by_match.group(1).split(",")
            num_group_by = len([col.strip() for col in group_by_columns if col.strip()])
        else:
            num_group_by = 0

        # Contar o número de colunas no ORDER BY
        order_by_match = re.search(
            r"\bORDER BY\b\s+(.*?)(LIMIT|$)", sql_query, re.IGNORECASE | re.DOTALL
        )
        if order_by_match:
            order_by_columns = order_by_match.group(1).split(",")
            num_order_by = len([col.strip() for col in order_by_columns if col.strip()])
        else:
            num_order_by = 0

        # Contar subconsultas com base nos parênteses
        num_nested = len(re.findall(r"\(SELECT\b", sql_query, re.IGNORECASE))

        # Contar o número de junções (JOIN)
        num_joins = len(re.findall(r"\bJOIN\b", sql_query, re.IGNORECASE))

        # Verificar a presença de EXCEPT, INTERSECT e UNION
        has_except = bool(re.search(r"\bEXCEPT\b", sql_query, re.IGNORECASE))
        has_intersect = bool(re.search(r"\bINTERSECT\b", sql_query, re.IGNORECASE))
        has_union = bool(re.search(r"\bUNION\b", sql_query, re.IGNORECASE))

        # Critério especial para subconsultas com JOIN
        has_nested_join = bool(
            re.search(r"\(SELECT\b.*?\bJOIN\b", sql_query, re.IGNORECASE | re.DOTALL)
        )

        # Classificação de dificuldade com base nos critérios do Spider
        if has_union:
            return "extra hard"  # `UNION` é sempre "extra hard"
        elif has_nested_join or num_nested > 1:
            return "extra hard"  # Subconsulta com JOIN ou múltiplas subconsultas
        elif (
            num_select <= 1
            and num_where <= 1
            and num_group_by == 0
            and num_order_by == 0
            and num_nested == 0
            and num_joins == 0
            and not (has_except or has_intersect)
        ):
            return "easy"
        elif (
            num_select <= 3
            and num_where <= 2
            and num_group_by <= 1
            and num_order_by <= 1
            and num_nested == 0
            and num_joins <= 1
            and not (has_except or has_intersect)
        ):
            return "medium"
        elif (
            num_group_by > 1
            or num_order_by > 1
            or num_nested > 0
            or num_where > 2
            or num_joins > 1
            or has_except
            or has_intersect
        ):
            return "hard"
        else:
            return "extra hard"

    def __process_file(self, file_name):
        with open(
            os.path.join(DATA_PATH, self.dataset["name"], file_name), "r"
        ) as file:
            dataset = json.load(file)

            print(f"Arquivo a ser tratado: {file_name}\n\n")
            print("Quantidade de amostras no arquivo: ", len(dataset))

            qtd_by_language = int(len(dataset) / len(self.dataset["languages"]))

            print(f"Quantidade de amostras por idioma: {qtd_by_language}\n\n")

            language_indicator = 0
            data_indicator = 0

            for i, data in enumerate(dataset):
                # descobre a complexidade da amostra
                data["complexity"] = self.__complexity_discover_of_query(data["query"])

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
        with open(f"{DATA_OUTPUT_PATH}/{file_name}", "w") as file:
            json.dump(data, file)
