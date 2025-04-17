"""Processa o dataset para o formato de entrada do modelo."""

import os
import re
import json
import uuid
from tqdm import tqdm
import jsonlines
import spacy
from src.dataset_marker import DatasetMarker
from src.dataset_reporter import DatasetReport
from src.env import (
    DATA_OUTPUT_PATH,
    DATA_PATH,
    DATASET_TARGET,
    INPUT_PROMPT,
    INSTRUCTION_ONE_SHOT_PROMPT,
    INSTRUCTION_PROMPT,
    PREFIX_ANNOTATED,
    PREFIX_ONE_SHOT,
    PREFIX_PROCESSED,
    SQL_DATA_INFO,
)


class ProcessDataset:
    """Processa o dataset para o formato de entrada do modelo."""
    def __init__(
        self,
        dataset,
        train_file,
        eval_file,
        dev_file,
        num_shot=0,
        code_representation=False,
    ):
        self.dataset = dataset
        self.num_shot = num_shot
        self.code_representation = code_representation

        self.train_file = train_file
        self.eval_file = eval_file
        self.dev_file = dev_file

        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_pt = spacy.load("pt_core_news_sm")

    def verifica_processado(self):
        """Verifica se o dataset já foi processado."""
        exists_train = os.path.isfile(
            self.train_file
        )
        exists_eval = os.path.isfile(
            self.eval_file
        )
        exists_dev = os.path.isfile(
            self.dev_file
        )

        if exists_train and exists_eval and exists_dev:
            return True

        return False

    def __verifica_anotacao(self):
        exists_train = os.path.isfile(
            os.path.join(
                DATA_OUTPUT_PATH, f"{PREFIX_ANNOTATED}{self.dataset['train_file']}"
            )
        )
        exists_eval = os.path.isfile(
            os.path.join(
                DATA_OUTPUT_PATH, f"{PREFIX_ANNOTATED}{self.dataset['evaluate_file']}"
            )
        )
        exists_dev = os.path.isfile(
            os.path.join(
                DATA_OUTPUT_PATH, f"{PREFIX_ANNOTATED}{self.dataset['dev_file']}"
            )
        )

        if exists_train and exists_eval and exists_dev:
            return True

        return False

    def __count_words(self, text, language):
        return len(self.nlp_en(text) if language == "EN" else self.nlp_pt(text))

    def __decode_json_file(
        self,
        data_file_list,
        table_file,
        db_folder_path,
        db_id_name,
        output_name,
        is_multiple_turn=False,
    ):
        """
        @TODO:
            1. Colocar o prompt relacionado no arquivo de configuração
            2. Colocar as informações dos campos de diferentes fontes de dados no arquivo de configuração
        """

        if table_file.endswith(".jsonl"):
            tables = jsonlines.open(table_file)
            datas = []
            for data_file in data_file_list:
                datas.extend(jsonlines.open(data_file))

        elif table_file.endswith(".json"):
            with open(table_file, encoding="utf-8") as table:
                tables = json.load(table)
                datas = []

                for data_file in data_file_list:
                    with open(data_file, encoding="utf-8") as data:
                        datas.extend(json.load(data))
        else:
            print("Unsupported file types")
            raise ValueError("Unsupported file types")

        # Primeiro, processe corretamente as tabelas e colunas do db_id
        db_dict = {}
        for item in tables:
            tables = item["table_names_original"]
            coloumns = item["column_names_original"][1:]
            primary_key = item["primary_keys"]
            foreign_keys = item["foreign_keys"]

            source = (
                item["db_id"] + " contains tables such as " + ", ".join(tables) + ". "
            )

            for i, name in enumerate(tables):
                data = [coloumn[1] for coloumn in coloumns if coloumn[0] == i]
                source += (
                    "Table " + name + " has columns such as " + ", ".join(data) + ". "
                )

                # get primary key info
                for j in range(len(primary_key)):
                    if type(primary_key[j]) == int:
                        if coloumns[primary_key[j] - 1][0] == i:
                            source += (
                                coloumns[primary_key[j] - 1][1]
                                + " is the primary key."
                                + "\n"
                            )

                    # combination primary key
                    elif type(primary_key[j]) == list:
                        combine_p = "The combination of ("
                        keys = []

                        for k in range(len(primary_key[j])):
                            if coloumns[primary_key[j][k] - 1][0] == i:
                                keys.append(coloumns[primary_key[j][k] - 1][1])

                        source += (
                            combine_p
                            + ", ".join(keys)
                            + ") are the primary key."
                            + "\n"
                        )
                    else:
                        print("not support type", type(primary_key[j]))
                        continue

            # get foreign key info
            for key in foreign_keys:
                source += (
                    "The "
                    + coloumns[key[0] - 1][1]
                    + " of "
                    + tables[coloumns[key[0] - 1][0]]
                    + " is the foreign key of "
                    + coloumns[key[1] - 1][1]
                    + " of "
                    + tables[coloumns[key[1] - 1][0]]
                    + ".\n"
                )

            db_dict[item["db_id"]] = source

        res = []
        base_instruction = INSTRUCTION_PROMPT

        if self.num_shot == 1:
            base_instruction = INSTRUCTION_ONE_SHOT_PROMPT

        for data in tqdm(datas):
            if data[db_id_name] in db_dict.keys():
                if is_multiple_turn:  # Múltiplas rodadas
                    history = []

                    for interaction in data["interaction"]:
                        sql_query = interaction[output_name]

                        input = INPUT_PROMPT.format(interaction["utterance"])
                        context = db_dict[data[db_id_name]]

                        input_data = {
                            "id": str(uuid.uuid4()),
                            "db_id": data[db_id_name],
                            "instruction": base_instruction.format(context),
                            "context": context,
                            "input": input,
                            "language": data["language"],
                            "output": sql_query,
                            "difficulty": data["complexity"],
                            "history": history,
                            "count_words_in": self.__count_words(
                                input.replace("\n\n###Response:", "\n\n###Context:\n")
                                + context
                                + "\n\n###Response:",
                                data["language"],
                            ),
                            "count_words_out": self.__count_words(
                                sql_query, data["language"]
                            ),
                        }

                        res.append(input_data)
                        history.append(
                            (
                                INPUT_PROMPT.format(interaction["utterance"]),
                                interaction[output_name],
                            )
                        )
                else:  # Rodada única
                    sql_query = data[output_name]

                    if self.code_representation:
                        db_path = os.path.join(db_folder_path, data[db_id_name])
                        sql_file_path = next(
                            (
                                file
                                for file in os.listdir(db_path)
                                if file.endswith(".sql")
                            ),
                            None,
                        )

                        if sql_file_path is None:
                            continue  # Encerrar a iteração antecipadamente

                        schema_file_path = os.path.join(db_path, sql_file_path)

                        with open(schema_file_path, "r", encoding="utf8") as file:
                            schema_content = file.read()

                        create_statements = re.findall(
                            r"CREATE\s.*?;", schema_content, re.DOTALL | re.IGNORECASE
                        )

                        input = INPUT_PROMPT.format(data["question"])

                        input_data = {
                            "id": str(uuid.uuid4()),
                            "db_id": data[db_id_name],
                            "instruction": INSTRUCTION_PROMPT.format(create_statements),
                            "context": "\n".join(create_statements),
                            "input": input,
                            "language": data["language"],
                            "output": sql_query,
                            "difficulty": data["complexity"],
                            "history": [],
                            "count_words_in": self.__count_words(
                                input.replace("\n\n###Response:", "\n\n###Context:\n")
                                + "\n".join(create_statements)
                                + "\n\n###Response:",
                                data["language"],
                            ),
                            "count_words_out": self.__count_words(
                                sql_query, data["language"]
                            ),
                        }
                        res.append(input_data)
                    else:
                        input = INPUT_PROMPT.format(data["question"])
                        context = db_dict[data[db_id_name]]
                        input_data = {
                            "id": str(uuid.uuid4()),
                            "db_id": data[db_id_name],
                            "instruction": base_instruction.format(context),
                            "context": context,
                            "input": input,
                            "language": data["language"],
                            "output": sql_query,
                            "difficulty": data["complexity"],
                            "history": [],
                            "count_words_in": self.__count_words(
                                input.replace("\n\n###Response:", "\n\n###Context:\n")
                                + context
                                + "\n\n###Response:",
                                data["language"],
                            ),
                            "count_words_out": self.__count_words(
                                sql_query, data["language"]
                            ),
                        }
                        res.append(input_data)
        return res

    def process(self, report=True):
        """Processa o dataset."""

        print("Iniciando processador do dataset.")

        if not self.__verifica_anotacao():
            print("Dataset ainda não foi anotado. Efetuando anotação...")
            marker = DatasetMarker(self.dataset)
            marker.process()

        print("Dataset devidamente anotado.")

        print("\nProcessando o dataset...")

        train_data = []
        eval_data = []
        dev_data = []

        for data_info in SQL_DATA_INFO.values():
            if data_info["name"] != DATASET_TARGET:
                continue

            tfile = data_info["train_file"]
            efile = data_info["evaluate_file"]
            dfile = data_info["dev_file"]

            train_data_file_list = [
                os.path.join(DATA_OUTPUT_PATH, f"{PREFIX_ANNOTATED}{tfile}")
            ]

            train_data.extend(
                self.__decode_json_file(
                    data_file_list=train_data_file_list,
                    table_file=os.path.join(
                        DATA_PATH,
                        data_info["name"],
                        data_info["train_tables"],
                    ),
                    db_folder_path=os.path.join(
                        DATA_PATH,
                        data_info["name"],
                        "database",
                    ),
                    db_id_name=data_info["db_id_name"],
                    output_name=data_info["output_name"],
                    is_multiple_turn=data_info["is_multiple_turn"],
                )
            )

            eval_data_file_list = [
                os.path.join(DATA_OUTPUT_PATH, f"{PREFIX_ANNOTATED}{efile}")
            ]

            eval_data.extend(
                self.__decode_json_file(
                    data_file_list=eval_data_file_list,
                    table_file=os.path.join(
                        DATA_PATH,
                        data_info["name"],
                        data_info["eval_tables"],
                    ),
                    db_folder_path=os.path.join(
                        DATA_PATH,
                        data_info["name"],
                        "database",
                    ),
                    db_id_name=data_info["db_id_name"],
                    output_name=data_info["output_name"],
                    is_multiple_turn=data_info["is_multiple_turn"],
                )
            )

            dev_data_file_list = [
                os.path.join(DATA_OUTPUT_PATH, f"{PREFIX_ANNOTATED}{dfile}")
            ]

            dev_data.extend(
                self.__decode_json_file(
                    data_file_list=dev_data_file_list,
                    table_file=os.path.join(
                        DATA_PATH,
                        data_info["name"],
                        data_info["dev_tables"],
                    ),
                    db_folder_path=os.path.join(
                        DATA_PATH,
                        data_info["name"],
                        "database",
                    ),
                    db_id_name=data_info["db_id_name"],
                    output_name=data_info["output_name"],
                    is_multiple_turn=data_info["is_multiple_turn"],
                )
            )

        if train_data:
            with open(self.train_file, "w", encoding="utf-8") as s:
                json.dump(train_data, s, indent=4, ensure_ascii=False)

        if eval_data:
            with open(self.eval_file, "w", encoding="utf-8") as s:
                json.dump(eval_data, s, indent=4, ensure_ascii=False)

        if dev_data:
            with open(self.dev_file, "w", encoding="utf-8") as s:
                json.dump(dev_data, s, indent=4, ensure_ascii=False)

        if not train_data and not eval_data and not dev_data:
            print("Nenhum dataset foi processado.")
            return

        if report:
            print("Dataset processado com sucesso!")
            reporter = DatasetReport(
                self.dataset, [self.train_file, self.eval_file, self.dev_file]
            )
            reporter.report()


if __name__ == "__main__":
    dataset = SQL_DATA_INFO[DATASET_TARGET]

    print(f"Iniciando processamento do dataset {DATASET_TARGET}")

    train_file = os.path.join(
        DATA_OUTPUT_PATH, f"{PREFIX_PROCESSED}{dataset['train_file']}"
    )
    eval_file = os.path.join(
        DATA_OUTPUT_PATH, f"{PREFIX_PROCESSED}{dataset['evaluate_file']}"
    )
    dev_file = os.path.join(
        DATA_OUTPUT_PATH, f"{PREFIX_PROCESSED}{dataset['dev_file']}"
    )

    process = ProcessDataset(
        dataset=dataset,
        train_file=train_file,
        eval_file=eval_file,
        dev_file=dev_file,
        code_representation=False,  # args.code_representation,
    )
    process.process()

    print(f"Iniciando processamento do dataset {DATASET_TARGET} com One Shot Learning")

    onse_shot_train_file = os.path.join(
        DATA_OUTPUT_PATH, f"{PREFIX_ONE_SHOT}{dataset['train_file']}"
    )
    onse_shot_eval_file = os.path.join(
        DATA_OUTPUT_PATH, f"{PREFIX_ONE_SHOT}{dataset['evaluate_file']}"
    )
    onse_shot_dev_file = os.path.join(
        DATA_OUTPUT_PATH, f"{PREFIX_ONE_SHOT}{dataset['dev_file']}"
    )

    process = ProcessDataset(
        dataset=dataset,
        train_file=onse_shot_train_file,
        eval_file=onse_shot_eval_file,
        dev_file=onse_shot_dev_file,
        num_shot=1,
        code_representation=False,  # args.code_representation,
    )
    process.process(report=False)

    print(train_file, eval_file, dev_file)

    print("Finalizado processamento do Dataset!")
