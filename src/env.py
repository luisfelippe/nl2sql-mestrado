"""Variáveis de ambiente para configuração do treinamento e execução do modelo."""

# Nome do dataset alvo, dentre os datasets listados em SQL_DATA_INFO
DATASET_TARGET = "spider-en"

# idioma alvo do treinamento para filtro do dataset
LANGUAGE_TARGET = "EN"

# MODEL = 'google/flan-t5-large' # quebrou com outofmemory no pytorch na linha de treinamento

# TODO reduzir o batch size para testar
# MODEL = 'google/flan-t5-base' # quebrou com outofmemory no pytorch na linha de treinamento
MODEL = "google/flan-t5-small"

# controla se deve ou não salvar as épocas durante o treinamento
SHOULD_SAVE_EPOCH = False

# numero de épocas a ser treinado
NUM_EPOCHS = 10
BATCH_SIZE = 10
USE_FP16 = False

BASE = "."

DATA_PATH = f"{BASE}/data"
DATA_OUTPUT_PATH = f"{DATA_PATH}/{DATASET_TARGET}-ajusted"

MODELS_PATH = f"{BASE}/models"
TRAINNING_PATH = f"{BASE}/training"

OUTPUT_MODEL = f"{MODELS_PATH}/{DATASET_TARGET}-{MODEL.replace('/', '-')}"


PREFIX_ANNOTATED = "annotated-"
PREFIX_PROCESSED = "processed-"
PREFIX_ONE_SHOT = "processed-one-shot-"


SQL_DATA_INFO = {
    "spider-en-pt": {
        "name": "spider-en-pt",
        "languages": ["EN", "PT"],
        "train_tables": "tables.json",
        "dev_tables": "tables.json",
        "eval_tables": "tables.json",
        "train_file": "train_spider.json",
        "evaluate_file": "train_others.json",
        "dev_file": "dev.json",
        "db_id_name": "db_id",
        "output_name": "query",
        "is_multiple_turn": False,
    },
    "spider-en": {
        "name": "spider-en",
        "languages": ["EN"],
        "train_tables": "tables.json",
        "dev_tables": "tables.json",
        "eval_tables": "tables.json",
        "train_file": "train_spider.json",
        "evaluate_file": "train_others.json",
        "dev_file": "dev.json",
        "db_id_name": "db_id",
        "output_name": "query",
        "is_multiple_turn": False,
    },
    "spider-pt": {
        "name": "spider-pt",
        "languages": ["PT"],
        "train_tables": "tables.json",
        "dev_tables": "tables.json",
        "eval_tables": "tables.json",
        "train_file": "train_spider.json",
        "evaluate_file": "train_others.json",
        "dev_file": "dev.json",
        "db_id_name": "db_id",
        "output_name": "query",
        "is_multiple_turn": False,
    },
    # "bird" : {
    #     "name": "bird",
    #     "train_file": "train/train.json",
    #     "evaluate_file": "",
    #     "dev_file": "dev/dev.json",
    #     "train_tables": "train/train_tables.json",
    #     "eval_tables": "train/train_tables.json",
    #     "dev_tables": "dev/dev_tables.json",
    #     "db_id_name": "db_id",
    #     "output_name": "SQL",
    #     "is_multiple_turn": False,
    # }
}

INSTRUCTION_PROMPT = """\
I want you to act as a SQL terminal in front of an example database, \
you need only to return the sql command to me.Below is an instruction that describes a task, \
Write a response that appropriately completes the request.\n"
##Instruction:\n{}\n"""

INSTRUCTION_ONE_SHOT_PROMPT = """\
I want you to act as a SQL terminal in front of an example database. \
You need only to return the sql command to me. \
First, I will show you few examples of an instruction followed by the correct SQL response. \
Then, I will give you a new instruction, and you should write the SQL response that appropriately completes the request.\
\n### Example1 Instruction:
The database contains tables such as employee, salary, and position. \
Table employee has columns such as employee_id, name, age, and position_id. employee_id is the primary key. \
Table salary has columns such as employee_id, amount, and date. employee_id is the primary key. \
Table position has columns such as position_id, title, and department. position_id is the primary key. \
The employee_id of salary is the foreign key of employee_id of employee. \
The position_id of employee is the foreign key of position_id of position.\
\n### Example1 Input:\nList the names and ages of employees in the 'Engineering' department.\n\
\n### Example1 Response:\nSELECT employee.name, employee.age FROM employee JOIN position ON employee.position_id = position.position_id WHERE position.department = 'Engineering';\
\n###New Instruction:\n{}\n"""

INPUT_PROMPT = "###Input:\n{}\n\n###Response:"
