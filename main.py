"""Script principal para execução das tarefas de treinamento, \
    marcação, processamento do dataset, predições e avaliação \
        das predições."""

import argparse
import json
import os

from tqdm import tqdm

from src.env import (
    DATA_OUTPUT_PATH,
    DATA_PATH,
    DATASET_TARGET,
    OUTPUT_MODEL,
    PREFIX_ONE_SHOT,
    PREFIX_PROCESSED,
    SQL_DATA_INFO,
)
from src.evaluation.evaluator import evaluate as evaluator


def train():
    """Treinamento do modelo."""

    from src.trainer import ModelTrainer

    print("Treinamento iniciado...")

    trainer = ModelTrainer()
    trainer.train()

    print("Treinamento concluído.")


def label():
    """Marcação de dados."""

    from src.dataset_marker import DatasetMarker

    print("Marcação iniciada...")

    dataset = SQL_DATA_INFO[DATASET_TARGET]
    marker = DatasetMarker(dataset)
    marker.process()

    print("Marcação concluída.")


def process_dataset(code_representation=False):
    """Processa o dataset."""

    from src.dataset_processor import ProcessDataset

    print("Processamento do dataset iniciado...")

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
        code_representation=code_representation,  # args.code_representation,
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
        code_representation=code_representation,  # args.code_representation,
    )
    process.process(report=False)

    print("Finalizado processamento do Dataset!")


def predict():
    """Gera as predições."""

    from src.predictions import Predictions
    from src.schemas import PredictionModel

    print("Predições iniciadas...")

    model_path = os.path.join(OUTPUT_MODEL, "")
    predictions = []

    pred = Predictions(model_path)
    dataset = pred.dataset_loader.load()

    progress_bar = tqdm(
        total=len(dataset["test"]), desc="Gerando Predições...", colour="red"
    )

    for data in dataset["test"]:
        predictions.append(
            PredictionModel(
                db_id=data["db_id"],
                difficulty=data["difficulty"],
                instruction=data["instruction"],
                nl=data["input"],
                sql_expected=data["output"],
                sql_predicted=pred.use_ajusted_model(
                    data["instruction"] + data["input"]
                ),
            ).model_dump()
        )

        progress_bar.update(1)

    predictions_file = os.path.join(OUTPUT_MODEL, "predictions.json")
    # Escrever o JSON em um arquivo
    with open(predictions_file, "w", encoding="utf8") as f:
        f.write(json.dumps(predictions, indent=4))

    print(f"Predições salvas no arquivo {predictions_file}!")


def evaluate(pred, db, table, etype, output):
    """Avalia as predições."""

    print("\n\nAvaliação das predições iniciada...")
    assert etype in ["all", "exec", "match"], "Unknown evaluation method"

    if db is None:
        db = os.path.join(DATA_PATH, SQL_DATA_INFO[DATASET_TARGET]["name"], "database")

    if table is None:
        table = os.path.join(
            DATA_PATH, SQL_DATA_INFO[DATASET_TARGET]["name"], "tables.json"
        )

    if output is None:
        output = os.path.join(OUTPUT_MODEL, "evaluation.json")

    evaluator(pred, db, etype, table, output)

    print("\n\nAvaliação das predições concluída.")


def main():
    """Função principal."""

    parser = argparse.ArgumentParser(
        description="Ferramenta para treinamento, marcação, processamento do dataset, \
            predições e avaliação das predições. Use --help para ver todas as opções."
    )

    # Utilizando subcomandos para cada ação
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando para "train"
    train_parser = subparsers.add_parser("train", help="Treina o modelo.")

    # Subcomando para "label"
    label_parser = subparsers.add_parser(
        "label", help="Anota o dataset que será utilizado no treinamento."
    )

    # Subcomando para "label"
    predict_parser = subparsers.add_parser(
        "predict", help="Utiliza o modelo para realizar as predições para a avaliação."
    )

    # Subcomando para "process"
    process_parser = subparsers.add_parser(
        "process",
        help="Realiza o pré-processamento do dataset que será utilizado no treinamento.",
    )
    process_parser.add_argument(
        "--code-representation",
        action="store_true",
        help="Define se o método de processamento deve utilizar code-representation (booleano).",
    )

    # Subcomando para "evaluate"
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Avalia o modelo através das métricas definidas."
    )
    evaluate_parser.add_argument("pred", help="Caminho do arquivo de predições.")
    evaluate_parser.add_argument(
        "db", help="Caminho onde estão os arquivos de banco de dados."
    )
    evaluate_parser.add_argument(
        "table", help="Caminho para o arquivo que contém as informações das tabelas."
    )
    evaluate_parser.add_argument(
        "etype",
        choices=["all", "exec", "match"],
        help='Tipo de avaliação: "all", "exec" ou "match".',
    )
    evaluate_parser.add_argument(
        "output", help="Caminho para salvar o resultado da avaliação."
    )

    args = parser.parse_args()

    # Verifica qual subcomando foi chamado e invoca a função correspondente
    if args.command == "label":
        label()
    elif args.command == "process":
        process_dataset(args.code_representation)
    elif args.command == "train":
        train()
    elif args.command == "predict":
        predict()
    elif args.command == "evaluate":
        evaluate(args.pred, args.db, args.table, args.etype, args.output)


if __name__ == "__main__":
    main()
