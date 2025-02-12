"""Script principal para execução das tarefas de treinamento, \
    marcação, processamento do dataset, predições e avaliação \
        das predições."""

import argparse
import json
import os

from tqdm import tqdm

from src.env import (
    DATA_OUTPUT_PATH,
    DATASET_TARGET,
    OUTPUT_MODEL,
    PREFIX_ONE_SHOT,
    PREFIX_PROCESSED,
    SQL_DATA_INFO,
)


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


def evaluate():
    """Avalia as predições."""

    print("Avaliação das predições iniciada...")
    # ...código de avaliação das predições...
    print("Avaliação das predições concluída.")


def main():
    """Função principal."""

    parser = argparse.ArgumentParser(
        description="Ferramenta para treinamento, marcação, processamento do dataset, \
            predições e avaliação das predições. Use --help para ver todas as opções."
    )
    parser.add_argument("--train", action="store_true", help="Executa o treinamento.")
    parser.add_argument("--label", action="store_true", help="Executa a marcação.")
    parser.add_argument(
        "--process", action="store_true", help="Executa o processamento do dataset."
    )
    parser.add_argument("--predict", action="store_true", help="Executa as predições.")
    parser.add_argument(
        "--evaluate", action="store_true", help="Executa a avaliação das predições."
    )
    parser.add_argument(
        "--code-representation",
        action="store_true",
        help="Usa representação de código no processamento do dataset.",
    )
    # parser.add_argument(
    #     "--help", action="store_true", help="Exibe esta mensagem de ajuda."
    # )

    args = parser.parse_args()

    if args.train:
        train()
    if args.label:
        label()
    if args.process:
        process_dataset(args.code_representation)
    if args.predict:
        predict()
    if args.evaluate:
        evaluate()


if __name__ == "__main__":
    main()
