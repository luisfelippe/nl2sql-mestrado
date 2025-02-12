"""Script responsável por gerar as predições do modelo."""

import json
import os
import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from src.dataset_loader import DatasetLoader
from src.env import OUTPUT_MODEL
from src.schemas import PredictionModel


class Predictions:
    """Classe responsável por gerar as predições do modelo."""

    def __init__(self, model_path=None):
        # Carregar o tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Carregar o modelo
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        self.dataset_loader = DatasetLoader()

    def use_ajusted_model(self, text):
        """Gerar predições do modelo."""

        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        ).input_ids

        # Gerar a saída
        output_ids = self.model.generate(input_ids, max_new_tokens=128)

        # Decodificar a saída
        saida = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return saida


if __name__ == "__main__":
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

    # Escrever o JSON em um arquivo
    with open(
        os.path.join(OUTPUT_MODEL, "predictions.json"), "w", encoding="utf8"
    ) as f:
        f.write(json.dumps(predictions, indent=4))

    print("Predictions saved!")
