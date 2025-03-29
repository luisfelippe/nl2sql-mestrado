"""Módulo responsável por treinar o modelo."""

import os
import torch
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import evaluate
import numpy as np

from dotenv import load_dotenv

from src.callbacks.save_model_by_epoch_callback import SaveModelByEpochCallback
from src.dataset_loader import DatasetLoader
from src.env import (
    BATCH_SIZE,
    MODEL,
    NUM_EPOCHS,
    OUTPUT_MODEL,
    USE_FP16,
)

# Load environment variables from the .env file (if present)
load_dotenv()

import nltk

nltk.download("punkt_tab")

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


class ModelTrainer:
    """Classe responsável por treinar o modelo."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

        self.rouge = evaluate.load("rouge")
        self.dataset_loader = DatasetLoader()

        nltk.download("punkt")

    def __preprocess_data(self, examples):
        inputs = [
            # ajusta para o t5 invertendo o posicionamento do contexto com o input
            input.replace("\n\n###Response:", "\n\n###Context:\n")
            + context
            + "\n\n###Response: "
            for context, input in zip(examples["context"], examples["input"])
        ]
        targets = [sql for sql in examples["output"]]

        # Tokenize inputs e targets
        model_inputs = self.tokenizer(
            inputs, max_length=512, truncation=True, padding="max_length"
        )
        labels = self.tokenizer(
            targets, max_length=128, truncation=True, padding="max_length"
        ).input_ids

        model_inputs["labels"] = labels

        return model_inputs

    def __compute_metrics(self, eval_pred):
        predictions, labels = eval_pred

        # Ensure predictions and labels are within the allowable range
        predictions = np.clip(predictions, a_min=0, a_max=self.tokenizer.vocab_size - 1)
        labels = np.clip(labels, a_min=0, a_max=self.tokenizer.vocab_size - 1)

        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_preds = [
            "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
        ]
        decoded_labels = [
            "\n".join(nltk.sent_tokenize(label_.strip())) for label_ in decoded_labels
        ]

        result = self.rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )

        result = {key: value for key, value in result.items()}
        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id)
            for pred in predictions
        ]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def train(self):
        """Função responsável por treinar o modelo."""
        dataset = self.dataset_loader.load()

        # Aplicar o preprocessamento
        tokenized_datasets = dataset.map(
            self.__preprocess_data,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

        # 5. Definir os argumentos de treinamento
        logging_eval_steps = len(tokenized_datasets["train"]) // BATCH_SIZE

        train_args = Seq2SeqTrainingArguments(
            output_dir=OUTPUT_MODEL,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=1e-5,  # 5.6e-5
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            weight_decay=0.01,
            eval_steps=logging_eval_steps,
            logging_steps=logging_eval_steps,
            eval_strategy="epoch",
            predict_with_generate=True,
            report_to="none",
            save_total_limit=1,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            greater_is_better=True,
            push_to_hub=False,
            fp16=USE_FP16,
        )

        torch.cuda.empty_cache()  # limpa o cache do CUDA

        os.makedirs(OUTPUT_MODEL, exist_ok=True)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=self.model
        )

        train_encoded_dataset = tokenized_datasets["train"]
        validation_encoded_dataset = tokenized_datasets["validation"]

        print(f"Utilizando o dispositivo: {self.device}")

        # 6. Inicializar o Trainer
        self.model.to(self.device)

        ft_trainer = Seq2SeqTrainer(
            model=self.model,
            args=train_args,
            train_dataset=train_encoded_dataset,
            eval_dataset=validation_encoded_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.__compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=5),
                SaveModelByEpochCallback(),
            ],
        )

        # 7. Iniciar o treinamento
        if os.path.exists(OUTPUT_MODEL) and len(os.listdir(OUTPUT_MODEL)) > 0:
            ft_trainer.train(resume_from_checkpoint=True)
        else:
            ft_trainer.train()

        ft_trainer.evaluate()

        # 8. Salvar o modelo fine-tuned
        ft_trainer.save_model(OUTPUT_MODEL)

        print(f"\n\n***Model Saved in {OUTPUT_MODEL}")
        print("\n***Finetunning Complete!***")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
