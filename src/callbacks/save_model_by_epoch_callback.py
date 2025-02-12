"""Módulo responsável por salvar o modelo a cada época."""

import os
from transformers import TrainerCallback
from src.env import SHOULD_SAVE_EPOCH


class SaveModelByEpochCallback(TrainerCallback):
    """Classe responsável por salvar o modelo a cada época."""

    def on_epoch_end(self, args, state, control, **kwargs):
        """ Método chamado ao final de cada época. """

        if SHOULD_SAVE_EPOCH:
            epoch = int(state.epoch)

            output_dir_epoch = os.path.join(args.output_dir, "epochs", f"epoch-{epoch}")

            os.makedirs(output_dir_epoch, exist_ok=True)

            kwargs["model"].save_pretrained(output_dir_epoch)
            kwargs["processing_class"].save_pretrained(output_dir_epoch)

            print(f"Modelo da época {epoch} salvo em {output_dir_epoch}")
