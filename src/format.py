import json
import os
import re

from tqdm import tqdm
from src.env import OUTPUT_COMMERCIAL_MODEL_PREDICTIONS, OUTPUT_MODEL
from src.env import MODELS

def format(text):
    text = re.sub(r"System[\s\S]*?###Response:", " ", text)
    text = re.sub(r"I want you[\s\S]*?###Response:", " ", text)
    
    text = text.replace('"', "")
    text = text.replace("FROM", " FROM ")
    text = text.replace("WHERE", " WHERE ")
    text = text.replace("JOIN", " JOIN ")
    text = text.replace("LEFT", " LEFT ")
    text = text.replace("RIGHT", " RIGHT ")
    text = text.replace("ORDER BY", " ORDER BY")
    text = text.replace("GROUP BY", " GROUP BY ")
    text = text.replace("HAVING", " HAVING ")
    text = text.replace("INTERSECT", " INTERSECT ")
    text = text.replace("UNION", " UNION ")
    text = text.replace("EXCEPT", " EXCEPT ")
    text = text.replace("LIMIT", " LIMIT ")

    text = (
        text
        .replace("```sql", " ")
        .replace("```", " ")
        .replace("\n", " ")
    )
    # remove os espaços em branco duplicados
    text = " ".join(text.split())
    # remove os espaços em branco no início e no final da string
    text = text.strip()

    return text

if __name__ == "__main__":
    predictions = []

    model = MODELS["unsloth/gemma-2-2b-it-bnb-4bit"]
    DELAY = 20
    DIR = (
        OUTPUT_MODEL
        if model["local"] and not model["zero_shot"]
        else f"{OUTPUT_COMMERCIAL_MODEL_PREDICTIONS}{model['model'].replace(':', '-').replace('/', '-').replace('.', '-')}"
    )

    predictions_file = os.path.join(
        DIR,
        "predictions.json",
    )

    print(f"Carregando as predições do arquivo {predictions_file}")

    with open(predictions_file, "r", encoding="utf8") as f:
        try:
            predictions = json.load(f)
        except json.JSONDecodeError:
            predictions = []

    progress_bar = tqdm(
        total=len(predictions), desc="Formatando as Predições...", colour="red"
    )

    for pred in predictions:
        pred["sql_predicted"] = format(pred["sql_predicted"])

        # reescreve o arquivo
        with open(os.path.join(DIR, "predictions.json"), "w", encoding="utf8") as f:
            f.write(json.dumps(predictions, indent=4))

        progress_bar.update(1)
