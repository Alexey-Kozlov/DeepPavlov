from deeppavlov import build_model, evaluate_model, configs 
from deeppavlov.core.commands.utils import parse_config
import json

with open(configs.ner.ner_rus_bert, "r") as raw_config:
    config = json.load(raw_config)
config["metadata"]["variables"]["MODEL_PATH"] = 'models/ModelCodename.TEST2';
model_config = parse_config(config)

#если нужно скачать исходную обученную модель
#ner_model = build_model(model_config, download=True, install=True)

ner_model = build_model(parse_config(model_config))

# оценка работы модели
# ner_model = evaluate_model(model_config)

print(ner_model(['королевич и Шрек сыграли свадьбу']))