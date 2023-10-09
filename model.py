# %%

import json
import os
import shutil
from copy import deepcopy
from enum import Enum, auto
from pathlib import Path
from sys import platform
from typing import Any, Tuple
from zipfile import ZipFile
from deeppavlov import build_model, configs 
from deeppavlov.core.commands.utils import parse_config

class ModelCodename(Enum):
    TEST = auto()
    TEST2 = auto()
    ORIGIN = auto()

class BaseDD:
    def __init__(self, codename: ModelCodename):
        self.codename = codename


class JsonDD(BaseDD):
    def __init__(self, codename: ModelCodename): 
        super().__init__(codename)


class DDFactory:
    @staticmethod
    def create(dataset_codename: ModelCodename) -> BaseDD:
        return JsonDD(dataset_codename)


class BaseDataset:
    def __init__(self, data: Any, descriptor: BaseDD):
        self.data = data
        self.descriptor = descriptor


class JsonDataset:
    def __init__(self, data: list, view_config: str, descriptor: BaseDD):
        self.data = data
        self.view_config = view_config
        self.descriptor = descriptor


class DatasetKeeper:
    DATASET_DIR = "DS"
    DATASET_TASKS_FILENAME = "data.json"

    @staticmethod
    def load_dataset(dataset_descriptor: BaseDD) -> BaseDataset:
        filename = DatasetKeeper._get_filename(dataset_descriptor.codename)

        data, view_config = DatasetKeeper._load_json_dataset(filename)
        return JsonDataset(data, view_config, dataset_descriptor)


    @staticmethod
    def _load_json_dataset(filename) -> Tuple[list, str]:
        directory = os.path.dirname(filename)
        json_filename = os.path.join(directory, DatasetKeeper.DATASET_TASKS_FILENAME)
        with open(json_filename, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        return data, None

    @staticmethod
    def _get_filename(dataset_codename: ModelCodename) -> str:
        path = os.path.join(DatasetKeeper.DATASET_DIR, f"{dataset_codename}")
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, DatasetKeeper.DATASET_TASKS_FILENAME)
        return filename



class BaseMD:
    def __init__(self, model_codename: ModelCodename):
        self.model_codename = model_codename


class MDFactory:
    @staticmethod
    def create(model_codename: ModelCodename) -> BaseMD:
        return BaseMD(model_codename)



class BaseModel:
    def __init__(self, instance, descriptor: BaseMD):
        self.instance = instance
        self.descriptor = descriptor

    def __call__(self, *args, **kwargs):
        return self.instance(*args, **kwargs)


class ModelKeeper:
    MODEL_DIR = "models"
    TRAIN_PATH = "training"

    @staticmethod
    def load_model(model_descriptor: BaseMD) -> BaseModel:
        model_dir = ModelKeeper._make_model_dir(model_descriptor.model_codename)
        parsed_config = ModelKeeper._load_config(model_dir)
        instance = build_model(parsed_config)
        return BaseModel(instance, model_descriptor)

    @staticmethod
    def train_model(model_descriptor: BaseMD, dataset_descriptor: BaseDD, keep_train_files: bool) -> BaseModel:
        model_dir = ModelKeeper._make_model_dir(model_descriptor.model_codename)
        basic_config = ModelKeeper._load_config(model_dir)
        train_config = ModelKeeper._prepare_for_training(basic_config)
        dataset = DatasetKeeper.load_dataset(dataset_descriptor)

        from trainer import Trainer

        trainer = Trainer(keep_train_files, train_config, dataset, ModelKeeper.TRAIN_PATH)
        model = trainer.fit()
        return BaseModel(model, model_descriptor)

    @staticmethod
    def _load_config(model_dir) -> dict:
        #используем модель ner_rus_bert
        with open(configs.ner.ner_rus_bert, "r") as raw_config:
            wrong_config = json.load(raw_config)
        wrong_config["metadata"]["variables"]["MODEL_PATH"] = model_dir
        parsed_config = parse_config(wrong_config)
        return parsed_config

    @staticmethod
    def _prepare_for_training(config: dict) -> dict:
        # TODO Magic numbers
        config = deepcopy(config)
        config["train"]["epochs"] = 10  # 45
        config["train"]["batch_size"] = 10  # 14
        config["dataset_reader"]["data_path"] = ModelKeeper.TRAIN_PATH
        return config

    @staticmethod
    def _backup_model_files(model_dir):
        ext = "zip"
        base_name = os.path.basename(model_dir)
        shutil.make_archive(base_name, ext, model_dir)
        shutil.move(f"{base_name}.{ext}", Path(model_dir).parent)

    @staticmethod
    def _make_model_dir(model_codename: ModelCodename) -> str:
        model_path = os.path.join(ModelKeeper.MODEL_DIR, f"{model_codename}")
        os.makedirs(model_path, exist_ok=True)
        return model_path


#------------------------------------------------------------------------------------------------------------------------------------------------
#запускаем здесь

if __name__ == "__main__":

    
    train_model = True  # True - обучаем модель, False - просмотриваем результат
    keep_train_files = True # True - не обновляем файлы обучения в папке training, False - обновляем файлы обучения в папке training из Label Studio
    model_path = ModelCodename.TEST2 # model_path - наименование папки с моделью
    dd = DDFactory.create(model_path)
    dd.codename='test2' # codename - наименование папки с обучающим json
    ds = DatasetKeeper.load_dataset(dd)
    md = MDFactory.create(model_path)
    if train_model:
        # Обучение новой модели
        model = ModelKeeper.train_model(md, dd, keep_train_files)
        model.instance.save()
    else:
        model = ModelKeeper.load_model(md)
        
    print(model.instance(['королевич и Шрек сыграли свадьбу']))
    
    
