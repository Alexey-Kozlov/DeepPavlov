# %%

from dataclasses import dataclass
from typing import Any

import model as mp  # TODO UGLY?
from nltk import word_tokenize


# класс отвечает за обучение/переобучение модели
class Trainer:
    def __init__(self, keep_trin_files: bool, train_config: dict, dataset: Any, base_dir: str = "training"):
        self.train_config = train_config
        self.dataset = dataset
        self.base_dir = base_dir
        self.keep_train_files = keep_trin_files

    def fit(self):
        convert_dataset_to_bio(self.keep_train_files, self.dataset, self.base_dir)
        from deeppavlov import train_model

        import os
        for temp_file in os.listdir(self.train_config["metadata"]["variables"]["MODEL_PATH"]):
            os.remove(os.path.join(self.train_config["metadata"]["variables"]["MODEL_PATH"], temp_file))
        #тренируем модель
        model = train_model(self.train_config, download=False)

        return model


# https://stackoverflow.com/questions/41912083/nltk-tokenize-faster-way
class LazyTokenizer:
    def __init__(self, language="russian"):
        self.language = language

    def __call__(self, text):
        return word_tokenize(text, language=self.language)


@dataclass
class TrainValidTestSplit:
    train: float = 0.9  # От 0 до train идет на обучение.
    valid: float = 1.0  # От train до valid на валидацию.
    # test:float=0.1 # От valid до 1 идет на тестирование.


@dataclass
class CompressedBio:
    tag: str = ""
    text: str = ""


@dataclass(order=True)
class TextBlock:
    start: int = 0
    end: int = 0
    text: str = ""
    tag: str = ""


tokenizer = LazyTokenizer()  # TODO


def fix_tag(index: int, tag: str) -> str:
    if tag == "O":
        return tag
    elif index == 0:
        return f"B-{tag}"
    else:
        return f"I-{tag}"



@dataclass
class TokenTag:
    token: str = ""
    tag: str = ""


# Принимает на вход compressed BIO: словарь с текстом и тегом.
def decompress_bio(value: CompressedBio) -> list:
    tag = value.tag
    text = value.text
    tokens = tokenizer(text)
    if not tokens:
        return []
    tags = [fix_tag(i, tag) for i in range(len(tokens))]
    assert len(tokens) == len(tags)
    return [TokenTag(token, tag) for token, tag in zip(tokens, tags)]


# Тут подстройка под формат label studio. 
def get_text_blocks(sample: dict, new_format: bool = True) -> list:
    text_blocks = []
    annotations_key = "annotations" if new_format else "completions"
    for completion in sample[annotations_key]:
        for result in completion["result"]:
            v = result["value"]
            tb = TextBlock(v["start"], v["end"], v["text"], v["labels"][0])  
            text_blocks.append(tb)
    return sorted(text_blocks)


def to_compressed_bio(sample: dict) -> list:
    blocks = get_text_blocks(sample)
    ner_text = list(sample["data"].keys())[0]
    text = sample["data"][ner_text]
    compressed_bios = []
    prev_end = 0
    for block in blocks:
        compressed_bios.append(CompressedBio("O", text[prev_end : block.start]))
        compressed_bios.append(CompressedBio(block.tag, block.text))
        prev_end = block.end
    if len(text) != prev_end:
        compressed_bios.append(CompressedBio("O", text[prev_end : len(text)]))
    return compressed_bios


def flatten(list_of_lists: list) -> list:
    return [item for sublist in list_of_lists for item in sublist]


def convert_dataset_to_bio(keep_train_files: bool, dataset: mp.JsonDataset, outpath: str, split: TrainValidTestSplit = TrainValidTestSplit()):
    converted_samples = []
    for sample in dataset.data:
        compressed_bios = to_compressed_bio(sample)
        converted_bios = flatten([decompress_bio(compressed_bio) for compressed_bio in compressed_bios])
        converted_samples.append(converted_bios)

    def dump_subset_to_file(filename: str, sample_subset):
        with open(filename, "w", encoding="utf8", newline="\n") as outfile:
            for converted_sample in sample_subset:
                converted_lines = [f"{v.token} {v.tag}\n" for v in converted_sample]
                for converted_line in converted_lines:
                    outfile.write(converted_line)
                outfile.write("\n")

    sample_count = len(converted_samples)
    if not keep_train_files:
        dump_subset_to_file(f"{outpath}/train.txt", converted_samples[: int(sample_count)])
        dump_subset_to_file(f"{outpath}/valid.txt", converted_samples[int(sample_count * split.train) : int(sample_count * split.valid)])
        dump_subset_to_file(f"{outpath}/test.txt", converted_samples[int(sample_count * split.valid) :])

    print("DONE")


if __name__ == "__main__":
    pass
