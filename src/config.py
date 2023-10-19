from dataclasses import dataclass
import os
from typing import Optional


@dataclass
class Config:
    @dataclass
    class Dataset:
        train_url: str = "https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json"
        test_url: str = "https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json"

        data_dir: str = "data"

        raw_dir: str = os.path.join(data_dir, "raw")
        train_data_raw: str = os.path.join(raw_dir, "train.json")
        test_data_raw: str = os.path.join(raw_dir, "test.json")

        annotated_dir: str = os.path.join(data_dir, "annotated")
        train_data_annotated: str = os.path.join(annotated_dir, "train.json")
        test_data_annotated: str = os.path.join(annotated_dir, "test.json")

        readable_dir: str = os.path.join(data_dir, "readable")
        train_data_readable: str = os.path.join(readable_dir, "train.txt")
        test_data_readable: str = os.path.join(readable_dir, "test.txt")

        filtered_dir: str = os.path.join(data_dir, "filtered")
        splitted_dir: str = os.path.join(data_dir, "splitted")
        processed_dir: str = os.path.join(data_dir, "processed")
        train_dir: str = os.path.join(data_dir, "train")

        def train(self, model_name: str, history:bool, split=""):
            if history:
                get_dataset_dir = self.train_with_history
            else:
                get_dataset_dir = self.train_no_history
            return get_dataset_dir(model_name, split=split)

        def train_no_history(self, model_name: str, split="") -> str:
            return os.path.join(self.train_dir, "train_no_history", model_name, split)

        def train_with_history(self, model_name: str, split="") -> str:
            return os.path.join(self.train_dir, "train_with_history", model_name, split)

    class Checkpoints:
        def __init__(
            self, distil_roberta="distilroberta-base", bert_tiny="prajjwal1/bert-tiny"
        ) -> None:
            self.distil_roberta = distil_roberta
            self.bert_tiny = bert_tiny

    class Models:
        def __init__(
            self,
            model_dir_name="models",
            checkpoint_dir_name="checkpoints",
            final_checkpoint_name="final.pt",
        ) -> None:
            self.__model_dir = model_dir_name
            self.__checkpoints_dir_name = checkpoint_dir_name
            self.__final_checkpoint_name = final_checkpoint_name

        def model_dir(self, model_name, history: Optional[bool] = None):
            if history is None:
                history_str = ""
            elif history:
                history_str = "history"
            else:
                history_str = "no_history"
            return os.path.join(self.__model_dir, model_name, history_str)

        def checkpoints_dir(self, model_name, history: Optional[bool], seed=None):
            checkpoint_dir =  os.path.join(
                self.model_dir(model_name, history=history), self.__checkpoints_dir_name
            )
            if seed is not None:
                checkpoint_dir = os.path.join(checkpoint_dir, str(seed))
            return checkpoint_dir

        def checkpoint(self, model_name, history: Optional[bool], seed=None):
            return os.path.join(
                self.checkpoints_dir(model_name=model_name, history=history, seed=seed),
                self.__final_checkpoint_name,
            )

    @dataclass
    class Preprocessing:
        encoder_max_length: int
        decoder_max_length: int
        stride: int = 196
        use_window: bool = False
        max_history_length: int = 4

    @dataclass
    class WandbConfig:
        """Specify the parameters of `wandb.init`"""

        project: str = "nlp_assignment2"
        entity: str = "nlp_assignment2"

    dataset: Dataset = Dataset()
    checkpoints: Checkpoints = Checkpoints()
    models: Models = Models()

    # remove all span answers longer than span_max_length words
    span_max_length: int = 37
    # ignore loss of rationales longer than rationale_max_length
    rationale_max_length: int = 150

    encoder_max_length = 512
    # decoder_max_length = 350
    decoder_max_length = 64

    preprocessing = Preprocessing(encoder_max_length, decoder_max_length)
    generation = dict(penalty_alpha=0.6, top_k=6)

    wandbConfig = WandbConfig()
