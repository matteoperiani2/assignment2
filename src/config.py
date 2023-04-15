import os


class Config():

    class Dataset():
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
        final_dir: str = os.path.join(data_dir, "final")
        
        def final_no_history(self, checkpoint_name: str)-> str:
            return os.path.join(self.final_dir, "final_no_history", checkpoint_name)
        
        def final_with_history(self, checkpoint_name: str)-> str:
            return os.path.join(self.final_dir, "final_with_history", checkpoint_name)

    class Checkpoints():
        def __init__(self,
                     distil_roberta="distilroberta-base",
                     bert_tiny="prajjwal1/bert-tiny") -> None:
            self.distil_roberta = distil_roberta
            self.bert_tiny = bert_tiny

    dataset: Dataset = Dataset()
    checkpoints: Checkpoints = Checkpoints()