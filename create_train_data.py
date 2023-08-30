import datasets
import numpy as np
from transformers import AutoTokenizer


from src.config import Config
from src.preprocessing import CoQADatasetPreprocessing


CONFIG = Config()

for checkpoint_name, checkpoint in CONFIG.checkpoints.__dict__.items():
    print("Processing data for:", checkpoint_name)
    dataset = datasets.load_from_disk(CONFIG.dataset.processed_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    preprocessing = CoQADatasetPreprocessing(tokenizer, **CONFIG.preprocessing.__dict__)
    dataset = dataset.map(
        preprocessing.process_data_to_model_inputs,
        fn_kwargs={"add_history": False},
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=None,
    )
    num_rows = dataset["train"].num_rows
    print("Number of rows in the training set before removing potentially unanswerable questions:", num_rows)
    dataset["train"] = dataset["train"].filter(lambda example: np.asarray(example["rationale_start"]) != -1, batched=True)
    print("Number of rows in the training set after removing potentially unanswerable questions:", dataset["train"].num_rows)
    removed_rows = num_rows - dataset["train"].num_rows
    removed_ratio = removed_rows / num_rows
    print(f"Removed rows: {removed_rows} ({removed_ratio:.2%})")
    dataset.save_to_disk(CONFIG.dataset.train_no_history(checkpoint_name))