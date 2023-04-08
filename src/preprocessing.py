import re
import string
import transformers

from typing import List, Tuple

class CoQADatasetPreprocessing:

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer = None, max_length=512, stride=128, use_window=False) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.use_window = use_window

    def explode_questions(self, example):
        questions = example["questions"]
        answers = example["answers"]
        histories = []

        for idx in range(len(questions)):
            history = self.__create_history(idx, questions, answers)
            histories.append(history)

        return {
            "passage": [example["story"]] * example["qa_length"],
            "turn": [question_item["turn_id"] for question_item in questions],
            "question":
            [question_item["input_text"] for question_item in questions],
            "answer": [answer_item["input_text"] for answer_item in answers],
            "rationale": [answer_item["span_text"] for answer_item in answers],
            "span_start":
            [answer_item["span_start"] for answer_item in answers],
            "span_end": [answer_item["span_end"] for answer_item in answers],
            "answer_type":
            [answer_item["answer_type"] for answer_item in answers],
            "history":
            histories,
            "history_length": [len(history) for history in histories]
        }

    def __create_history(self, current_index, questions, answers):
        history = [{
            'question': questions[i]['input_text'],
            'answer': answers[i]['input_text'],
            'turn': questions[i]['turn_id']
        } for i in range(current_index)]
        return history

    def preprocess_texts(self, example):
        example = self.__fix_rationale(example)
        example = self.__fix_white_space(example)

        return example
    
    def __fix_rationale(self, example):
        rationale, span_start, span_end = fix_rationale(
            example["passage"], example["rationale"],
            example["span_start"], example["span_end"])

        example["rationale"] = rationale
        example["span_start"] = span_start
        example["span_end"] = span_end
        
        return example

    def __fix_white_space(self, example):
        example = self.__fix_passage_white_space(example)
        example["question"] = white_space_fix(example["question"])
        example["answer"] = white_space_fix(example["answer"])

        for item in example["history"]:
            item["question"] = white_space_fix(item["question"])
            item["answer"] = white_space_fix(item["answer"])

        return example

    def __fix_passage_white_space(self, example):
        passage = example["passage"]
        span_start = example["span_start"]
        span_end = example["span_end"]

        if span_end - span_start > 0:
            # assert rationale has already been fixed
            assert passage[span_start].isalnum() and passage[span_end - 1].isalnum(), \
            "Rationale must start and end with alphanumeric characters. You must fix it before."

        passage_start = white_space_fix(passage[:span_start])
        rationale = white_space_fix(passage[span_start:span_end])
        passage_end = white_space_fix(passage[span_end:])

        passage = " ".join((passage_start, rationale, passage_end))
        span_start = len(passage_start) + 1
        span_end = span_start + len(rationale)

        assert rationale == passage[span_start:span_end]

        example["passage"] = passage
        example["rationale"] = rationale
        example["span_start"] = span_start
        example["span_end"] = span_end

        return example
    
    def prepare_inputs(self, examples):
        assert self.tokenizer is not None, "A tokenizer is required to prepare the inputs for the model"
        inputs = self.tokenizer(
            examples["question"],
            examples["passage"],
            padding="max_length",
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=self.use_window,
            return_offsets_mapping=True,
        )

        offset_mapping = inputs["offset_mapping"]
        if self.use_window:
            sample_map = lambda i: inputs["overflow_to_sample_mapping"][i]
        else:
            sample_map = lambda i: i

        passage_masks = []
        rationale_starts = []
        rationale_ends = []
        rationale_masks = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map(i)
            start_char = examples["span_start"][sample_idx]
            end_char = examples["span_end"][sample_idx]
            sequence_ids = inputs.sequence_ids(i)

            passage_start, passage_end = self.__find_passage(sequence_ids)
            rationale_start, rationale_end = self.__char2token_rationale_span(offset, (passage_start, passage_end), (start_char, end_char))
            rationale_starts.append(rationale_start)
            rationale_ends.append(rationale_end)        
            
            passage_masks.append(self.__create_mask(sequence_ids, passage_start, passage_end+1))
            rationale_masks.append(self.__create_mask(sequence_ids, rationale_start, rationale_end))

        inputs["passage_mask"] = passage_masks
        inputs["rationale_start"] = rationale_starts
        inputs["rationale_end"] = rationale_ends
        inputs["rationale_mask"] = rationale_masks

        return inputs

    def __find_passage(self, sequence_ids: List[int]) -> Tuple[int, int]:
        """
        Find the start and the end of the passage w.r.t. the tokens.
        """

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        passage_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        passage_end = idx - 1

        return passage_start, passage_end

    def __char2token_rationale_span(
            self,
            token_to_span: List[Tuple[int, int]],
            passage_token_span: Tuple[int, int],
            rationale_char_span: Tuple[int, int]) -> Tuple[int, int]:
        """
        Map the rationale span from char indexes to token indexes
        """
        passage_start, passage_end = passage_token_span
        start_char, end_char = rationale_char_span

        # If the rationale is not fully inside the passage, returns (-1, -1)
        if token_to_span[passage_start][0] > start_char or token_to_span[
                passage_end][1] < end_char:
            return (-1, -1)

        # Otherwise it's the start and end token positions
        idx = passage_start
        while idx <= passage_end and token_to_span[idx][0] <= start_char:
            idx += 1
        start_position = idx - 1

        idx = passage_end
        while idx >= passage_start and token_to_span[idx][1] >= end_char:
            idx -= 1
        end_position = idx + 1

        return start_position, end_position + 1
    
    def __create_mask(self, arr, start, end):
        mask = [0] * len(arr)
        mask[start:end] = [1] * (end - start)
        return mask
    
def remove_articles_(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)


def white_space_fix(text: str):
    return ' '.join(text.split())


def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)


def lower(text):
    return text.lower()


def normalize_text(text, remove_articles=False):
    """Lower text and remove punctuation, articles and extra whitespace."""
    text = remove_punc(lower(text))
    if remove_articles:
        text = remove_articles_(text)
    return white_space_fix(text)


def normalize_answer(text):
    """Lower text and remove punctuation, articles and extra whitespace."""
    return normalize_text(text, remove_articles=True)


def strip_non_alphanumeric_chars(text: str):
    """
    Removes trailing and leading non alpha-numeric characters from a given string.
    """
    start_index = 0
    while (start_index < len(text) and not text[start_index].isalnum()):
        start_index += 1

    end_index = len(text) - 1
    while (end_index > start_index and not text[end_index].isalnum()):
        end_index -= 1

    return text[start_index:end_index + 1]


def find_span(passage: str,
              text: str,
              span_start: int = None,
              span_end: int = None):

    if len(text) == 0: return (span_start, span_start)
    assert text[0].isalnum() and text[-1].isalnum(), \
        "Text must begin and end with an alphanumeric character."

    start_idx = passage.find(text, span_start, span_end)
    end_idx = start_idx + len(text) - 1

    if start_idx == -1:
        raise ValueError("The text is not present in the passage.")

    # Find the beginning of the word in the passage
    while (start_idx > 0 and passage[start_idx - 1].isalnum()):
        start_idx -= 1

    # Find the end of the word in the passage
    while (end_idx < len(passage) - 1 and passage[end_idx + 1].isalnum()):
        end_idx += 1

    return start_idx, end_idx + 1


def fix_rationale(passage: str, rationale: str, span_start: int,
                  span_end: int):
    rationale = strip_non_alphanumeric_chars(rationale)
    span_start, span_end = find_span(passage,
                                     rationale,
                                     span_start=span_start,
                                     span_end=span_end)
    return passage[span_start:span_end], span_start, span_end

if __name__ == "__main__":
    from transformers import AutoTokenizer
    import datasets
    from src.config import *
    from src.utils import *
    CONFIG = Config()

    tokenizer = AutoTokenizer.from_pretrained(CONFIG.checkpoints.distil_roberta)
    preprocessing = CoQADatasetPreprocessing(tokenizer, use_window=True)

    raw_dataset = datasets.DatasetDict.load_from_disk(CONFIG.dataset.filtered_dir)
    train_dataset = raw_dataset["train"].select(range(10))

    train_dataset = train_dataset.map(
        batched_function(preprocessing.explode_questions, scalar_output=False),
        batched=True,
        remove_columns=raw_dataset["train"].column_names)
    train_dataset = train_dataset.map(batched_function(preprocessing.preprocess_texts), batched=True)

    
    data = train_dataset.select(range(2))
    inputs = preprocessing.prepare_inputs(data)

    idx = 0
    sample_idx = inputs["overflow_to_sample_mapping"][idx]

    input_ids = np.asarray(inputs["input_ids"][idx])
    passage_mask = np.asarray(inputs["passage_mask"][idx])
    rationale_mask = np.asarray(inputs["rationale_mask"][idx])
    rationale_start = inputs["rationale_start"][idx]
    rationale_end = inputs["rationale_end"][idx]

    passage = input_ids[passage_mask.astype(np.bool_)]
    rationale = input_ids[rationale_mask.astype(np.bool_)]