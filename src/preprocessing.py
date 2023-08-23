import itertools
import re
import string
import transformers

import numpy as np

from typing import List, Tuple


def answer_to_idx(answer: str) -> int:
    if answer.lower() == "yes":
        return 0
    if answer.lower() == "no":
        return 1
    return 2


def idx_to_answer(idx: int) -> str:
    if idx == 0:
        return "yes"
    if idx == 1:
        return "no"
    return None


class CoQADatasetPreprocessing:
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer = None,
        label_pad_token_id=-100,
        encoder_max_length=512,
        decoder_max_length=350,
        stride=196,
        use_window=False,
        max_history_length=4,
    ) -> None:
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length
        self.stride = stride
        self.use_window = use_window
        self.max_history_length = max_history_length

    def explode_questions(self, example):
        questions = example["questions"]
        answers = example["answers"]
        histories = []

        for idx in range(len(questions)):
            history = self.__create_history(idx, questions, answers)
            histories.append(history)

        output = {
            "id": [example["id"]] * example["qa_length"],
            "turn": [question_item["turn_id"] for question_item in questions],
            "question": [question_item["input_text"] for question_item in questions],
            "answer": [answer_item["input_text"] for answer_item in answers],
            "rationale": [answer_item["span_text"] for answer_item in answers],
            "span_start": [answer_item["span_start"] for answer_item in answers],
            "span_end": [answer_item["span_end"] for answer_item in answers],
            "answer_type": [answer_item["answer_type"] for answer_item in answers],
            "history": histories,
            "history_length": [len(history) for history in histories],
        }
        for key, value in example.items():
            if key not in output:
                output[key] = [value] * example["qa_length"]
        return output

    def __create_history(self, current_index, questions, answers):
        history = [
            {
                "question": questions[i]["input_text"],
                "answer": answers[i]["input_text"],
                "turn": questions[i]["turn_id"],
            }
            for i in range(current_index)
        ]
        return history

    def preprocess_texts(self, example):
        handle_rationale = "rationale" in example
        if handle_rationale:
            example = self.__fix_rationale(example)

        example = self.__preprocess_passage(example, handle_rationale=handle_rationale)
        example = self.__preprocess_questions(example)
        example = self.__preprocess_answers(example)

        return example

    def __fix_rationale(self, example):
        rationale, span_start, span_end = fix_rationale(
            example["passage"],
            example["rationale"],
            example["span_start"],
            example["span_end"],
        )

        example["rationale"] = rationale
        example["span_start"] = span_start
        example["span_end"] = span_end

        return example

    def __preprocess_passage(self, example, handle_rationale=True):
        return self.__fix_passage_white_space(
            example, handle_rationale=handle_rationale
        )

    def __preprocess_questions(self, example):
        example["question"] = white_space_fix(example["question"])
        for item in example.get("history", []):
            item["question"] = white_space_fix(item["question"])
        return example

    def __preprocess_answers(self, example):
        if "answer" in example:
            example["answer"] = self.__preprocess_answer(example["answer"])
        for item in example.get("history", []):
            item["answer"] = self.__preprocess_answer(item["answer"])
        return example

    def __preprocess_answer(self, answer):
        answer = white_space_fix(answer)
        answer = strip_non_alphanumeric_chars(answer)
        return answer

    def __fix_passage_white_space(self, example, handle_rationale=True):
        passage = example["passage"]
        if handle_rationale:
            span_start = example["span_start"]
            span_end = example["span_end"]

            if span_end - span_start > 0:
                # assert rationale has already been fixed
                assert (
                    passage[span_start].isalnum() and passage[span_end - 1].isalnum()
                ), "Rationale must start and end with alphanumeric characters. You must fix it before."

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
        else:
            example["passage"] = white_space_fix(passage)

        return example

    def process_data_to_model_inputs(
        self,
        examples,
        add_history=False,
        padding=False,
    ) -> transformers.BatchEncoding:
        assert (
            self.tokenizer is not None
        ), "A tokenizer is required to prepare the inputs for the model"
        process_rationale = "rationale" in examples
        process_answer = "answer" in examples

        sentences = [examples["question"], examples["passage"]]

        if add_history:
            sentences[0] = self.__concat_history_and_question(
                examples["history"], examples["question"]
            )

        inputs = self.tokenizer(
            *sentences,
            padding=padding,
            truncation="only_second",
            max_length=self.encoder_max_length,
            stride=self.stride,
            return_overflowing_tokens=self.use_window,
            return_offsets_mapping=True,
        )
        if process_answer:
            outputs = self.tokenizer(
                examples["answer"],
                padding=padding,
                truncation=True,
                max_length=self.decoder_max_length,
            )

        offset_mapping = inputs["offset_mapping"]
        if self.use_window:
            sample_map = lambda i: inputs["overflow_to_sample_mapping"][i]
        else:
            sample_map = lambda i: i

        yes_no_types = []
        yng_labels = []

        passage_masks = []
        rationale_starts = []
        rationale_ends = []
        rationale_labels = []
        decoder_input_ids = []
        labels = []
        decoder_attention_masks = []

        ids = []
        turns = []

        # # store the presence of the rationale in the passage for at least one row
        # rationale_in_passage = [False] * len(examples["question"])
        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map(i)
            sequence_ids = inputs.sequence_ids(i)

            passage_start, passage_end = self.__find_passage(sequence_ids)
            passage_masks.append(
                self.__create_mask(sequence_ids, passage_start, passage_end + 1)
            )

            if process_rationale:
                start_char = examples["span_start"][sample_idx]
                end_char = examples["span_end"][sample_idx]
                rationale_start, rationale_end = self.__char2token_rationale_span(
                    offset, (passage_start, passage_end), (start_char, end_char)
                )
                rationale_starts.append(rationale_start)
                rationale_ends.append(rationale_end)
                rationale_labels_ = self.__create_mask(
                    sequence_ids, rationale_start, rationale_end, dtype=np.float32
                )
                rationale_labels_[passage_masks[-1] == 0] = self.label_pad_token_id
                rationale_labels.append(rationale_labels_)

                # rationale_in_passage[sample_idx] |= rationale_start != -1

            if process_answer:
                # Remove <eos> from decoder_input_ids
                decoder_input_ids_ = outputs.input_ids[sample_idx][:-1]
                # Remove <bos> from labels
                labels_ = outputs.input_ids[sample_idx].copy()[1:]
                labels_ = [
                    self.label_pad_token_id
                    if token == self.tokenizer.pad_token_id
                    else token
                    for token in labels_
                ]
                decoder_attention_mask = outputs.attention_mask[sample_idx][:-1]

                decoder_input_ids.append(decoder_input_ids_)
                labels.append(labels_)
                decoder_attention_masks.append(decoder_attention_mask)

                yng_label = answer_to_idx(examples["answer"][sample_idx])
                is_yes_no = yng_label < 2
                assert is_yes_no == (examples["answer_type"][sample_idx] == "yes_no")
                yng_labels.append(yng_label)
                yes_no_types.append(int(is_yes_no))

            ids.append(examples["id"][sample_idx])
            if "turn" in examples:
                turns.append(examples["turn"][sample_idx])

        # if process_rationale:
        #     for sample_idx, is_rationale_in_passage in enumerate(rationale_in_passage):
        #         if not is_rationale_in_passage:
        #             warnings.warn(f"The rationale is never contained in the passage. Id: {examples['id'][sample_idx]}, turn:{examples['turn'][sample_idx]}")

        inputs["passage_mask"] = passage_masks
        if process_rationale:
            inputs["rationale_start"] = rationale_starts
            inputs["rationale_end"] = rationale_ends
            inputs["rationale_labels"] = rationale_labels
        if process_answer:
            inputs["decoder_input_ids"] = decoder_input_ids
            inputs["labels"] = labels
            inputs["decoder_attention_mask"] = decoder_attention_masks
            inputs["yng_label"] = yng_labels
            inputs["yes_no"] = yes_no_types
        inputs["id"] = ids
        if len(turns) > 0:
            inputs["turn"] = turns

        return inputs

    def __concat_history_and_question(self, histories, questions):
        outputs = []
        for history, question in zip(histories, questions):
            history_str = self.__create_history_str(history)
            history_question = self.tokenizer.sep_token.join((history_str, question))
            outputs.append(history_question)
        return outputs

    def __create_history_str(self, history):
        history_items = reversed(
            tuple(itertools.islice((reversed(history)), self.max_history_length))
        )
        qa_pairs = []
        for item in history_items:
            qa = (item["question"], item["answer"])
            qa = self.tokenizer.sep_token.join(qa)
            qa_pairs.append(qa)
        return self.tokenizer.sep_token.join(qa_pairs)

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
        rationale_char_span: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        Map the rationale span from char indexes to token indexes
        """
        passage_start, passage_end = passage_token_span
        start_char, end_char = rationale_char_span

        # If the rationale is not fully inside the passage, returns (-1, -1)
        if (
            token_to_span[passage_start][0] > start_char
            or token_to_span[passage_end][1] < end_char
        ):
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

    def __create_mask(self, arr, start, end, dtype=np.int8):
        mask = np.zeros_like(arr, dtype=dtype)
        mask[start:end] = 1
        return mask


def remove_articles_(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def white_space_fix(text: str):
    return " ".join(text.split())


def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


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
    while start_index < len(text) and not text[start_index].isalnum():
        start_index += 1

    end_index = len(text) - 1
    while end_index > start_index and not text[end_index].isalnum():
        end_index -= 1

    return text[start_index : end_index + 1]


def find_span(passage: str, text: str, span_start: int = None, span_end: int = None):
    if len(text) == 0:
        return (span_start, span_start)
    assert (
        text[0].isalnum() and text[-1].isalnum()
    ), "Text must begin and end with an alphanumeric character."

    start_idx = passage.find(text, span_start, span_end)
    end_idx = start_idx + len(text) - 1

    if start_idx == -1:
        raise ValueError("The text is not present in the passage.")

    # Find the beginning of the word in the passage
    while start_idx > 0 and passage[start_idx - 1].isalnum():
        start_idx -= 1

    # Find the end of the word in the passage
    while end_idx < len(passage) - 1 and passage[end_idx + 1].isalnum():
        end_idx += 1

    return start_idx, end_idx + 1


def fix_rationale(passage: str, rationale: str, span_start: int, span_end: int):
    rationale = strip_non_alphanumeric_chars(rationale)
    span_start, span_end = find_span(
        passage, rationale, span_start=span_start, span_end=span_end
    )
    return passage[span_start:span_end], span_start, span_end


# if __name__ == "__main__":
#     from transformers import AutoTokenizer
#     import datasets
#     from config import *
#     from utils import *
#     CONFIG = Config()

#     tokenizer = AutoTokenizer.from_pretrained(CONFIG.checkpoints.distil_roberta)
#     preprocessing = CoQADatasetPreprocessing(tokenizer, use_window=True)

#     raw_dataset = datasets.DatasetDict.load_from_disk(CONFIG.dataset.filtered_dir)
#     train_dataset = raw_dataset["train"].select(range(10))

#     train_dataset = train_dataset.map(
#         batched_function(preprocessing.explode_questions, scalar_output=False),
#         batched=True,
#         remove_columns=raw_dataset["train"].column_names)
#     train_dataset = train_dataset.map(batched_function(preprocessing.preprocess_texts), batched=True)


#     data = train_dataset.select(range(2))
#     inputs = preprocessing.prepare_inputs(data)

#     idx = 0
#     sample_idx = inputs["overflow_to_sample_mapping"][idx]

#     input_ids = np.asarray(inputs["input_ids"][idx])
#     passage_mask = np.asarray(inputs["passage_mask"][idx])
#     rationale_mask = np.asarray(inputs["rationale_mask"][idx])
#     rationale_start = inputs["rationale_start"][idx]
#     rationale_end = inputs["rationale_end"][idx]

#     passage = input_ids[passage_mask.astype(np.bool_)]
#     rationale = input_ids[rationale_mask.astype(np.bool_)]
