import inspect
import re
import numpy as np
from typing import List, Optional, Union
import numpy as np

import torch
from torchmetrics.classification import MulticlassF1Score
from tqdm import tqdm

import datasets
from accelerate import Accelerator

from src.preprocessing import CoQADatasetPreprocessing, idx_to_answer
from src.squad_f1 import compute_f1
from src.train import DynamicPaddingCollatorForSeq2Seq
from src.utils import batched_function, logits_to_class
from src.config import Config

CONFIG: Config = Config()


per_token_f1_metric = MulticlassF1Score(
    num_classes=2,
    average="macro",
    multidim_average="samplewise",
    ignore_index=-100,
)

macro_f1 = MulticlassF1Score(
    num_classes=3,
    average="macro",
    ignore_index=-100,
)

f1 = MulticlassF1Score(
    num_classes=3,
    average="none",
    ignore_index=-100,
)


wh = ["what", "when", "where", "which", "who", "how", "whose", "why"]


class QuestionAnswerer():
    def __init__(self,
            tokenizer,
            model,
            preprocessing, 
            device = torch.device("cpu")
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessing = preprocessing
        self.device = device

    def generate_answer(
            self,
            passage: Union[str, List[str]],
            question: Union[str, List[str]],
            history: Optional[Union[str, List[str]]] = None
    ) -> List[str]:
        use_history = history is not None
        preprocess = batched_function(self.preprocessing.preprocess_texts)
        
        if isinstance(passage, str):
            passage = [passage]
            question = [question]
            history = [history] if use_history else []

        inputs = {
            "id": list(range(len(passage))),
            "passage": passage,
            "question": question,
        }

        if use_history:
            inputs["history"] = history

        inputs = preprocess(inputs)
        inputs = self.preprocessing.process_data_to_model_inputs(
            inputs, add_history=use_history, padding="max_length")
        inputs = inputs.convert_to_tensors("pt")

        return self.generate_answer_from_input_tensors(inputs)

    def generate_answer_from_input_tensors(self, model_inputs):
        encoder = self.model.get_encoder()
        encoder_inputs = self.__prepare_model_inputs(encoder, model_inputs)

        model_inputs = self.__prepare_model_inputs(self.model, model_inputs)
        model_inputs.pop("decoder_input_ids", None)

        with torch.no_grad():
            encoder_outputs = encoder(**encoder_inputs, return_dict=True)
            model_outputs = self.model.generate(**model_inputs)

        answer_types = logits_to_class(encoder_outputs["yng_logits"], task="multiclass")
        output_str = self.tokenizer.batch_decode(model_outputs, skip_special_tokens=True)

        for i in range(len(output_str)):
            if answer_types[i] != 2:
                output_str[i] = idx_to_answer(answer_types[i])


        return {
            "pred_answer": output_str,
            "yng_logits": encoder_outputs["yng_logits"],
            "pred_yng_label": logits_to_class(
                encoder_outputs["yng_logits"], task="multiclass"
            ),
            "rationale_logits": encoder_outputs["rationale_logits"],
            "pred_rationale_labels": logits_to_class(
                encoder_outputs["rationale_logits"], task="binary"
            ),
        }

    def __prepare_model_inputs(self, model, inputs):
        forward_signature = set(inspect.signature(model.forward).parameters)
        inputs = {
            argument: value.to(model.device)
            for argument, value in inputs.items()
            if argument in forward_signature
        }
        return inputs


def evaluate_generation(model, tokenizer, data, config, use_history=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    preprocessing = CoQADatasetPreprocessing(tokenizer, **CONFIG.preprocessing.__dict__)
    question_answerer = QuestionAnswerer(tokenizer, model, preprocessing, device)

    if not use_history:
        outputs = data.map(
            lambda example: question_answerer.generate_answer(
                example["passage"], example["question"]
            ),
            batched=True,
            batch_size=config.get("batch_size", 0),
            load_from_cache_file=False
        )
    else:
        outputs = data.map(
            lambda example: question_answerer.generate_answer(
                example["passage"], example["question"], example["history"]
            ),
            batched=True,
            batch_size=config.get("batch_size", 0),
            load_from_cache_file=False
        )


    outputs = outputs.map(evaluate_answer,
                          load_from_cache_file=False)

    outputs = outputs.select_columns(["source",
                                      "passage",
                                      "question",
                                      "rationale",
                                      "answer",
                                      "pred_answer",
                                      "answer_type",
                                      "answer_f1",
                                    ])
    
    yes_answer = outputs.filter(lambda ex: "yes" in re.findall(r"[\w']+", ex["answer"].lower()), load_from_cache_file=False)
    no_answer = outputs.filter(lambda ex: "no" in re.findall(r"[\w']+", ex["answer"].lower()), load_from_cache_file=False)
    mc_question = outputs.filter(lambda ex: "or" in re.findall(r"[\w']+", ex["question"].lower()), load_from_cache_file=False)
    wh_question = outputs.filter(lambda ex: any(w in re.findall(r"[\w']+", ex["question"].lower()) for w in wh), load_from_cache_file=False)

    len_data = len(outputs)
    return outputs, {
        "tot_squad_f1":  (np.mean(outputs["answer_f1"]), 1),
        "yes_ans_f1": (np.mean(yes_answer["answer_f1"]), len(yes_answer) / len_data),
        "no_ans_f1":  (np.mean(no_answer["answer_f1"]), len(no_answer) / len_data),
        "mc_quest_f1":  (np.mean(mc_question["answer_f1"]), len(mc_question) / len_data),
        "wh_quest_f1":  (np.mean(wh_question["answer_f1"]), len(wh_question) / len_data)
    }


def evaluate_conversation(model, tokenizer, df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    preprocessing = CoQADatasetPreprocessing(tokenizer, **CONFIG.preprocessing.__dict__)
    question_answerer = QuestionAnswerer(tokenizer, model, preprocessing, device)

    
    conversations_results = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], leave=False):
        passage   = row['story']
        questions = [q['input_text'] for q in row['questions']]
        answers   = [a['input_text'] for a in row['answers']]
    
        answer_f1_scores = []   # f1-score of single answers within the dialogue

        pred_answer = []
        for quest, answ in zip(questions, answers):
            outputs = question_answerer.generate_answer(passage, quest)
            pred_answer.append(outputs["pred_answer"][0])
            answer_f1_scores.append(evaluate_answer({"answer": answ,
                                                     "pred_answer": pred_answer[-1]})["answer_f1"])

        results = {'source' : row['source'],
                    'passage': passage,
                    'questions' : questions,
                    'answers': answers,
                    'predicted_answers' : pred_answer,
                    'answers_f1_scores' : answer_f1_scores,
                    'conversation_f1_score' : np.mean(answer_f1_scores)
                    }

        conversations_results.append(results)

    return conversations_results


def print_worst_answers(conv_res):
    answers = [(ans,ans_f1)  for idx,answers in enumerate(conv_res["predicted_answers"]) for ans,ans_f1 in zip(answers, conv_res["answers_f1_scores"][idx]) if ans_f1 <= conv_res["conversation_f1_score"].min()]
    ans_idx = [idx for idx,obj in enumerate(answers) if obj[1] == min(answers)[1]]

    return np.random.choice(ans_idx, size=5) #return random 5 worst answers


def labels_to_answer(labels: torch.Tensor, tokenizer, ignore_index=-100) -> str:
    labels[labels == ignore_index] = tokenizer.pad_token_id
    answer = tokenizer.decode(labels, skip_special_tokens=True)
    return answer


def pad_input_tensors(inputs, collator):
    features = [dict(zip(inputs.keys(), values)) for values in zip(*inputs.values())]
    features = collator(features)

    return features


def evaluate_answer(example: dict):
    return {"answer_f1": compute_f1(example["answer"], example["pred_answer"])}


def evaluate_rationale_f1(example: dict):
    rationale_f1 = per_token_f1_metric(
        example["pred_rationale_labels"].long(),
        example["pred_rationale_labels"].long(),
        example["rationale_labels"].long(),
    )
    # Ensure it is an array, not a scalar
    if rationale_f1.dim() == 0:
        rationale_f1.unsqueeze_(dim=0)
    return {"rationale_f1": rationale_f1}
