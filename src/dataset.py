import numpy as np

import torch
from torch.utils.data import Dataset


class TokenizedDataset(Dataset):
    def __init__(self, df, tokenizer, encoder_max_len=512, decoder_max_len=64):
        super(TokenizedDataset).__init__()
        self.df = df
        self.tokenizer = tokenizer
        # self.history = history
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len

    def __len__(self):
        return len(self.df)

    def _relocate_rationale_idxs(self, sequence_ids, span_start, span_end, offset):
        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        passage_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        passage_end = idx - 1

        passage_mask = np.zeros_like(sequence_ids)
        passage_mask[passage_start : passage_end + 1] = 1

        if offset[passage_start][0] > span_start or offset[passage_end][1] < span_end:
            return passage_mask.tolist(), -1, -1

        # Otherwise it's the start and end token positions
        idx = passage_start
        while idx <= passage_end and offset[idx][0] <= span_start:
            idx += 1
        start_position = idx - 1

        idx = passage_end
        while idx >= passage_start and offset[idx][1] >= span_end:
            idx -= 1
        end_position = idx + 1

        return passage_mask.tolist(), start_position, end_position + 1

    def __getitem__(self, idx):
        row = self.df[idx]
        passage = row[1]
        # history = row.history
        question = row[3]
        answer = row[4]

        # INPUT:  [CLS] QUESTION [SEP] PASSAGE [SEP] [PAD]*
        input_encoding = self.tokenizer(
            question,
            passage,
            padding="max_length",
            truncation="only_second",
            return_offsets_mapping=True,
            max_length=self.encoder_max_len,
        )

        sequence_ids = input_encoding.sequence_ids()

        passage_mask, span_start, span_end = self._relocate_rationale_idxs(
            sequence_ids, row[6], row[7], input_encoding["offset_mapping"]
        )

        # OUTPUT: [CLS] ANSWER [SEP] [PAD]*
        output_encoding = self.tokenizer(
            answer,
            truncation=True,
            padding="max_length",
            max_length=self.decoder_max_len,
        )

        labels = output_encoding.input_ids

        # Ignore the loss of the [PAD] labels by setting them to -100
        labels = [
            -100 if token == self.tokenizer.pad_token_id else token for token in labels
        ]
        labels = torch.tensor(labels, dtype=torch.int64)

        rat_labels = np.zeros_like(passage_mask)
        if span_start != -1 and span_end != -1:
            rat_labels[span_start:span_end] = 1

        inputs = {
            "input_ids": torch.tensor(input_encoding.input_ids, dtype=torch.int64),
            "attention_mask": torch.tensor(
                input_encoding.attention_mask, dtype=torch.int64
            ),
            "passage_mask": torch.tensor(passage_mask, dtype=torch.int64),
            "rat_labels": torch.tensor(rat_labels, dtype=torch.float),
            #'token_type_ids': None,
            "decoder_attention_mask": torch.tensor(
                output_encoding.attention_mask, dtype=torch.int64
            ),
        }

        if "token_type_ids" in input_encoding:
            inputs["token_type_ids"] = torch.tensor(
                input_encoding.token_type_ids, dtype=torch.int64
            )

        return inputs, labels

# class TokenizedDataset(Dataset):
#     def __init__(self, df, tokenizer, encoder_max_len=512, decoder_max_len=64):
#         super(TokenizedDataset).__init__()
#         self.df = df
#         self.tokenizer = tokenizer
#         # self.history = history
#         self.encoder_max_len = encoder_max_len
#         self.decoder_max_len = decoder_max_len

#     def __len__(self):
#         return len(self.df["input_ids"])

#     def __getitem__(self, idx):
#         return {k:v[idx] for k,v in self.df.items()}

# class TokenizedDataset(Dataset):
#     def __init__(self, df, tokenizer, encoder_max_len=512, decoder_max_len=64):
#         super(TokenizedDataset).__init__()
#         self.df = df
#         self.tokenizer = tokenizer
#         # self.history = history
#         self.encoder_max_len = encoder_max_len
#         self.decoder_max_len = decoder_max_len

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df[idx]
#         # passage = row[1]
#         # # history = row.history
#         # question = row[3]
#         # answer = row[4]

#         # # INPUT:  [CLS] QUESTION [SEP] PASSAGE [SEP] [PAD]*
#         # input_encoding = self.tokenizer(
#         #     question,
#         #     passage,
#         #     padding="max_length",
#         #     truncation="only_second",
#         #     return_offsets_mapping=True,
#         #     max_length=self.encoder_max_len,
#         # )

#         # sequence_ids = input_encoding.sequence_ids()

#         # passage_mask, span_start, span_end = self._relocate_rationale_idxs(
#         #     sequence_ids, row[6], row[7], input_encoding["offset_mapping"]
#         # )

#         # # OUTPUT: [CLS] ANSWER [SEP] [PAD]*
#         # output_encoding = self.tokenizer(
#         #     answer,
#         #     truncation=True,
#         #     padding="max_length",
#         #     max_length=self.decoder_max_len,
#         # )

#         # labels = output_encoding.input_ids

#         # # Ignore the loss of the [PAD] labels by setting them to -100
#         # labels = [
#         #     -100 if token == self.tokenizer.pad_token_id else token for token in labels
#         # ]
#         # labels = torch.tensor(labels, dtype=torch.int64)

#         # rat_labels = np.zeros_like(passage_mask)
#         # if span_start != -1 and span_end != -1:
#         #     rat_labels[span_start:span_end] = 1

#         inputs = {
#             "input_ids": torch.tensor(row[0], dtype=torch.int64),
#             "token_type_ids": torch.tensor(
#                 row[1], dtype=torch.int64
#             ),
#             "attention_mask": torch.tensor(
#                 row[2], dtype=torch.int64
#             ),
#             "passage_mask": torch.tensor(row[3], dtype=torch.int64),
#             "rat_labels": torch.tensor(row[4], dtype=torch.float),
#             #'token_type_ids': None,
#             "decoder_attention_mask": torch.tensor(row[6], dtype=torch.int64
#             )
#         }

#         # if "token_type_ids" in row:
#         #     inputs["token_type_ids"] = torch.tensor(
#         #         row["token_type_ids"], dtype=torch.int64
#         #     )

#         return inputs, torch.tensor(row[5], dtype=torch.int64)
