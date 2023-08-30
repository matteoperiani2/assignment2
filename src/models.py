from typing import List, Optional, Union, Tuple
from dataclasses import dataclass
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import AutoTokenizer, AutoModel, EncoderDecoderModel


@dataclass
class QAEncoderModelOutput(transformers.utils.ModelOutput):
    """
    Base class for outputs of question answering with rationale models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total rationale extraction loss is the sum of a Binary Cross-Entropy for the tokens in the passage.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rationale_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Rationale classification scores (before Sigmoid).
        yng_logits (`torch.FloatTensor` of shape `(batch_size, 3)`):
            Yes/No/Generative scores (before Sigmoid).
    """

    loss: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rationale_logits: torch.FloatTensor = None
    yng_logits: torch.FloatTensor = None


@dataclass
class QAEncoderDecoderModelOutput(transformers.utils.ModelOutput):
    """
    Class for [`QAEncoderDecoderModel`] outputs.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_rationale_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Encoder rationale classification scores (before Sigmoid).
        encoder_yng_logits (`torch.FloatTensor` of shape `(batch_size, 3)`):
            Encoder Yes/No/Generative scores (before Sigmoid).
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_rationale_logits: Optional[torch.FloatTensor] = None
    encoder_yng_logits: torch.FloatTensor = None


class QAEncoderDecoderModel(transformers.EncoderDecoderModel):
    def __init__(
        self,
        encoder: transformers.PreTrainedModel,
        decoder: transformers.PreTrainedModel,
        config: Optional[transformers.EncoderDecoderConfig] = None,
    ):
        super(QAEncoderDecoderModel, self).__init__(
            encoder=encoder, decoder=decoder, config=config
        )

    def tie_weights(self):
        # tie encoder & decoder if needed
        if self.config.tie_encoder_decoder:
            # tie encoder and decoder base model
            encoder_base_model_prefix = self.encoder.base_model_prefix
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.encoder._modules[encoder_base_model_prefix],
                self.decoder._modules[decoder_base_model_prefix],
                self.decoder.base_model_prefix,
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        passage_mask: Optional[torch.BoolTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        teacher_force: Optional[float] = None,
        rationale_labels: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Union[Tuple, transformers.modeling_outputs.Seq2SeqLMOutput]:
        kwargs_encoder = {
            argument: value
            for argument, value in kwargs.items()
            if not argument.startswith("decoder")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                passage_mask=passage_mask,
                return_dict=return_dict,
                teacher_force=teacher_force,
                rationale_labels=rationale_labels,
                **kwargs_encoder,
            )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            # labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        if not return_dict:
            return outputs + encoder_outputs[-2:]

        return QAEncoderDecoderModelOutput(
            **outputs,
            encoder_rationale_logits=encoder_outputs.rationale_logits,
            encoder_yng_logits=encoder_outputs.yng_logits,
        )


class QAEncoder(transformers.PreTrainedModel):
    base_model_prefix = "encoder"

    def __init__(self, encoder, config) -> None:
        super().__init__(config)
        self.config = config

        self.encoder = encoder
        self.rationale_head = TokenSelectionHead(config)
        self.yes_no_gen_head = YesNoGenHead(config)

        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        passage_mask: Optional[torch.Tensor] = None,
        # labels: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        teacher_force: Optional[float] = None,
        rationale_labels: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Union[Tuple, QAEncoderModelOutput]:
        assert passage_mask is not None, "Passage mask is required"

        outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            # labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        last_hidden_state = outputs[0]
        pooled_output = outputs[1]

        rationale_logits = self.rationale_head(last_hidden_state)

        passage_mask = passage_mask.unsqueeze(-1)
        p_rationale = torch.sigmoid(rationale_logits)
        # substitute the last_hidden_state[passage] with p_rationale * last_hidden_state[passage]
        # ideally, our network keeps only the span of the passage which represents the rationale
        if self.training:
            if teacher_force is not None:
                use_labels = torch.rand(rationale_labels.shape[0]) < teacher_force
                use_labels = use_labels.to(p_rationale.device)
                true_labels = rationale_labels.unsqueeze(-1).type(p_rationale.dtype)
                p_rationale = torch.where(use_labels.reshape(-1, 1, 1), true_labels, p_rationale)
        else:
            p_rationale = (p_rationale > self.config.p_rationale_threshold).type(p_rationale.dtype)

        weighted_passage_hidden_state = passage_mask * p_rationale * last_hidden_state
        qa_seq_hidden_state = (1 - passage_mask) * last_hidden_state
        last_hidden_state = weighted_passage_hidden_state + qa_seq_hidden_state

        yng_logits = self.yes_no_gen_head(weighted_passage_hidden_state, pooled_output)

        rationale_logits = rationale_logits.squeeze(-1)

        loss = None
        if not return_dict:
            output = (last_hidden_state,) + outputs[2:] + (rationale_logits,)
            return ((loss,) + output) if loss is not None else output

        return QAEncoderModelOutput(
            loss=loss,
            last_hidden_state=last_hidden_state,
            rationale_logits=rationale_logits,
            yng_logits=yng_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TokenSelectionHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.act_fn = nn.ReLU()
        self.hidden_to_logit = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        logits = self.hidden_to_logit(hidden_states)
        return logits


class YesNoGenHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.act_fn = nn.ReLU()
        self.hidden_to_logit = nn.Linear(config.hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=-2)

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(2 * config.hidden_size, 3)

    def forward(
        self,
        rationale_weighted_hidden_states: torch.FloatTensor,
        pooled_output: torch.FloatTensor,
    ) -> torch.FloatTensor:
        rationale_weighted_hidden_states = self.dense(rationale_weighted_hidden_states)
        rationale_weighted_hidden_states = self.act_fn(
            rationale_weighted_hidden_states
        )  # BxTxD
        logits = self.hidden_to_logit(rationale_weighted_hidden_states)  # BxTx1
        attention_scores = self.softmax(logits)  # BxTx1
        weighted_tokens = attention_scores * rationale_weighted_hidden_states  # BxTxD
        weighted_pooled_output = torch.sum(weighted_tokens, dim=-2)  # BxD
        pooled_output = torch.cat(
            (weighted_pooled_output, pooled_output), dim=-1
        )  # Bx2D

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


def initialize_cross_attention_layer_with_self_attention_layer(
    self_attention: nn.Module,
    cross_attention: nn.Module,
    cross_attention_layer_prefix: str,
):
    uninitialized_cross_attention_weights: List[str] = []
    if cross_attention.__class__ != self_attention.__class__:
        print(
            f"{cross_attention.__class__} and {self_attention.__class__} are not equal. In this case make sure that all encoder"
            " weights are correctly initialized."
        )

    def initialize_cross_attention_with_self_attention_recursively(
        self_attention_pointer: nn.Module,
        cross_attention_pointer: nn.Module,
        module_name: str,
        uninitialized_cross_attention_weights: List[str],
        depth=0,
    ):
        assert isinstance(self_attention_pointer, nn.Module) and isinstance(
            cross_attention_pointer, nn.Module
        ), f"{self_attention_pointer} and {cross_attention_pointer} have to be of type nn.Module"
        if hasattr(self_attention_pointer, "weight"):
            assert hasattr(cross_attention_pointer, "weight")
            cross_attention_pointer.weight.data = (
                self_attention_pointer.weight.data.clone().detach()
            )
            if hasattr(self_attention_pointer, "bias"):
                assert hasattr(cross_attention_pointer, "bias")
                cross_attention_pointer.bias.data = (
                    self_attention_pointer.bias.data.clone().detach()
                )
            return

        cross_attention_modules = cross_attention_pointer._modules
        self_attention_modules = self_attention_pointer._modules
        if len(self_attention_modules) > 0:
            assert (
                len(cross_attention_modules) > 0
            ), f"Cross-attention module {cross_attention_pointer} does not match self-attention module {self_attention_pointer}"

            all_cross_attention_weights = {
                module_name + "/" + sub_name
                for sub_name in cross_attention_modules.keys()
            }
            cross_attention_layer_pos = 0
            for name, module in self_attention_modules.items():
                if name.isdigit():
                    cross_attention_name = str(int(name) + cross_attention_layer_pos)
                    self_attention_name = name
                    if not isinstance(
                        self_attention_modules[self_attention_name],
                        type(cross_attention_modules[cross_attention_name]),
                    ) and len(cross_attention_modules) != len(self_attention_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        cross_attention_layer_pos -= 1
                        continue
                elif name not in cross_attention_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `initialize_cross_attention_with_self_attention` reached. It seems that there is"
                        " a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    self_attention_name = cross_attention_name = name
                initialize_cross_attention_with_self_attention_recursively(
                    self_attention_modules[self_attention_name],
                    cross_attention_modules[cross_attention_name],
                    module_name + "/" + name,
                    uninitialized_cross_attention_weights,
                    depth=depth + 1,
                )
                all_cross_attention_weights.remove(
                    module_name + "/" + cross_attention_name
                )

            uninitialized_cross_attention_weights += list(all_cross_attention_weights)

    # initialize weights recursively
    initialize_cross_attention_with_self_attention_recursively(
        self_attention,
        cross_attention,
        cross_attention_layer_prefix,
        uninitialized_cross_attention_weights,
    )
    if len(uninitialized_cross_attention_weights) > 0:
        warnings.warn(
            f"The following cross_attention weights were not initialized with self_attention weights: {uninitialized_cross_attention_weights}"
        )


def initialize_cross_attention_with_self_attention(model: EncoderDecoderModel):
    decoder_base_model_prefix = model.decoder.base_model_prefix
    for layer_idx in range(model.config.decoder.num_hidden_layers):
        decoder_layer = model.decoder._modules[decoder_base_model_prefix].encoder.layer[
            layer_idx
        ]
        cross_attention = decoder_layer.crossattention
        self_attention = decoder_layer.attention
        cross_attention_name = f"layer.{layer_idx}.crossattention"
        initialize_cross_attention_layer_with_self_attention_layer(
            self_attention, cross_attention, cross_attention_name
        )
    print("Cross-attention has been initialized with self-attention weights.")


def make_encoder_decoder_model(
    checkpoint,
    decoder_max_length,
    generation_kwargs,
    tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
    initialize_cross_attention=True,
):
    tokenizer, encoder = make_qa_encoder(checkpoint, tokenizer=tokenizer)
    decoder = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint,
        is_decoder=True,
        add_cross_attention=True,
        decoder_start_token_id=tokenizer.bos_token_id,
    )

    config = transformers.EncoderDecoderConfig.from_encoder_decoder_configs(
        encoder.config,
        decoder.config,
        tie_encoder_decoder=True,
        decoder_start_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # sensible parameters for generation
        vocab_size=decoder.config.vocab_size,
        max_new_tokens=decoder_max_length,
        **generation_kwargs,
    )

    model = QAEncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)
    if initialize_cross_attention:
        initialize_cross_attention_with_self_attention(model)

    return tokenizer, model


def make_qa_encoder(
    checkpoint, tokenizer: Optional[transformers.PreTrainedTokenizer] = None
):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    encoder = AutoModel.from_pretrained(checkpoint)
    encoder.config.p_rationale_threshold = 0.5
    encoder = QAEncoder(encoder, encoder.config)
    return tokenizer, encoder
