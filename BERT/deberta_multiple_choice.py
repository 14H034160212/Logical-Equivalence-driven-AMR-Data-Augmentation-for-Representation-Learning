"""DebertaForMultipleChoice — not present in HuggingFace transformers
(only DebertaV2 has one). This mirrors the RobertaForMultipleChoice
architecture: pool the [CLS]/[SEP] token via `DebertaModel`, project to
one logit per choice, then cross-entropy over choices.

Used by BERT/run_multiple_choice.py for ReClor / LogiQA fine-tuning of
DeBERTa-v1 contrastive-pretrained checkpoints (e.g. the v5 / v6
deberta-large checkpoints produced via run_glue_no_trainer.py).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import DebertaConfig, DebertaModel, DebertaPreTrainedModel
from transformers.modeling_outputs import MultipleChoiceModelOutput


class DebertaForMultipleChoice(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: DebertaConfig):
        super().__init__(config)
        self.deberta = DebertaModel(config)
        # DeBERTa uses ContextPooler with pooler_hidden_size + pooler_dropout
        from transformers.models.deberta.modeling_deberta import ContextPooler

        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.classifier = nn.Linear(output_dim, 1)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = nn.Dropout(drop_out)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = (
            attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        )
        flat_token_type_ids = (
            token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        )
        flat_position_ids = (
            position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        )
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.deberta(
            flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            position_ids=flat_position_ids,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled = self.pooler(encoder_layer)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
