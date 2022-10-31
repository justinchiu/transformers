from copy import deepcopy

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration

from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device

from tests.test_modeling_common import floats_tensor, ids_tensor
from tests.models.bart.test_modeling_bart import (
    BartModelTest, BartModelTester, prepare_bart_inputs_dict,
)

model_test = BartModelTest()
model_test.setUp()
model_tester = model_test.model_tester

config, inputs = model_tester.prepare_config_and_inputs()

model = BartForConditionalGeneration(config=config).to(torch_device)
model.eval()

emb = model.get_input_embeddings()
embedding_dim = emb.embedding_dim

input_ids = inputs["input_ids"]
bsz, time = input_ids.shape
labels = ids_tensor([bsz, time], config.vocab_size).to(torch_device)

dots = floats_tensor((bsz, 7, 4))
W = nn.Linear(4, embedding_dim).to(torch_device)
dots_input = W(dots)
tokens_input = emb(input_ids)


output_tokens = model(input_ids=input_ids, labels=labels, return_dict=True)

inputs_embeds = torch.cat([dots_input, tokens_input], 1)
output_dots = model(inputs_embeds=inputs_embeds, labels=labels, return_dict=True)


class DotBartForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config, dot_encoder):
        super().__init__(config)
        self.dot_encoder = dot_encoder

    def forward(
        self,
        input_ids = None,
        dots = None,
        inputs_embeds = None,
        encoder_outputs = None,
        **kwargs,
    ):
        # maybe take this out 
        cond1 = input_ids is None and dots is None and inputs_embeds is not None
        cond2 = input_ids is not None and dots is not None and inputs_embeds is None

        if encoder_outputs is not None:
            return super().forward(
                encoder_outputs = encoder_outputs,
                **kwargs,
            )

        # let inputs_embeds short-circuit input_ids
        if inputs_embeds is None:
            emb = self.get_input_embeddings()
            embedding_dim = emb.embedding_dim

            dots_input = self.dot_encoder(dots)
            tokens_input = emb(input_ids)

            inputs_embeds = torch.cat([dots_input, tokens_input], 1)
            self.inputs_embeds = inputs_embeds

        return super().forward(
            inputs_embeds = inputs_embeds,
            **kwargs,
        )

model2 = DotBartForConditionalGeneration(config, W).to(torch_device)
model2.model = model.model
model2.lm_head = model.lm_head
model2.final_logits_bias = model.final_logits_bias
model2.eval()
out2 = model2(input_ids, dots, labels = labels)
out3 = model2(input_ids, dots, labels = labels)

output_dots2 = model(inputs_embeds=inputs_embeds, labels=labels, return_dict=True)

print(output_dots.loss)
print(output_dots2.loss)
print(out2.loss)
print(out3.loss)
print((inputs_embeds == model2.inputs_embeds).all())

gen1 = model.generate(inputs_embeds = inputs_embeds)
gen2 = model2.generate(inputs_embeds = inputs_embeds)

out3.loss.backward()
print(model2.dot_encoder.weight.grad)

import pdb; pdb.set_trace()
