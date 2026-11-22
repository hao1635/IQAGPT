'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from torch import nn
import torch
import torch.nn.functional as F
from models.ibot_vit import vit_small
from models.xbert import BertConfig, BertLMHeadModel
from models.predictor import TextGenerator


class XVLModel(nn.Module):
    def __init__(self,
                 tokenizer=None,
                 config=None,
                 temp=0.07
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.distill = config['distill']

        self.visual_encoder = vit_small(
            img_size=(config['image_res'], config['image_res']),
            patch_size=config['patch_size'],
            drop_path_rate=config['drop_path'],
            return_all_tokens=True,
        )

        # self.visual_encoder = vit_base(
        #     img_size=(config['image_res'], config['image_res']),
        #     patch_size=config['patch_size'],
        #     drop_path_rate=config['drop_path'],
        #     return_all_tokens=True,
        # )

        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertLMHeadModel(config=bert_config)

        self.beam_generator = TextGenerator(config, self.text_encoder)

        if self.distill:
            raise ValueError('This release supports the non-distilled IQAGPT model only.')
    def forward(self, image, answer=None, train=True, alpha=0):

        #ipdb.set_trace()

        bs = image.size(0)

        image_embeds = self.visual_encoder(image)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        if train:
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)

            answer_output = self.text_encoder(answer.input_ids,
                                              attention_mask=answer.attention_mask,
                                              encoder_hidden_states=image_embeds,
                                              encoder_attention_mask=image_atts,
                                              labels=answer_targets,
                                              return_dict=True,
                                              is_decoder=True,
                                              reduction='none',
                                              output_hidden_states=True
                                              )
            loss = answer_output.loss
            loss = loss.sum() / image.size(0)

            return loss

        else:
            topk_ids, topk_probs = self.generation(image_embeds, image_atts)

            return topk_ids, topk_probs


    def generation(self, question_states, question_atts, out_size=1):
        encoder_inputs = [question_states, question_atts]
        # topk_ids, topk_probs = self.beam_generator.translate_batch_scst(encoder_inputs, out_size=out_size)
        topk_ids, topk_probs = self.beam_generator.translate_batch(encoder_inputs, out_size=out_size)
        return topk_ids, topk_probs


