import torch
import torch.nn as nn
from transformers import BertTokenizer, AlbertModel, AlbertConfig, AlbertForSequenceClassification
from src.utils.score_func import l1_score, l2_score

import sys

sys.path.append('/home/sy/project/bert_ndcg_lp/')

from src.utils.log import logger

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_tokenizer(model_path, special_token):
    logger.info('loading tokenizer {}'.format(model_path))
    tokenizer = BertTokenizer.from_pretrained(model_path)
    if special_token:
        tokenizer.add_special_tokens(special_token)

    return tokenizer

def load_config(pretrained_model_path, tokenizer):
    albertConfig = AlbertConfig.from_pretrained(pretrained_model_path,
                                                cls_token_id=tokenizer.cls_token_id,
                                                sep_token_id=tokenizer.sep_token_id,
                                                pad_token_id=tokenizer.pad_token_id,
                                                unk_token_id=tokenizer.unk_token_id,
                                                output_attentions=False,  # whether or not return [attentions weights]
                                                output_hidden_states=False)  # whether or not return [hidden states]
    return albertConfig

def load_pretrained_model(pretrained_model_path, tokenizer, special_token):
    logger.info('loading pretrained model {}'.format(pretrained_model_path))
    albertConfig = load_config(pretrained_model_path, tokenizer)
    model = AlbertModel.from_pretrained(pretrained_model_path, config=albertConfig)
    # model = AlbertForSequenceClassification.from_pretrained(pretrained_model_path, config=albertConfig, num_labels=1)
    if special_token:
        # resize special token
        model.resize_token_embeddings(len(tokenizer))

    return model, albertConfig

def build_model(albertConfig, tokenizer):
    logger.info('build albertmodel!')
    model = AlbertModel(config=albertConfig)
    model.resize_token_embeddings(len(tokenizer))
    return model

class AlbertFC(nn.Module):
    def __init__(self, config, pretrained_model, score_func):
        super(AlbertFC, self).__init__()
        self.config = config
        self.model = pretrained_model
        self.fc = nn.Linear(768, 1)
        self.relu = nn.ReLU()
        if score_func == 'l1':
            self.score_func = l1_score
        else:
            self.score_func = l2_score

    def forward(self, input_idx, token_type_ids, attention_mask, head_mask, rel_mask, tail_mask):
        outputs = self.model(input_ids=input_idx, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]  # last hidden_size of albert => [batch_size, seq_len, hidden_size]
        sequence_output = sequence_output[:, 0, :] # -> 拿到[CLS]
        sequence_output = self.fc(self.relu(sequence_output))
        return sequence_output
        '''
        # get head hidden_size, [batch_size * hidden_size]
        head_output = torch.masked_select(sequence_output, head_mask.unsqueeze(-1))
        # get rel hidden_size, [batch_size * hidden_size]
        rel_output = torch.masked_select(sequence_output, rel_mask.unsqueeze(-1))
        # get tail hidden_size, [batch_size * hidden_size]
        tail_output = torch.masked_select(sequence_output, tail_mask.unsqueeze(-1))

        batch_size, hidden_size = sequence_output.shape[0], sequence_output.shape[2]

        # [batch_size, hidden_size]
        head_output = head_output.reshape(batch_size, -1)
        # [batch_size, hidden_size]
        rel_output = rel_output.reshape(batch_size, -1)
        # [batch_size, hidden_size]
        tail_output = tail_output.reshape(batch_size, -1)
        # predict score (head & rel & tail)
        
        return self.score_func(head_output, rel_output, tail_output)
        '''

