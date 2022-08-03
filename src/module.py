import torch
import random
import os
import numpy as np
import pandas as pd

import sys
sys.path.append('/home/sy/project/bert_ndcg_lp/')

from src.utils.log import logger

from .model import AlbertFC, load_tokenizer, load_config, load_pretrained_model, build_model

from .dataset import GetDataset
from .ltr_dataset import DataLoader, get_time
from .metrics import NDCG
from .util import eval_cross_entropy_loss, eval_ndcg_at_k, save_to_ckpt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    '''
    set seed
    :param seed:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(2021)

class KRL():
    '''
    re
    '''
    def __init__(self, args):
        self.args = args
        self.SPECIAL_TOKEN = args.SPECIAL_TOKEN
        self.label2i = args.LABEL2I
        self.model = None
        self.tokenizer = None

    def train(self):
        self.tokenizer = load_tokenizer(self.args.pretrained_model_path, self.SPECIAL_TOKEN)
        pretrained_model, albertConfig = load_pretrained_model(self.args.pretrained_model_path, self.tokenizer, self.SPECIAL_TOKEN)

        train_loader = DataLoader(self.args.train_path)
        df_train = train_loader.load()

        if self.args.dev_path:
            dev_loader = DataLoader(self.args.dev_path)
            df_valid = dev_loader.load()

        albertfc = AlbertFC(albertConfig, pretrained_model, self.args.score_func)
        self.model = albertfc.to(DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=0.75)

        ideal_dcg = NDCG(2**9, self.args.ndcg_gain_in_train)

        for epoch in range(self.args.epochs):
            self.model.train()
            self.model.zero_grad()

            count = 0
            grad_batch, y_pred_batch = list(), list()
            for heads, relations, tails, rels in train_loader.generate_batch_per_query():
                if np.sum(rels) == 0:
                    # negative session, connot learn userful signal
                    continue
                N = 1.0 / ideal_dcg.maxDCG(rels)

                input_ids_list = list()
                input_mask_list = list()
                segment_ids_list = list()
                head_mask_list = list()
                rel_mask_list = list()
                tail_mask_list = list()

                for index in range(len(heads)):
                    head, relation, tail, rel = heads[index], relations[index], tails[index], rels[index]

                    single_sample_dict = DataLoader.convert_examples_to_features(head, relation, tail, rel, self.tokenizer, self.args.max_length)

                    input_ids_list.append(single_sample_dict['input_ids'])
                    input_mask_list.append(single_sample_dict['input_mask'])
                    segment_ids_list.append(single_sample_dict['segment_ids'])
                    head_mask_list.append(single_sample_dict['head_mask'])
                    rel_mask_list.append(single_sample_dict['rel_mask'])
                    tail_mask_list.append(single_sample_dict['tail_mask'])

                label = torch.tensor(rels)
                input_ids = torch.tensor(input_ids_list)
                input_mask = torch.tensor(input_mask_list)
                segment_ids = torch.tensor(segment_ids_list)
                head_mask = torch.BoolTensor(head_mask_list)
                rel_mask = torch.BoolTensor(rel_mask_list)
                tail_mask = torch.BoolTensor(tail_mask_list)

                label = label.to(DEVICE)
                input_ids = input_ids.to(DEVICE)
                input_mask = input_mask.to(DEVICE)
                segment_ids = segment_ids.to(DEVICE)
                head_mask = head_mask.to(DEVICE)
                rel_mask = rel_mask.to(DEVICE)
                tail_mask = tail_mask.to(DEVICE)

                y_pred = self.model(input_idx=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, head_mask=head_mask, rel_mask=rel_mask, tail_mask=tail_mask)
                # y_pred = torch.exp(-y_pred)
                y_pred_batch.append(y_pred)

                # compute the rank order of each doc
                rank_df = pd.DataFrame({"Y":rels, "doc":np.arange(rels.shape[0])})
                # order the doc using the relevance score, higher score's order rank's higher
                rank_order = np.argsort(-rank_df["Y"]) + 1

                with torch.no_grad():
                    pos_pairs_score_diff = 1.0 + torch.exp(self.args.sigma * (y_pred - y_pred.t()))

                    Y_tensor = torch.tensor(rels).to(DEVICE).reshape(-1, 1)
                    rel_diff = Y_tensor - Y_tensor.t()
                    pos_pairs = (rel_diff > 0).type(torch.float32)
                    neg_pairs = (rel_diff < 0).type(torch.float32)
                    Sij = pos_pairs - neg_pairs
                    if self.args.ndcg_gain_in_train == 'exp2':
                        gain_diff = torch.pow(2.0, Y_tensor) - torch.pow(2.0, Y_tensor.t())
                    elif self.args.ndcg_gain_in_train == 'identity':
                        gain_diff = Y_tensor - Y_tensor.t()
                    else:
                        raise ValueError('ndcg_gain method not supported yes {}'.format(self.args.ndcg_gain_in_train))

                    rank_order_tensor = torch.tensor(rank_order).to(DEVICE).reshape(-1, 1)
                    decay_diff = 1.0 / torch.log2(rank_order_tensor + 1.0) - 1.0 / torch.log2(rank_order_tensor.t() + 1.0)

                    delta_ndcg = torch.abs(N * gain_diff * decay_diff)
                    lambda_update = self.args.sigma * (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
                    lambda_update = torch.sum(lambda_update, 1, keepdim=True)

                    assert lambda_update.squeeze().shape == y_pred.shape
                    check_grad = torch.sum(lambda_update, (0, 1)).item()
                    if check_grad == float('inf') or np.isnan(check_grad):
                        import ipdb; ipdb.set_trace()
                    grad_batch.append(lambda_update)

                count += 1
                if count % self.args.batch_size == 0:
                    for grad, y_pred in zip(grad_batch, y_pred_batch):
                        y_pred.backward(grad.squeeze() / self.args.batch_size)

                    optimizer.step()
                    self.model.zero_grad()
                    # grad_batch, y_pred_batch used for gradient_acc
                    grad_batch, y_pred_batch = list(), list()

            print(get_time(), 'training dataset at epoch {}, total rels: {}'.format(epoch, count))
            if self.args.debug:
                eval_cross_entropy_loss(self.model, DEVICE, train_loader, epoch, tokenizer=self.tokenizer, max_length=self.args.max_length, phase='Train')
                eval_ndcg_at_k(self.model, DEVICE, df_train, train_loader, 100000, [1, 3, 5, 10], epoch, tokenizer=self.tokenizer, max_length=self.args.max_length)

            # if epoch % 5 == 0 and epoch != 0:
            #     print(get_time(), 'eval for epoch: {}'.format(epoch))
            #     eval_cross_entropy_loss(self.model, DEVICE, dev_loader, epoch, tokenizer=self.tokenizer, max_length=self.args.max_length)
            #     eval_ndcg_at_k(self.model, DEVICE, df_valid, dev_loader, 100000, [10, 30], epoch, tokenizer=self.tokenizer, max_length=self.args.max_length)

            # if epoch % 10 == 0 and epoch != 0:
            #     save_to_ckpt(self.args.model_path, epoch, self.model, optimizer, scheduler)

            scheduler.step()
        # save the last ckpt
        # save_to_ckpt(self.args.model_path, self.args.epochs, self.model, optimizer, scheduler)

        # save the final model
        torch.save(self.model.state_dict(), self.args.model_path)
        # ndcg_result = eval_ndcg_at_k(self.model, DEVICE, df_valid, dev_loader, 100000, [10, 30], self.args.epochs)
        # print(get_time(), 'finish training ' + ', '.join(['NDCG@{}: {:.5f}'.format(k, ndcg_result[k]) for k in ndcg_result]), '\n\n')

    def predict_tail(self, head, rel, entity_list):
        '''
        predict tail entity
        :param head:
        :param rel:
        :param entity_list:
        :param topk:
        :return:
        '''
        self.model.eval()
        tail_list = []
        predict_result = {}
        for tail in entity_list:
            tmp_triple = [head, rel, tail, 1]
            if tmp_triple not in tail_list:
                tail_list.append(tmp_triple)
                tmp_dict = DataLoader.convert_examples_to_features(head, rel, tail, 1, self.tokenizer, self.args.max_length)

                single_input_ids = tmp_dict['input_ids']
                single_attention_mask = tmp_dict['input_mask']
                single_segment_ids = tmp_dict['segment_ids']

                single_head_mask = tmp_dict['head_mask']
                single_rel_mask = tmp_dict['rel_mask']
                single_tail_mask = tmp_dict['tail_mask']

                single_input_ids = torch.tensor(single_input_ids)
                single_attention_mask = torch.tensor(single_attention_mask)
                single_segment_ids = torch.tensor(single_segment_ids)
                single_head_mask = torch.BoolTensor(single_head_mask)
                single_rel_mask = torch.BoolTensor(single_rel_mask)
                single_tail_mask = torch.BoolTensor(single_tail_mask)

                single_input_ids = single_input_ids.unsqueeze(0)
                single_attention_mask = single_attention_mask.unsqueeze(0)
                single_segment_ids = single_segment_ids.unsqueeze(0)

                single_head_mask = single_head_mask.unsqueeze(0)
                single_rel_mask = single_rel_mask.unsqueeze(0)
                single_tail_mask = single_tail_mask.unsqueeze(0)

                single_input_ids = single_input_ids.to(DEVICE)
                single_attention_mask = single_attention_mask.to(DEVICE)
                single_segment_ids = single_segment_ids.to(DEVICE)

                single_head_mask = single_head_mask.to(DEVICE)
                single_rel_mask = single_rel_mask.to(DEVICE)
                single_tail_mask = single_tail_mask.to(DEVICE)

                with torch.no_grad():
                    out = self.model(input_idx=single_input_ids, attention_mask=single_attention_mask,
                                         token_type_ids=single_segment_ids, head_mask=single_head_mask, rel_mask=single_rel_mask,
                                         tail_mask=single_tail_mask)
                    tail_score = torch.exp(-out)
                    predict_result[tail] = tail_score.item()

        return predict_result

    def load(self):
        '''
        load model
        :return:
        '''
        self.tokenizer = load_tokenizer(self.args.pretrained_model_path, self.SPECIAL_TOKEN)
        albertConfig = load_config(self.args.pretrained_model_path, self.tokenizer)
        albert_model = build_model(albertConfig, self.tokenizer)
        self.model = AlbertFC(albertConfig, albert_model, self.args.score_func)

        # self.model = torch.load(self.args.model_path, map_location=DEVICE)
        self.model.load_state_dict(torch.load(self.args.model_path, map_location=DEVICE))
        logger.info('loading model {}'.format(self.args.model_path))
        self.model = self.model.to(DEVICE)
        print('load model done!')

