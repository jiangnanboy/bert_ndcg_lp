"""
Common function used in training Learn to Rank
"""
from argparse import ArgumentParser, ArgumentTypeError
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .ltr_dataset import get_time, DataLoader
from .metrics import NDCG


def save_to_ckpt(ckpt_file, epoch, model, optimizer, lr_scheduler):
    ckpt_file = ckpt_file + '_{}'.format(epoch)
    print(get_time(), 'save to ckpt {}'.format(ckpt_file))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }, ckpt_file)

    print(get_time(), 'finish save to ckpt {}'.format(ckpt_file))

def eval_cross_entropy_loss(model, device, loader, epoch, tokenizer, max_length, writer=None, phase="Eval", sigma=1.0):
    """
    formula in https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf

    C = 0.5 * (1 - S_ij) * sigma * (si - sj) + log(1 + exp(-sigma * (si - sj)))
    when S_ij = 1:  C = log(1 + exp(-sigma(si - sj)))
    when S_ij = -1: C = log(1 + exp(-sigma(sj - si)))
    sigma can change the shape of the curve
    """
    print(get_time(), "{} Phase evaluate pairwise cross entropy loss".format(phase))
    model.eval()
    with torch.set_grad_enabled(False):
        total_cost = 0
        total_pairs = loader.get_num_pairs()
        pairs_in_compute = 0

        for heads, relations, tails, rels in loader.generate_batch_per_query(loader.df):
            rels = rels.reshape(-1, 1)
            rel_diff = rels - rels.T
            pos_pairs = (rel_diff > 0).astype(np.float32)
            num_pos_pairs = np.sum(pos_pairs, (0, 1))
            # skip negative sessions, no relevant info:
            if num_pos_pairs == 0:
                continue
            neg_pairs = (rel_diff < 0).astype(np.float32)
            num_pairs = 2 * num_pos_pairs  # num pos pairs and neg pairs are always the same
            pos_pairs = torch.tensor(pos_pairs, device=device)
            neg_pairs = torch.tensor(neg_pairs, device=device)
            Sij = pos_pairs - neg_pairs
            # only calculate the different pairs
            diff_pairs = pos_pairs + neg_pairs
            pairs_in_compute += num_pairs

            input_ids_list = list()
            input_mask_list = list()
            segment_ids_list = list()
            head_mask_list = list()
            rel_mask_list = list()
            tail_mask_list = list()

            for index in range(len(heads)):
                head, relation, tail, rel = heads[index], relations[index], tails[index], rels[index]
                single_sample_dict = loader.convert_examples_to_features(head, relation, tail, rel,
                                                                               tokenizer, max_length)
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

            label = label.to(device)
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            head_mask = head_mask.to(device)
            rel_mask = rel_mask.to(device)
            tail_mask = tail_mask.to(device)

            y_pred = model(input_idx=input_ids, attention_mask=input_mask, token_type_ids=segment_ids,
                                head_mask=head_mask, rel_mask=rel_mask, tail_mask=tail_mask)
            y_pred_diff = y_pred - y_pred.t()
            # logsigmoid(x) = log(1 / (1 + exp(-x))) equivalent to log(1 + exp(-x))
            C = 0.5 * (1 - Sij) * sigma * y_pred_diff - F.logsigmoid(-sigma * y_pred_diff)
            C = C * diff_pairs
            cost = torch.sum(C, (0, 1))
            if cost.item() == float('inf') or np.isnan(cost.item()):
                import ipdb; ipdb.set_trace()
            total_cost += cost

        assert total_pairs == pairs_in_compute
        avg_cost = total_cost / total_pairs
    print(
        get_time(),
        "Epoch {}: {} Phase pairwise cross entropy loss {:.6f}, total_paris {}".format(
            epoch, phase, avg_cost.item(), total_pairs
        ))
    if writer:
        writer.add_scalars('loss/cross_entropy', {phase: avg_cost.item()}, epoch)

def eval_ndcg_at_k(
        inference_model, device, df_valid, valid_loader, batch_size, k_list, epoch, tokenizer, max_length,
        writer=None, phase="Eval"
):
    # print("Eval Phase evaluate NDCG @ {}".format(k_list))
    ndcg_metrics = {k: NDCG(k) for k in k_list}
    qids_list, rels_list, scores_list = [], [], []
    inference_model.eval()
    with torch.no_grad():
        for qids, heads, relations, tails, rels in valid_loader.generate_query_batch(df_valid, batch_size):
            input_ids_list = list()
            input_mask_list = list()
            segment_ids_list = list()
            head_mask_list = list()
            rel_mask_list = list()
            tail_mask_list = list()

            for index in range(len(heads)):
                head, relation, tail, rel = heads[index], relations[index], tails[index], rels[index]

                single_sample_dict = DataLoader.convert_examples_to_features(head, relation, tail, rel, tokenizer,
                                                                             max_length)

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

            label = label.to(device)
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            head_mask = head_mask.to(device)
            rel_mask = rel_mask.to(device)
            tail_mask = tail_mask.to(device)

            y_tensor = inference_model(input_idx=input_ids, attention_mask=input_mask, token_type_ids=segment_ids,
                                head_mask=head_mask, rel_mask=rel_mask, tail_mask=tail_mask)
            scores_list.append(y_tensor.cpu().numpy().squeeze())
            qids_list.append(qids)
            rels_list.append(rels)

    qids_list = np.hstack(qids_list)
    rels_list = np.hstack(rels_list)
    scores_list = np.hstack(scores_list)
    result_df = pd.DataFrame({'qid': qids_list, 'rel': rels_list, 'score': scores_list})
    session_ndcgs = defaultdict(list)
    for qid in result_df.qid.unique():
        result_qid = result_df[result_df.qid == qid].sort_values('score', ascending=False)
        rel_rank = result_qid.rel.values
        for k, ndcg in ndcg_metrics.items():
            if ndcg.maxDCG(rel_rank) == 0:
                continue
            ndcg_k = ndcg.evaluate(rel_rank)
            if not np.isnan(ndcg_k):
                session_ndcgs[k].append(ndcg_k)

    ndcg_result = {k: np.mean(session_ndcgs[k]) for k in k_list}
    ndcg_result_print = ", ".join(["NDCG@{}: {:.5f}".format(k, ndcg_result[k]) for k in k_list])
    print(get_time(), "{} Phase evaluate {}".format(phase, ndcg_result_print))
    if writer:
        for k in k_list:
            writer.add_scalars("metrics/NDCG@{}".format(k), {phase: ndcg_result[k]}, epoch)
    return ndcg_result


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    """Common Args needed for different Learn to Rank training method.
    :rtype: ArgumentParser
    """
    parser = ArgumentParser(description="additional training specification")
    parser.add_argument("--start_epoch", dest="start_epoch", type=int, default=0)
    parser.add_argument("--additional_epoch", dest="additional_epoch", type=int, default=100)
    parser.add_argument("--lr", dest="lr", type=float, default=0.0001)
    parser.add_argument("--optim", dest="optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--leaky_relu", dest="leaky_relu", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument(
        "--ndcg_gain_in_train", dest="ndcg_gain_in_train",
        type=str, default="exp2", choices=["exp2","identity"]
    )
    parser.add_argument("--small_dataset", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--double_precision", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--standardize", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="/tmp/ranking_output/")
    return parser
