import os
import argparse
import sys
sys.path.append('/home/sy/project/bert_ndcg_lp/')

from src.module import KRL

if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    print("Base path : {}".format(path))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pretrained_model_path',
        default=os.path.join(path, 'model/pretrained_model'),
        type=str,
        required=False,
        help='The path of pretrained model!'
    )
    parser.add_argument(
        "--model_path",
        default=os.path.join(path, 'model/kgc_model.bin'),
        type=str,
        required=False,
        help="The path of model!",
    )
    parser.add_argument(
        '--SPECIAL_TOKEN',
        default={"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]",
                 "additional_special_tokens":["[E1]", "[/E1]", "[RE]", "[/RE]", "[E2]", "[/E2]"]},
        type=dict,
        required=False,
        help='The dictionary of special tokens!'
    )
    parser.add_argument(
        '--LABEL2I',
        default={'0': 0,
                 '1': 1,},
        type=dict,
        required=False,
        help='The dictionary of label2i!'
    )
    parser.add_argument(
        "--train_path",
        default=os.path.join(path, 'data/umls/ptrain.csv'),
        type=str,
        required=False,
        help="The path of training set!",
    )
    parser.add_argument(
        '--dev_path',
        default=os.path.join(path, 'data/umls/pdev.csv'),
        type=str,
        required=False,
        help='The path of dev set!'
    )
    parser.add_argument(
        '--test_path',
        default=os.path.join(path, 'data/umls/ptest.csv'),
        type=str,
        required=False,
        help='The path of test set!'
    )
    parser.add_argument(
        '--log_path',
        default=None,
        type=str,
        required=False,
        help='The path of Log!'
    )
    parser.add_argument("--epochs", default=100, type=int, required=False, help="Epochs!")
    parser.add_argument(
        "--batch_size", default=1, type=int, required=False, help="Batch size!"
    )
    parser.add_argument('--step_size', default=10, type=int, required=False, help='lr_scheduler step size!')
    parser.add_argument("--lr", default=0.01, type=float, required=False, help="Learning rate!")
    parser.add_argument('--clip', default=1.0, type=float, required=False, help='Clip!')
    parser.add_argument("--weight_decay", default=0, type=float, required=False, help="Regularization coefficient!")
    parser.add_argument(
        "--max_length", default=50, type=int, required=False, help="Maximum text length!"
    )
    parser.add_argument('--score_func', default='l2', type=str, required=False, help='Euclidean distance norm！')
    parser.add_argument('--loss_margin', default=2.0, type=float, required=False, help='Loss margin!')
    parser.add_argument('--train', default='true', type=str, required=False, help='Train or predict!')
    parser.add_argument("--sigma", dest="sigma", type=float, default=1.0)
    parser.add_argument(
        "--ndcg_gain_in_train", dest="ndcg_gain_in_train",
        type=str, default="identity", choices=["exp2", "identity"]
    )
    parser.add_argument("--debug", type=bool, default=True, required=False, help='Debug!')

    args = parser.parse_args()
    train_bool = lambda x:x.lower() == 'true'
    re = KRL(args)
    if train_bool(args.train):
        re.train()
    else:
        re.load()
        entities = ['genetic function', 'organ or tissue function']
        predict_result = re.predict_tail('genetic function', 'process of', entities)
        predict_result = sorted(predict_result.items(), key=lambda x: x[1], reverse=True)
        print(predict_result[:10])

