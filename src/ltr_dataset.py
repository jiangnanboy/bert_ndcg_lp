"""
Microsoft Learning to Rank Dataset:
https://www.microsoft.com/en-us/research/project/mslr/
"""
import datetime

import pandas as pd
import numpy as np

def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class DataLoader:

    def __init__(self, path):
        """
        :param path: str
        """
        self.path = path
        self.df = None
        self.num_pairs = None
        self.num_sessions = None

    def get_num_pairs(self):
        if self.num_pairs is not None:
            return self.num_pairs
        self.num_pairs = 0
        for _, _, _, Y in self.generate_batch_per_query(self.df):
            Y = Y.reshape(-1, 1)
            pairs = Y - Y.T
            pos_pairs = np.sum(pairs > 0, (0, 1))
            neg_pairs = np.sum(pairs < 0, (0, 1))
            assert pos_pairs == neg_pairs
            self.num_pairs += pos_pairs + neg_pairs

        return self.num_pairs

    def get_num_sessions(self):
        return self.num_sessions

    def _load_mslr(self):
        print(get_time(), "load file from {}".format(self.path))
        df = pd.read_csv(self.path, sep=",", header=None)
        print(get_time(), "finish loading from {}".format(self.path))
        print("dataframe shape: {}".format(df.shape))
        # 返回相关性，qid，特征
        return df

    def _parse_feature_and_label(self, df):
        """
        :param df: pandas.DataFrame
        :return: pandas.DataFrame
        """
        print(get_time(), "parse dataframe ...", df.shape)
        # 增加对应的列名
        df.columns = ['rel', 'qid', 'head', 'relation', 'tail']
        print(get_time(), "finish parsing dataframe")
        self.df = df
        # qid个数
        self.num_sessions = len(df.qid.unique())
        return df

    def generate_query_pairs(self, df, qid):
        """
        :param df: pandas.DataFrame, contains column qid, rel, fid from 1 to self.num_features
        :param qid: query id
        :returns: numpy.ndarray of x_i, y_i, x_j, y_j
        """
        df_qid = df[df.qid == qid]
        rels = df_qid.rel.unique()
        x_i, x_j, y_i, y_j = [], [], [], []
        for r in rels:
            df1 = df_qid[df_qid.rel == r]
            df2 = df_qid[df_qid.rel != r]
            df_merged = pd.merge(df1, df2, on='qid')
            df_merged.reindex(np.random.permutation(df_merged.index))
            y_i.append(df_merged.rel_x.values.reshape(-1, 1))
            y_j.append(df_merged.rel_y.values.reshape(-1, 1))
            x_i.append(df_merged[['{}_x'.format(i) for i in range(1, self.num_features + 1)]].values)
            x_j.append(df_merged[['{}_y'.format(i) for i in range(1, self.num_features + 1)]].values)
        return np.vstack(x_i), np.vstack(y_i), np.vstack(x_j), np.vstack(y_j)

    def generate_query_pair_batch(self, df=None, batchsize=2000):
        """
        :param df: pandas.DataFrame, contains column qid
        :returns: numpy.ndarray of x_i, y_i, x_j, y_j
        """
        if df is None:
            df = self.df
        x_i_buf, y_i_buf, x_j_buf, y_j_buf = None, None, None, None
        qids = df.qid.unique()
        np.random.shuffle(qids)
        for qid in qids:
            x_i, y_i, x_j, y_j = self.generate_query_pairs(df, qid)
            if x_i_buf is None:
                x_i_buf, y_i_buf, x_j_buf, y_j_buf = x_i, y_i, x_j, y_j
            else:
                x_i_buf = np.vstack((x_i_buf, x_i))
                y_i_buf = np.vstack((y_i_buf, y_i))
                x_j_buf = np.vstack((x_j_buf, x_j))
                y_j_buf = np.vstack((y_j_buf, y_j))
            idx = 0
            while (idx + 1) * batchsize <= x_i_buf.shape[0]:
                start = idx * batchsize
                end = (idx + 1) * batchsize
                yield x_i_buf[start: end, :], y_i_buf[start: end, :], x_j_buf[start: end, :], y_j_buf[start: end, :]
                idx += 1

            x_i_buf = x_i_buf[idx * batchsize:, :]
            y_i_buf = y_i_buf[idx * batchsize:, :]
            x_j_buf = x_j_buf[idx * batchsize:, :]
            y_j_buf = y_j_buf[idx * batchsize:, :]

        yield x_i_buf, y_i_buf, x_j_buf, y_j_buf

    def generate_query_batch(self, df, batchsize=100000):
        """
        :param df: pandas.DataFrame, contains column qid
        :returns: numpy.ndarray qid, rel, x_i
        """
        idx = 0
        while idx * batchsize < df.shape[0]:
            r = df.iloc[idx * batchsize: (idx + 1) * batchsize, :]
            yield r['qid'].values, r['head'].values, r['relation'].values, r['tail'].values, r['rel'].values
            idx += 1

    def generate_batch_per_query(self, df=None):
        """
        :param df: pandas.DataFrame
        :return: X for features, y for relavance
        :rtype: numpy.ndarray, numpy.ndarray
        """
        if df is None:
            df = self.df
        # unique qid
        qids = df['qid'].unique()
        # shuffle qids
        np.random.shuffle(qids)
        for qid in qids:
            df_qid = df[df.qid == qid]
            # 每个qid下对应的所有样本 ['rel', 'qid', 'head', 'relation', 'tail']
            yield df_qid['head'].values, df_qid['relation'].values, df_qid['tail'].values, df_qid['rel'].values

    @classmethod
    def truncate_seq_triple(cls, tokens_a, tokens_b, tokens_c, max_length):
        """Truncates a sequence triple in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
                tokens_a.pop()
            elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
                tokens_b.pop()
            elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
                tokens_c.pop()
            else:
                tokens_c.pop()

    @classmethod
    def convert_examples_to_features(cls, head, rel, tail, label, tokenizer, max_length):
        head_tokens = tokenizer.tokenize(head)
        relation_tokens = tokenizer.tokenize(rel)
        tail_tokens = tokenizer.tokenize(tail)

        DataLoader.truncate_seq_triple(head_tokens, relation_tokens, tail_tokens, max_length - 10)

        head_mask = [0] # [CLS]
        rel_mask = [0] # [CLS]
        tail_mask = [0] # [CLS]

        tokens = ["[CLS]"] + ["[E1]"] + head_tokens + ["[/E1]"] + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        # head mask
        head_mask.append(1)
        head_mask.extend([0] * (len(head_tokens) + 2))
        # rel mask
        rel_mask.extend([0] * (len(head_tokens) + 3))
        # tail mask
        tail_mask.extend([0] * (len(head_tokens) + 3))

        tokens += ["[RE]"] + relation_tokens + ["[/RE]"] + ["[SEP]"]
        segment_ids += [1] * (len(relation_tokens) + 3)
        # head mask
        head_mask.extend([0] * (len(relation_tokens) + 3))
        # rel mask
        rel_mask.append(1)
        rel_mask.extend([0] * (len(relation_tokens) + 2))
        # tail mask
        tail_mask.extend([0] * (len(relation_tokens) + 3))

        tokens += ["[E2]"] + tail_tokens + ["[/E2]"] + ["[SEP]"]
        segment_ids += [0] * (len(tail_tokens) + 3)
        # head mask
        head_mask.extend([0] * (len(tail_tokens) + 3))
        # rel mask
        rel_mask.extend([0] * (len(tail_tokens) + 3))
        # tail mask
        tail_mask.append(1)
        tail_mask.extend([0] * (len(tail_tokens) + 2))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        head_mask += padding
        rel_mask += padding
        tail_mask += padding

        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(segment_ids) == max_length
        assert len(head_mask) == max_length
        assert len(rel_mask) == max_length
        assert len(tail_mask) == max_length

        return {'label': label,
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'head_mask' : head_mask,
                'rel_mask' : rel_mask,
                'tail_mask' : tail_mask}

    def load(self):
        """
        :return: pandas.DataFrame
        """
        # -> ['rel', 'qid', 'head', 'relation', 'tail']
        self.df = self._parse_feature_and_label(self._load_mslr())
        return self.df
