import random

def process(train_path, train_result_path, test_path, test_result_path, dev_path, dev_result_path):
    '''
    (1).train data will be processed to in the followling format:
        0 1 head rel1 tail
        1 1 head rel1 tail
        0 1 head rel1 tail
        1 2 head rel2 tail
        0 2 head rel2 tail
        ...

    (2).test and dev data will be processed to in the following format:
        1 1 head rel1 tail
        1 1 head rel1 tail
        1 2 head rel2 tail
        1 2 head rel2 tail
        ...

    :param train_path:
    :param train_result_path:
    :param test_path:
    :param test_result_path:
    :param dev_path:
    :param dev_result_path:
    :return:
    '''
    entity2text_dict = dict()
    with open('./entity2text.txt', 'r', encoding='utf-8') as entityRead:
        for line in entityRead:
            line = line.strip().split('\t')
            entity1, entity2 = tuple(line)
            entity2text_dict[entity1] = entity2

    relation2text_dict = dict()
    with open('./relation2text.txt', 'r', encoding='utf-8') as relationRead:
        for line in relationRead:
            line = line.strip().split('\t')
            relation1, relation2 = tuple(line)
            relation2text_dict[relation1] = relation2

    entity_set = set()
    data_list = list()
    with open(train_path, 'r', encoding='utf-8') as fRead:
        for line in fRead:
            line = line.strip().split('\t')
            head, rel, tail = tuple(line)
            entity_set.add(head)
            entity_set.add(tail)
            data_list.append([head, rel, tail])
        entities = list(entity_set)
        print('entity count: {}'.format(len(entity_set)))

    rel_dict = dict()
    with open(train_path, 'r', encoding='utf-8') as fRead, open(train_result_path, 'w', encoding='utf-8') as fWrite:
        count = 0
        line_count = 0
        for line in fRead:
            print('line count: {}'.format(line_count))
            line_count += 1
            line = line.strip().split("\t")
            head = line[0]
            rel = line[1]
            tail = line[2]

            if rel not in rel_dict:
                count += 1
                rel_dict[rel] = count

            line_write = '1' + ',' + str(rel_dict[rel]) + ',' + entity2text_dict[head] + ',' + relation2text_dict[rel] + ',' + entity2text_dict[tail]
            fWrite.write(line_write)
            fWrite.write('\n')

            # construct negative sample
            rnd = random.random()
            if rnd <= 0.5:
                while True:
                    tmp_ent_list = set(entities)
                    tmp_ent_list.remove(head)  # remove head
                    tmp_ent_list = list(tmp_ent_list)
                    tmp_head = random.choice(tmp_ent_list)  # get a random head entity
                    tmp_triple_str = [tmp_head, rel, tail]
                    if tmp_triple_str not in data_list:
                        break
                line_write = '0' + ',' + str(rel_dict[rel]) + ',' + entity2text_dict[tmp_head] + ',' + relation2text_dict[rel] + ',' + entity2text_dict[tail]
                fWrite.write(line_write)
                fWrite.write('\n')
            else:
                while True:
                    tmp_ent_list = set(entities)
                    tmp_ent_list.remove(line[2])  # remove tail
                    tmp_ent_list = list(tmp_ent_list)
                    tmp_tail = random.choice(tmp_ent_list)  # get a random tail entity
                    tmp_triple_str = [line[0], line[1], tmp_tail, 0]
                    if tmp_triple_str not in data_list:
                        break
                line_write = '0' + ',' + str(rel_dict[rel]) + ',' + entity2text_dict[head] + ',' + relation2text_dict[rel] + ',' + entity2text_dict[tmp_tail]
                fWrite.write(line_write)
                fWrite.write('\n')

    # test data
    with open(test_path, 'r', encoding='utf-8') as fRead, open(test_result_path, 'w', encoding='utf-8') as fWrite:
        for line in fRead:
            line = line.strip().split("\t")
            head = line[0]
            rel = line[1]
            tail = line[2]
            line_write = '1' + ',' + str(rel_dict[rel]) + ',' + entity2text_dict[head] + ',' + relation2text_dict[rel] + ',' + entity2text_dict[tail]
            fWrite.write(line_write)
            fWrite.write('\n')

    # dev data
    with open(dev_path, 'r', encoding='utf-8') as fRead, open(dev_result_path, 'w', encoding='utf-8') as fWrite:
        for line in fRead:
            line = line.strip().split("\t")
            head = line[0]
            rel = line[1]
            tail = line[2]
            line_write = '1' + ',' + str(rel_dict[rel]) + ',' + entity2text_dict[head] + ',' + relation2text_dict[rel] + ',' + entity2text_dict[tail]
            fWrite.write(line_write)
            fWrite.write('\n')

    print('process done!')

if __name__ == '__main__':
    train_path = './train.tsv'
    train_result_path = './ptrain.csv'

    test_path = './test.tsv'
    test_result_path = './ptest.csv'

    dev_path = './dev.tsv'
    dev_result_path = './pdev.csv'

    process(train_path, train_result_path, test_path, test_result_path, dev_path, dev_result_path)


