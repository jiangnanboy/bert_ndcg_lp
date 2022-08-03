import os

def process_data(read_path, save_path):
    result_list = dict() # "a":[[], []]
    with open(read_path, 'r', encoding='utf-8') as f_read, open(save_path, 'w', encoding='utf-8') as f_write:
        for line in f_read:
            line = line.strip().split(',')
            head = line[0].strip()
            rel = line[1].strip()
            tail = line[2].strip()
            if rel in result_list:
                result_list[rel][0].append(head)
                result_list[rel][1].append(tail)
            else:
                head_list = []
                tail_list = []
                head_list.append(head)
                tail_list.append(tail)
                result_list[rel] = [head_list, tail_list]

        for key,value in result_list.items():
            for head, tail in zip(value[0], value[1]):
                # print('({}) -[{}]-> ({})'.format(head, key, tail))
                f_write.write(head + ',' + key + ',' + tail)
                f_write.write('\n')
            f_write.write('\n')

if __name__ == '__main__':
    train_read_path = './train.sample.csv'
    test_read_path = './test.sample.csv'
    test_save_path = './test.csv'
    train_save_path = './train.csv'
    process_data(train_read_path, train_save_path)
