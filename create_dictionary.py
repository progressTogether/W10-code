from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json

# Read the data into a list of strings.
def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    #count是list，第一个元素是['UNK', -1]，之后按照在单词在words中出现的频数的高低以此进入list
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
        #生成字典其中'UNK': 0，key为对应的单词，value为出现次数的排名，即如果love出现的次数最高
        #那么则love的value为1.
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    #这段代码的大致意思是将words的单词根据dictionary的key和获取对应的value
    #如果该word是dictionary的key那么则获取对应的value
    #如果没有则其值是0
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    #将key和value互换生成新的reversed_dictionary
    return data, count, dictionary, reversed_dictionary


filename = './QuanSongCi.txt'

vocabulary = read_data(filename)
print('Data size', len(vocabulary))

vocabulary_size = 5000

data, count, dictionary, reversed_dictionary = build_dataset(vocabulary,vocabulary_size - 1)

#json.dump()函数的使用，将json文件保存
with open("./dictionary.json","w",encoding='utf-8') as f:
    json.dump(dictionary,f)

with open("./reverse_dictionary.json","w",encoding='utf-8') as f:
    json.dump(reversed_dictionary,f)

