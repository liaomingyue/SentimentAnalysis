import json
import os

from allennlp.common.file_utils import cached_path
from allennlp.data import Vocabulary
from allennlp.common.util import ensure_list
# from gensim.models import word2vec
from gensim.models import word2vec, KeyedVectors

from model.chat_dataset_reader import ChatDatasetReader




# import pkuseg
#
#
# ""
# seg = pkuseg.pkuseg(postag=True)  # 开启词性标注功能
# text = seg.cut('像“金灵根”和“水灵根”异变产生的“雷灵根”；“土灵根”和“水灵根”异变后产生的“冰灵根”，当然还有“暗灵根”、“风灵根”等其他变异灵根。')    # 进行分词和词性标注
# print(text)

# import pkuseg
#
# def dataset_to_cut(filename_read='dataset/train.data',filename_writer='dataset_copy/train.data'):
#     with open(filename_read, mode='r', encoding='utf-8') as f, open(filename_writer, mode='w',encoding='utf-8') as w_f:
#         for dict_info in f:
#             dict_info = eval(dict_info)
#         seg = pkuseg.pkuseg()
#         all_info_list = []
#         count = 0
#         for line in dict_info:
#             line['sentence'] = line["sentence"].replace(" ", "")
#             text = seg.cut(line["sentence"])
#             line["sentence"] = " ".join(text)
#             # line=json.dumps(line)
#             all_info_list.append({"sentence":line["sentence"],"label":line["label"]})
#             count += 1
#             # w_f.writelines(str({"sentence":line["sentence"],"label":line["label"]}))
#         w_f.write(json.dumps(all_info_list, ensure_ascii=False))
#         print(count)
#         return all_info_list
# all_info_list=[]
# all_info_list1=dataset_to_cut()
# all_info_list2=dataset_to_cut('dataset/test.data','dataset_copy/test.data')
# all_info_list3=dataset_to_cut('dataset/eval.data','dataset_copy/eval.data')
# all_info_list=all_info_list+all_info_list1
# all_info_list=all_info_list+all_info_list2
# all_info_list=all_info_list+all_info_list3
#
# train_path = os.path.join('dataset_copy', 'train.data')
# eval_path = os.path.join('dataset_copy', 'eval.data')
# # test_path = os.path.join('dataset', 'test.data')
#
# reader = ChatDatasetReader()
# train_dataset = ensure_list(reader.read(train_path))
# eval_dataset = ensure_list(reader.read(eval_path))
#
# vocab = Vocabulary.from_instances(train_dataset + eval_dataset,
#                                   min_count={'sentence': 3})
# print(vocab.save_to_files('lmy'))

# model2 = word2vec.Word2Vec.load('word2vecpath/news12g_bdbk20g_nov90g_dim128.model')
model_3 =KeyedVectors.load_word2vec_format("word2vecpath/news12g_bdbk20g_nov90g_dim128.bin",binary=True)

sim3 = model_3.most_similar(u'萧炎', topn=20)

print(u'和 新华社 与相关的词有：\n')
for key in sim3:
    print(key[0], key[1])