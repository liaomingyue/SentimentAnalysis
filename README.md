# Sentiment Analysis


## 目录说明

```
D:\PYCHARMPROJECTS\SENTIMENTANALYSIS
│  README.md
│  train.py              程序入口*
│
├─dataset               数据集 数据集比例（6:2:2）
│  │  eval.data         验证集
│  │  test.data         测试集
│  │  train.data        训练集
│  │
│  └─raw_data           原始数据、生成的中间数据
│      │  add.data
│      │  add2.data
│      │  ChineseStopWords.txt
│      │  mood_data.csv
│      │  微博情绪标注语料.xml
│      │  微博情绪样例数据V5-13.xml
│      │
│      └─mood
│              恐惧.txt
│              感谢.txt
│              痛苦.txt
│              表扬.txt
│
├─model     模型相关
│  │  chat_dataset_reader.py    数据读取
│  └  sentiment_classifier.py   模型
│
├─outputs   模型输出（模型参数和出现错误的数据）
│  │  2019-04-23-17-36-39_model_bs32_ep15_hs256_emb100_ad64_ah16_cs1024_dp0.5.tar.gz
│  │
│  └─wrong  出现错误数据的记录（包括：原始句子、标注类别、预测类别）
│          2019-04-23-17-36-39_wrong.json
│
└─script    处理数据的一些脚本
        make_data.ipynb
        make_data2.ipynb
        make_dataset.ipynb
        make_dataset2.ipynb
        make_dataset3.ipynb
```

## 后期任务

1. 原始数据集标注不准确，有提升的空间
2. 训练数据处理：目前没有对数据去除停用词等，之后可以添加
3. 目前模型没有进行调参，以及效率测试等，模型还有提升的空间


