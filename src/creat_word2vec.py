import os

current_path = os.getcwd()
print("當前工作檔案路徑:", current_path)
target_path = 'D:/user/桌面/Meeting/圍棋術語分類及評論生成/src/'
os.chdir(target_path)
new_path = os.getcwd()
print("切換工作檔案路徑:", new_path)

import numpy as np

from gensim import corpora
from gensim import models
from gensim.models import Word2Vec

#from opencc import OpenCC
#cc = OpenCC('s2tw')
#NOTE：t2s 繁體中文->簡體中文；s2tw 簡體中文->繁體中文(台灣)；s2twp 簡體中文->繁體中文(台灣,包含慣用詞轉換)

import jieba
from jieba import analyse
#from ckiptagger import WS, POS, NER

#jieba.set_dictionary()
jieba.analyse.set_stop_words('../asset/stop_words.txt')
jieba.load_userdict('../asset/custom_dict.txt')
"""
jieba.add_word('<SOS>')
jieba.add_word('<EOS>')
jieba.add_word('<PAD>')
jieba.add_word('<UNK>')
"""
#NOTE：載入自定義字典
#TODO：停用字 處理
#TODO：低頻字 處理

file_name = "Go_All_R2C"
file_dirs = "../asset/"
file_path = file_dirs + file_name +".txt"

Comment_doc = []
Records_doc = []
#NOTE：corpus 語料庫
#TODO：以 yield 儲存生成器
try:
    with open(file_path, mode='r', encoding='UTF-8') as R2C_file:
        lines = R2C_file.readlines()
        Comment_line = ""
        Records_line = ""
        for i in range(0, len(lines), 2):
            Comment_line = lines[i].strip()
            if i+1 < len(lines):
                Records_line = lines[i+1].strip()
            else:
                print(f'Comment: {Comment_line}\nNo Records\n')
            Comment_doc.append(Comment_line)
            Records_doc.append(Records_line)
            
except FileNotFoundError:
    print(f"file '{file_path}' not found!")
except Exception as e:
    print(f"error {e}")

else:
    tokenized_comment = [ ["<SOS>"] + list(jieba.cut(sentence)) + ["<EOS>"] for sentence in Comment_doc]
    tokenized_records = [ ["<SOS>"] + list(jieba.cut(sentence)) + ["<EOS>"] for sentence in Records_doc]
    
    # Wrod2Vec 模型參數
    predefined_words = ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]
    vector_size = 100   # 詞彙向量大小(維度)
    window = 5          # 詞彙視窗大小，詞向量上下文最大距離，與上下文關係有關，預設為5，一般推薦[5,10]
    min_count = 5       # 低頻詞彙處理，詞彙計數小於該值則丟棄，預設值為5
    sample = 1e-3       # 高頻詞彙處理，高頻詞彙的隨機降採樣的配置閾值，預設為1e-3，範圍是（0,1e-5）
    max_final_vocab = 500 # 詞彙量上限，設值為None則無上限
    seed = 1            # 初始詞彙向量的隨機種子，預設為1
    workers = 4         # 訓練時的並行數
    sg = 0              # 訓練用演算法，預設為0，{ 0:skip-gram, 1:CBOW }
    hs = 0              # 加速訓練方法，預設為0，{ 0:negative sampling, 1:hierarchica softmax }
    negative = 5        # 負採樣的個數，用於採用多少個 noise words，預設為5?，一般推薦[5,20]
    iter = 5            # 梯度下降反覆訓練次數，預設為5
    alpha = 0.025       # 學習效率
    min_alpha = 0.0001  # 學習效率最小值
    
    # Word2Vec 訓練資訊
    epochs = 10
    version = "NEW"
    
    # Word2Vec train!
    comment_model_w2v =  Word2Vec(vector_size=500, window=5, min_count=1, workers=8, max_final_vocab=None)
    comment_model_w2v.build_vocab([predefined_words])
    comment_model_w2v.build_vocab(tokenized_comment, update=True)
    comment_model_w2v.train(tokenized_comment, total_examples=len(tokenized_comment), epochs=10)
    comment_model_w2v.save('./temp/word2vec/comment/w2v_byC_version_'+ version +'.model')
    
    records_model_w2v = Word2Vec(vector_size=500, window=5, min_count=1, workers=8, max_final_vocab=None)
    records_model_w2v.build_vocab([predefined_words])
    records_model_w2v.build_vocab(tokenized_records, update=True)
    records_model_w2v.train(tokenized_records, total_examples=len(tokenized_records), epochs=10)
    records_model_w2v.save('./temp/word2vec/records/w2v_byR_version_'+ version +'.model')
    
    test_w2v = Word2Vec(vector_size=5)
    
print("end")