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
            #Comment_line = "<SOS> " + Comment_line + " <EOS>"
            #Records_line = "<SOS> " + Records_line + " <EOS>"
            Comment_doc.append(Comment_line)
            Records_doc.append(Records_line)
    #NOTE：載入文件，creat doc
            
except FileNotFoundError:
    print(f"file '{file_path}' not found!")
except Exception as e:
    print(f"error {e}")

else:
    """
    for comment in Comment_doc:
        corpus = []
        result = jieba.tokenize(comment)
        for tk in result:
            corpus.append(tk[0])
        Comment_corpus.append(corpus)

    for records in Records_doc:
        corpus = []
        result = jieba.tokenize(records)
        for tk in result:
            corpus.append(tk[0])
        Records_corpus.append(corpus)

    participle_comment = []
    for comment in Comment_corpus:
        seq = ""
        for corpus in comment:
            seq += corpus
            seq += ' '
        participle_comment.append(seq)
    """
    tokenized_comment = [ ["<SOS>"] + list(jieba.cut(sentence)) + ["<EOS>"] for sentence in Comment_doc]
    tokenized_records = [ ["<SOS>"] + list(jieba.cut(sentence)) + ["<EOS>"] for sentence in Records_doc]
    
    predefined_words = ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]
    
    comment_model_w2v =  Word2Vec(vector_size=100, window=5, min_count=1, workers=4, max_final_vocab=500)
    comment_model_w2v.build_vocab([predefined_words])
    comment_model_w2v.build_vocab(tokenized_comment, update=True)
    comment_model_w2v.train(tokenized_comment, total_examples=len(tokenized_comment), epochs=10)
    comment_model_w2v.save('./temp/W2V_'+ file_name[0:-3] +'C_NEW.model')
    
    records_model_w2v = Word2Vec(vector_size=100, window=5, min_count=1, workers=4, max_final_vocab=500)
    records_model_w2v.build_vocab([predefined_words])
    records_model_w2v.build_vocab(tokenized_records, update=True)
    records_model_w2v.train(tokenized_records, total_examples=len(tokenized_records), epochs=10)
    records_model_w2v.save('./temp/W2V_'+ file_name[0:-3] +'R_NEW.model')
    
print("end")