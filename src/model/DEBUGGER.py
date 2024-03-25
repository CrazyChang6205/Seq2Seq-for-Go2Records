import re
import time
from tqdm import tqdm

def extract_substrings(text, pattern):
    matches = re.findall(pattern, text)
    return matches

def seq2seq_original(input_seq, target_seq, input_w2v_model, output_w2v_model):
    original_input_seq  = ""
    original_output_seq = ""
    print("seq2seq_Translator.")
    print(f"len(input_seq) = {len(input_seq)},\t len(target_seq) = {len(target_seq)}")
    #print("input_seq：\n", input_seq)
    #print("target_seq：\n", target_seq, '\n')
    for key_index in input_seq:
        original_input_seq += input_w2v_model.wv.index_to_key[key_index]
        original_input_seq += " "
    for key_index in target_seq:
        original_output_seq += output_w2v_model.wv.index_to_key[key_index]
        original_output_seq += " "
    print("original_input_seq ：\n", original_input_seq)
    print("original_output_seq：\n", original_output_seq,"\n")
    return original_input_seq, original_output_seq

def seq_original(input_seq, word2vec):
    original_seq  = ""
    for key_index in input_seq:
        original_seq += word2vec.wv.index_to_key[key_index]
        original_seq += " "
    return original_seq