from gensim.models import Word2Vec

def seq_original(input_seq, target_seq):
    output_w2v_model = Word2Vec.load('./temp/W2V_Go_All_C_NEW.model')
    input_w2v_model =  Word2Vec.load('./temp/W2V_Go_All_R_NEW.model')
    original_input_seq  = ""
    original_output_seq = ""
    print("seq_Translator.")
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