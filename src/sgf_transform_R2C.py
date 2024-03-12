import os

current_path = os.getcwd()
print("當前工作檔案路徑:", current_path)
target_path = 'D:/user/桌面/Meeting/圍棋術語分類及評論生成/src/'
os.chdir(target_path)
new_path = os.getcwd()
print("切換工作檔案路徑:", new_path)

from sgfmill import sgf
#DONE：挑選出發生意外錯誤的檔案，並找出原因
#DONE：棋譜編號
#DONE：分支狀況處理
#TODO：生成兩種註解測資 1.完整棋譜 2.無變化圖

import jieba
from jieba import analyse

jieba.analyse.set_stop_words('../asset/stop_words.txt')
jieba.load_userdict('../asset/custom_dict.txt')
#NOTE：載入自定義字典
#TODO：停用字 處理
#TODO：低頻字 處理

def creat_R2Cfile(file_name):
    print(file_name, end=' ')
    file_path = "../asset/Go_DB/"+ file_name +".sgf"
    with open(file_path, mode='r', encoding='UTF-8') as sgf_file:
        sgf_doc = sgf_file.read()
        sgf_doc = sgf_doc.replace('\n', ' ')
        """
        print(bool(re.search(r'[^A-Z|\]|\s]\[.*\]', sgf_doc)), re.search(r'[^A-Z|\]|\s]\[.*\]', sgf_doc))
        if bool(re.search(r'[^A-Z|\]|\s]\[.*\]', sgf_doc)):
            sgf_doc = re.sub(r'[^A-Z|\]|\s]\[.*\]', '{}', sgf_doc)
            print(bool(re.search(r'[^A-Z|\]|\s]\[.*\]', sgf_doc)),re.search(r'[^A-Z|\]|\s]\[.*\]', sgf_doc))
        """
        try:
            sgf_game = sgf.Sgf_game.from_string(sgf_doc)
        except ValueError as e:
            Exception(type(sgf_doc.encode()), e)
            sgf_game = sgf.Sgf_game.from_string(sgf_doc)
        finally:
            print("sgf_game creat!")
            root_node = sgf_game.get_root()
            
            def print_move(node):
                print(node.get_move(),end="\t")
                return node.get_move()
            def print_comments(node):
                if node.has_property("C"):
                    print_move(node)
                    print(node.get("C"))
                else:
                    print_move(node)
                    print("NA")
                for child in node:
                    print_comments(child)
            
            Records2Comment = []
            records = ""
            comment = ""
            def file_transform_Records2Comment(node ,Records2Comment ,records):
                node_move = node.get_move()
                if node_move[1] != None :
                    move = str(node_move[0].upper()) + str(chr(node_move[1][0] + 97)) + str(chr(node_move[1][1] + 97)) + " "
                    records = records + move
                if node.has_property("C"):
                    comment = node.get("C")
                    tokenized_comment = list(jieba.cut(comment))
                    participle_comment = ""
                    for word in tokenized_comment:
                        participle_comment += word
                        participle_comment += " "
                    Records2Comment.append([participle_comment,records])
                for child in node:
                    file_transform_Records2Comment(child ,Records2Comment ,records)
    
            file_transform_Records2Comment(root_node ,Records2Comment, records)
            file_path = '../asset/GO_DB_R2C/R2C_'+ file_name +'.txt'
            with open(file_path, mode='w', encoding='UTF-8') as output_file:
                for comment, records in Records2Comment:
                    output_file.write(comment+'\n')
                    output_file.write(records+'\n')
            
            file_path = '../asset/'+ 'Go_All_R2C' +'.txt'
            with open(file_path, mode='a', encoding='UTF-8') as output_file:
                for comment, records in Records2Comment:
                    output_file.write(comment+'\n')
                    output_file.write(records+'\n')
            
            file_path = '../asset/'+ 'Go_All_C' +'.txt'
            with open(file_path, mode='a', encoding='UTF-8') as output_file:
                for comment, records in Records2Comment:
                    output_file.write(comment+'\n')
                 
            file_path = '../asset/'+ 'Go_All_R' +'.txt'
            with open(file_path, mode='a', encoding='UTF-8') as output_file:
                for comment, records in Records2Comment:
                    output_file.write(records+'\n')

def Run_code_with_Traverse_files_in_folders(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_name = os.path.splitext(os.path.join(file))[0]
            creat_R2Cfile(file_name)
#NOTE：遍歷指定資料夾下的所有檔案
#NOTE：os.walk()：是一個用在檔案目錄樹中遍歷的方法，有三個回傳(root,dirs,files)
#NOTE：os.path.join()：是一個路徑合併的方法

file_name = "AI001"
file_dirs = "../asset/Go_DB/"
file_path = file_dirs + file_name +".sgf"

Run_code_with_Traverse_files_in_folders(file_dirs)
#creat_R2Cfile("AI001")