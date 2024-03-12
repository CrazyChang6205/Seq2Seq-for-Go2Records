目錄：
1. 文檔編排

以下為專案的文檔編排：
/評論生成
|-- /asset	(文件存放區)
|	|-- /Go_DB_original	(原始資料集，不會做更動，備份用)
|	|-- /Go_DB			(初始資料集，已做處理[請見異常紀錄1])
|	|-- /GO_R2C_DB		(輸入資料集，兩行為一組，內容為"棋步及對映的評論")
|	|-- /Go_DB_for_test	(測試用)
|	|-- custom_dict.txt	(中文斷詞用，自定義辭典)
|	|-- stop_words.txt	(中文斷詞用，停用字)
|	`-- (其他)
|
|-- /src	(程式存放區)
|	|-- /config			(設定檔)
|	|	`-- config.py
|	|-- /dataset		(資料檔)
|	|	`-- dataset.py
|	|-- /model			(模組區)
|	|	|-- Creat_dataset.py (自定義資料集)
|	|	|-- Encoder.py	(編碼器)
|	|	|-- Decoder.py	(解碼器)
|	|	`-- Seq2Seq.py	(序列轉換模型)
|	|-- /temp			(暫存用)
|	|	|-- XXX.dict
|	|	`-- XXX.model
|	|-- sgf_transform_R2C.py	(將 初始資料 轉換成 輸入資料)
|	|-- creat_word2vec.py		(將 輸入資料R2C 轉換成 Word2Vec並儲存成model )
|	|-- train.py				(訓練模型)
|	|-- test.py					(測試用)
|	`-- (其他)
|
|-- /README
|	|-- 開發日誌.log
|	|-- 異常紀錄.txt
|	|-- 研究日誌.txt
|	`-- 棋譜檔案集整理文件.xlsx
`-- /參考文獻

以下為目前專案的命名原則：
部分縮寫意思:
Go	-> 圍棋
DB	-> 資料庫
sgf	-> Smart Game Format File
R -> Records 紀錄
C -> Commend 註解
R2C	-> Records to Commend
W2V	-> Word2Vec