import torch
from torch.utils.data import Dataset

class Custom_Dataset(Dataset):

    def __init__(self, dataset, input_w2v_model, output_w2v_model):
        if type(dataset) is list:
            self.dataset = dataset
        elif type(dataset) is str:
            self.dataset = self.load_dataset(dataset)
        self.input_w2v_model  = input_w2v_model
        self.output_w2v_model = output_w2v_model
        
        self.n_samples = len(self.dataset)
        self.comments = [comment for record, comment in self.dataset]
        self.records  = [record  for record, comment in self.dataset]
        self.in_word_to_index  = {word: idx for idx, word in enumerate(self.input_w2v_model.wv.index_to_key) }
        self.out_word_to_index = {word: idx for idx, word in enumerate(self.output_w2v_model.wv.index_to_key)}
        
    def load_dataset(self, file_name):
        file_dirs = "../asset/"
        file_path = file_dirs + file_name + ".txt"
        dataset = []
        try:
            with open(file_path, mode='r', encoding='UTF-8') as R2C_file:
                SOS = "<SOS> "
                EOS = " <EOS>"
                lines = R2C_file.readlines()
                for i in range(0, len(lines), 2):
                    comment_line = lines[i].strip()
                    records_line = lines[i+1].strip() if i+1 < len(lines) else None
                    dataset.append([ SOS + records_line + EOS, SOS + comment_line + EOS ])
                    if records_line is None:
                        print(f'Comment: {comment_line}\nNo Records\n')
        except FileNotFoundError:
            print(f"File '{file_path}' not found!")
        except Exception as e:
            print(f"Error: {e}")
        return dataset

    def text_to_tensor_byC(self, text):
        return torch.tensor([self.out_word_to_index.get(word, self.out_word_to_index.get('<UNK>')) for word in text.split()])

    def text_to_tensor_byR(self, text):
        return torch.tensor([self.in_word_to_index.get(word, self.in_word_to_index.get('<UNK>')) for word in text.split()])

    def __getitem__(self, index):
        input_tensor_R = self.text_to_tensor_byR(self.records[index])
        input_tensor_C = self.text_to_tensor_byC(self.comments[index])
        return input_tensor_R, input_tensor_C

    def __len__(self):
        return self.n_samples

