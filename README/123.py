class Encoder(nn.Module):
    def __init__(self, word2vec_model, hidden_size, num_layers=1):
        super(Encoder, self).__init__()        
        vocab_size, embed_dim = word2vec_model.wv.vectors.shape
        self.word2vec_model = word2vec_model
        self.embedding_size = word2vec_model.vector_size
		self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word2vec_model.wv.vectors))
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers)
        
    def forward(self, input_seq, hidden=None):
        input_seq_size = input_seq.size()
        embedding_hidden = input_seq_size[1]
        hidden_size = hidden[0].size()        
        print("Encoder forward")
        print("input_seq  ：", input_seq.size())
        print("hidden_size：", hidden_size)
        print("embedding_hidden：", embedding_hidden)        
        print(hidden)
        embedded = self.embedding(input_seq)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.lstm(embedded, hidden)
        output = self.out(output)
        return output, hidden

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_w2v_model = Word2Vec.load('./temp/W2V_Go_All_C_NEW.model')
        self.input_w2v_model  =  Word2Vec.load('./temp/W2V_Go_All_R_NEW.model')

    def forward(self, input_seq, target_seq, initial_hidden=None):
        if initial_hidden is None:
            batch_size = input_seq.size(0)
            initial_hidden = self.encoder.init_hidden(batch_size)
        print("Seq2Seq forward")
        print("input_seq  ：", input_seq.size())
        hidden_size = initial_hidden[0].size()
        print("hidden_size：", hidden_size)
        print("initial_hidden：",len(initial_hidden),len(initial_hidden[0]),len(initial_hidden[0][0]),len(initial_hidden[0][0][0]))
        print(initial_hidden)
        print()
        encoder_output, encoder_hidden = self.encoder(input_seq, initial_hidden)
        decoder_hidden = encoder_hidden
        decoder_output, _ = self.decoder(target_seq, decoder_hidden)
        return decoder_output

class Custom_Dataset(Dataset):
    def __init__(self, load_dataset_name, input_w2v_name, output_w2v_name):
        self.dataset = self.load_dataset(load_dataset_name)
        self.comments = [comment for records, comment in self.dataset]
        self.records  = [records for records, comment in self.dataset]
        self.n_samples = len(self.dataset)
        self.input_w2v_model = Word2Vec.load(output_w2v_name)
        self.input_word_to_index = {word: idx for idx, word in enumerate(self.w2v_model.wv.index_to_key)}
		self.output_w2v_model = Word2Vec.load(input_w2v_name )
		self.output_word_to_index = {word: idx for idx, word in enumerate(self.w2v_model.wv.index_to_key)}
        
    def load_dataset(self, file_name):
		"""
		load file and make dataset
		dataset sformat -> [[ record, comment ],...]
		"""
        return dataset

    def text_to_tensor_byC(self, text):
        return torch.tensor([self.output_word_to_index.get(word, self.output_word_to_index.get('<UNK>')) for word in text.split()])

    def text_to_tensor_byR(self, text):
        return torch.tensor([self.input_word_to_index.get(word, self.input_word_to_index.get('<UNK>')) for word in text.split()])

    def __getitem__(self, index):
        input_tensor_R = self.text_to_tensor_byR(self.records[index])
        input_tensor_C = self.text_to_tensor_byC(self.comments[index])
        return input_tensor_R, input_tensor_C

    def __len__(self):
        return self.n_samples
		
class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
		
    def train_step(self, input_seqs, target_seqs):
        self.optimizer.zero_grad()
        total_loss = 0
        batch_size = input_seqs.size(0)
        for i in range(batch_size):
            input_seq = input_seqs[i]
            target_seq = target_seqs[i]
            original_input_seq, original_outpur_seq = seq_Translator(input_seq, target_seq)
            outputs = self.model(input_seq.unsqueeze(0), target_seq.unsqueeze(0))
            loss = self.criterion(outputs.squeeze(0), target_seq)
            total_loss += loss.item()
            loss.backward()
        self.optimizer.step()
        return total_loss / batch_size
    
    def train_epoch(self, dataloader):
        total_loss = 0
        num_batches = len(dataloader)
        for i, (input_seqs, target_seqs) in enumerate(dataloader):
            loss = self.train_step(input_seqs, target_seqs)
            total_loss += loss
            break
        average_loss = total_loss/num_batches
        return average_loss

# 初始化數據集和 DataLoader
custom_dataset = Custom_Dataset()
batch_size = 32

# 新增<PAD>處理
from torch.nn.utils.rnn import pad_sequence
def seqs_padded(batch):
    input_seqs, target_seqs = zip(*batch)
    input_seqs_padded  = pad_sequence(input_seqs , batch_first=True, padding_value=1)
    target_seqs_padded = pad_sequence(target_seqs, batch_first=True, padding_value=1)
    return input_seqs_padded, target_seqs_padded
train_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=seqs_padded)

# 初始化數據集和 DataLoader
custom_dataset = Custom_Dataset()
batch_size = 3

# 新增<PAD>處理
from torch.nn.utils.rnn import pad_sequence
def seqs_padded(batch):
    input_seqs, target_seqs = zip(*batch)
    input_seqs_padded  = pad_sequence(input_seqs , batch_first=True, padding_value=1)
    target_seqs_padded = pad_sequence(target_seqs, batch_first=True, padding_value=1)
    return input_seqs_padded, target_seqs_padded
train_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=seqs_padded, drop_last=True)

# 初始化模型参数
output_w2v_model = Word2Vec.load('./temp/W2V_Go_All_C_NEW.model')
input_w2v_model =  Word2Vec.load('./temp/W2V_Go_All_R_NEW.model')
hidden_size = 256
output_size = 256
num_layers = 2
learning_rate = 0.01
num_epochs = 2

# 模型 優化器 損失函數 Trainer
encoder = Encoder(output_w2v_model, hidden_size, num_layers)
decoder = Decoder(hidden_size, output_size)
seq2seq_model = Seq2Seq(encoder, decoder)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(seq2seq_model.parameters(), lr=learning_rate)
trainer = Trainer(seq2seq_model, criterion, optimizer)

# 訓練迴圈
for epoch in range(num_epochs):
    loss = trainer.train_epoch(train_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss}\n')