'''
Class BERT_DATALOADER:
    - Input:
        + dir: đường dẫn của file .pkl
        + tokenizer : .....
        + tag_values : ['PAD',...]
        + type_data: 'test', 'train', 'dev'
        + is_train: True, False
        + maxlen : 256, ..
        + bs: 32
        + device: 
    - Output:
        + Attibute: tag2idx, device, X, Y, dataset
        + Method: create_dataloader
'''
from keras.preprocessing.sequence import pad_sequences  
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pickle

class BertDataLoader:
    def __init__(self, dir, tokenizer, tag_values, device, is_train = False, bs = 32, maxlen = 256):
        self.dir_train = dir
        self.MAX_LEN = maxlen
        self.BATCH_SIZE = bs
        self.tag_values = tag_values
        self.tag2idx = {t: i for i, t in enumerate(self.tag_values)}
        self.device = device
        self.tokenizer = tokenizer
        self.is_train = is_train
    
    def read_dataset(self):
        with open(self.dir_train ,'rb') as f:
            _data = pickle.load(f)
        data = [sq for sq in _data if len(sq) >= 1]
        return data

    def split_data(self, data):
        #(x,y)=> X= [x...] , Y= [y....]
        X, Y = [], []
        for sent in data:
            temp_x = []
            temp_y = []
            for word in sent:
                temp_x.append(word[0])
                temp_y.append(word[1])
            X.append(temp_x)
            Y.append(temp_y)
        return X, Y    

    def check_label(self, data):
        '''
        input: [[('Hello','O'),...],...]
        output: {'O','LOC',"ORG",...}
        '''
        a = []
        for i in data:
            for j in i:
                _, l = j
                a.append(l)
        return list(set(a))

    def isSubword(self, x, idx, sub = '##'):
        return sub not in x[idx] and idx > 0 and idx < len(x) - 1 and sub not in x[idx-1] and sub not in x[idx+1]

    def cutting_subword(self, X, y):
        res_X, res_y = [], []
        punct = '.!?'
        st = 0
        cur = 0
        
        while (st < len(X)-self.MAX_LEN):
            flag = True
            for i in range(st + self.MAX_LEN - 1, st-1, -1):
                if X[i] in punct and y[i] == 'O':
                    cur = i+1
                    flag = False
                    break
            if flag:
                for i in range(st + self.MAX_LEN - 1 , st-1, -1):
                    if self.isSubword(X, i):
                        cur = i+1
                        if y[i] == 'O':
                            cur = i+1
                            break
                    
            res_X.append(X[st: cur])
            res_y.append(y[st: cur])
            st = cur

        res_X.append(X[cur:])
        res_y.append(y[cur:])
        return res_X, res_y

    def add_subword(self, sentence, text_labels):
        '''
        input:
            sentence = ['Phạm', 'Văn', 'Mạnh']
            text_labels = ['B-PER', 'I-PER','I-PER']

        output: 
            ['Phạm', 'Văn', 'M', '##ạnh'],
            ['B-PER', 'I-PER', 'I-PER', 'I-PER']
        '''
        tokenized_sentence = []
        labels = []
        for word, label in zip(sentence, text_labels):
            subwords = self.tokenizer.tokenize(word)
            tokenized_sentence.extend(subwords)
            
            labels.extend([label] * len(subwords))
        return tokenized_sentence, labels


    def add_subword2data(self, X, Y):
        '''
            input:
                sentence = [['Phạm', 'Văn', 'Mạnh',..],....]
                text_labels = [['B-PER', 'I-PER','I-PER',..],...]

            output: 
                [['Phạm', 'Văn', 'M', '##ạnh',..],....],
                [['B-PER', 'I-PER','I-PER','I-PER',..],...]
        '''
        tokenized_texts_and_labels = [self.add_subword(sentence, text_labels) for sentence, text_labels in zip(X, Y)]
        tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
        labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
        return tokenized_texts,labels
    
    
    def padding_data(self,X_subword,y_subword):
        '''
            input:
                X = [['Phạm', 'Văn', 'M', '##ạnh',..],....]
                Y = [['B-PER', 'I-PER','I-PER','I-PER',..],...]

            output: 
            [[10,20,30,40,0,0,0,0,0,0,0,0...],...],
            [[1, 2,3,4,5,5,5,5,5,5,5,5,5,...],...]
        '''
        X_padding = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in X_subword],
                          maxlen=self.MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

        y_padding = pad_sequences([[self.tag2idx.get(l) for l in lab] for lab in y_subword],
                        maxlen=self.MAX_LEN, value=self.tag2idx["PAD"], padding="post",
                        dtype="long", truncating="post")
        attention_masks = [[float(i != 0.0) for i in ii] for ii in X_padding]
        
        return X_padding, y_padding,attention_masks
    
    
    def covert2tensor(self, X_padding, Y_padding, attention_masks):
        if self.is_train == True:
            X_tensor = torch.tensor(X_padding).to(self.device) 
            y_tensor = torch.tensor(Y_padding).to(self.device) 
            masks = torch.tensor(attention_masks).to(self.device)  
        elif self.is_train == False:
            X_tensor = torch.tensor(X_padding).type(torch.LongTensor).to(self.device) 
            y_tensor = torch.tensor(Y_padding).type(torch.LongTensor).to(self.device) 
            masks = torch.tensor(attention_masks).type(torch.LongTensor).to(self.device) 
        return  X_tensor, y_tensor, masks

    def create_dataloader(self):
        dataset = self.read_dataset()
        X, Y = self.split_data(dataset)
        X_subword, y_subword = self.add_subword2data(X, Y)
        #long_subword = [seq for seq in X_subword if len(seq) > self.MAX_LEN]
        #print(f"Before cutting: \nX_subword: {X_subword[0]}, \nMax_seq: {max([len(line) for line in X_subword])}\
        #    \nThe number of seq have len larger {self.MAX_LEN}: {len(long_subword)} \nThe number of total seq: {len(X_subword)}")
        X_subword_at, y_subword_at = [], []
        for i in range(len(X_subword)):
            res_x, res_y = self.cutting_subword(X_subword[i], y_subword[i])
            X_subword_at += res_x
            y_subword_at += res_y
        #long_subword_at = [seq for seq in X_subword_at if len(seq) > self.MAX_LEN]
        #print(f"After cutting: \nX_subword: {X_subword_at[0]}, \nMax_seq: {max([len(line) for line in X_subword_at])}\
        #    \nThe number of seq have len larger: {self.MAX_LEN}: {len(long_subword_at)} \nThe number of total seq: {len(X_subword_at)}")
        X_padding, y_padding, attention_masks = self.padding_data(X_subword_at, y_subword_at)
        X_tensor,y_tensor,masks = self.covert2tensor(X_padding, y_padding, attention_masks)
        train_data = TensorDataset(X_tensor, masks, y_tensor)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = self.BATCH_SIZE)
        #labels = self.check_label(dataset)
        return train_dataloader
#################################################### SOME FUNCTION FOR DATALOADER #####################################################################