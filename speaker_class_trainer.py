#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 16:10:04 2019

@author: yonic
"""
import glob
import os
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from speaker_embed import networks
from database import tedlium
from tacotron2.layers import TacotronSTFT


SFT_CONFIG={    
    "sampling_rate": 16000,
    "filter_length": 400,
    "hop_length": 160,
    "win_length": 400,
    "mel_fmin": 0.0,
    "mel_fmax": 8000.0}

MAX_WAV_VALUE = 32768.0
EMBEDDING_SIZE = 512
#TEDIUM_NUM_OF_CLASSES = 774
TEDIUM_NUM_OF_CLASSES = 140
MAX_DATA_CHUNK = 11
MAX_TRAIN_EPOCH = 1000

class EmbeddingNetClassifier(networks.EmbeddingNet):
    def __init__(self,num_of_clusses):
        super(EmbeddingNetClassifier, self).__init__()
        self.logits = networks.EmbeddingNet()        
        self.fc = nn.Linear(EMBEDDING_SIZE,num_of_clusses)
        
    def forward(self, x): 
        x = self.logits(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)        
    

def data_generator(data_base,chunk_length_in_sec,label_order_list):    
    stft = TacotronSTFT(**SFT_CONFIG)
    keys = data_base.get_db_keys()        
    batch_train_x = []
    batch_train_y = []    
    
    while True:        
        for k in keys:                
            label = np.array([label_order_list.index(k)])            
            for t in data_base.get_db_wv_times(k):                       
                sampling_rate, speech = data_base.get_wav(k,*t)
                chunks = int(len(speech)/sampling_rate/chunk_length_in_sec)
                audio_length = sampling_rate*chunk_length_in_sec                
                for chunk in range(chunks):                
                    audio = speech[chunk*audio_length:(chunk+1)*audio_length]
                    audio_norm = audio / MAX_WAV_VALUE
                    audio_norm = torch.from_numpy(audio_norm).float()
                    audio_norm = audio_norm.unsqueeze(0)
                    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
                    melspec = stft.mel_spectrogram(audio_norm)
                    mel_np = melspec.detach().numpy()
                    for i in range(mel_np.shape[1]):
                        channel_mean = np.mean(mel_np[0,i,:])  
                        mel_np[0,i,:] = mel_np[0,i,:] - channel_mean                    
                    
                    batch_train_x.append(mel_np)
                    batch_train_y.append(label)
            if len(batch_train_x) >0 and len(batch_train_y)>0:
                yield np.array(batch_train_x),np.concatenate(np.array(batch_train_y))
            else:
                yield np.array([]),np.array([])
            
            batch_train_x  = []
            batch_train_y = []  
            
        yield None,None
                        
        
'''Generate train matrix, take labels that have samples more than threshold
'''
def merger_train(threshold):
    labels = []
    H = pickle.load(open('./data/meta.bin','rb'))    
    for k in H.keys():
        if H[k]>threshold:
            labels.append(k)
            
    X_train = []
    Y_train = []
    X_val  = []
    Y_val  = []
    
    index = 0
    for l in labels:        
        fname = './data/data_x_{}.npy'.format(l)
        x = np.load(fname)        
        
        fname = './data/data_y_{}.npy'.format(l)
        y = np.load(fname)
        
        index = list(range(len(x)))
        np.random.shuffle(index)
        x = x[index]
        y = y[index]
        
        X_train.append(x[0:-1])
        Y_train.append(y[0:-1])
        X_val.append(x[-1])
        Y_val.append(y[-1])
        print('Append label:{}'.format(l))        
    
    
    X_train = np.concatenate(X_train)
    X_val = np.concatenate(X_val)
    Y_train = np.concatenate(Y_train)
    Y_val = np.array(Y_val)
    print('X_train shape:',X_train.shape)
    print('X_val shape:',X_val.shape)
    
    index = list(range(len(X_train)))
    np.random.shuffle(index)
    
    X_train = X_train[index]
    Y_train = Y_train[index]
    
    fname = './data/x_train_merged'
    np.save(fname,X_train)
    
    fname = './data/x_val_merged'
    np.save(fname,X_val)    
    
    fname = './data/y_train_merged'
    np.save(fname,Y_train)
    
    fname = './data/y_val_merged'
    np.save(fname,Y_val)
    
    
'''Generate train matrix, take labels that have samples more than threshold
'''
def merger_embed_train(threshold1,threshold2,prefix):
    labels = []
    H = pickle.load(open('./data/meta.bin','rb'))    
    for k in H.keys():
        if H[k]>=threshold1 and H[k]<=threshold2:
            labels.append(k)
            
    X_test = []
    Y_test = []    
    
    index = 0
    for l in labels:        
        fname = './data/data_x_{}.npy'.format(l)
        x = np.load(fname)        
        
        fname = './data/data_y_{}.npy'.format(l)
        y = np.load(fname)
        
        index = list(range(len(x)))
        np.random.shuffle(index)
        x = x[index]
        y = y[index]
        
        X_test.append(x[0:-1])
        Y_test.append(y[0:-1])        
        print('Append label:{}'.format(l))        
    
    
    X_test = np.concatenate(X_test)
    Y_test = np.concatenate(Y_test)        
    print('X_test shape:',X_test.shape)
    print('Y_test shape:',Y_test.shape)
    
    index = list(range(len(X_test)))
    np.random.shuffle(index)
    
    X_test = X_test[index]
    Y_test = Y_test[index]
    
    fname = './data/x_embed_merged_{}'.format(prefix)
    np.save(fname,X_test)   
    
    fname = './data/y_embed_merged_{}'.format(prefix)
    np.save(fname,Y_test)    
    
        
    
    
def train_matrix_generator():            
    chunk_length_in_sec=3
    
    data_base = tedlium.TedLium(mode='train')
    label_order_list = sorted(data_base.get_db_keys())
    num_of_clusses = len(set(label_order_list))
    print('label_order_list',label_order_list)
    
    print('Tedium DB num of classes is:',num_of_clusses)    
    G_data = data_generator(data_base=data_base,
        chunk_length_in_sec=chunk_length_in_sec,
        label_order_list=label_order_list)

    H = {}
    while True:
        data,target = next(G_data)
        if data is None:        
            break        
        
        if len(target)>0:
            label = target[0]
            fname = './data/data_x_{}'.format(label)        
            np.save(fname,data)
            
            fname = './data/data_y_{}'.format(label)        
            np.save(fname,target)        
            print('Label:{}, size:{}'.format(label,len(target)))
            H[label] = len(target)        
    
    pickle.dump(H,open('./data/meta.bin','wb'))
                    
 
def get_ckpt(folder='./ckpt/'):
    
    def parse_cktp_name(ckpt_name):        
        epoch = ckpt_name.split('/')[-1].split('epoch')[1].split('_')[1]        
        return int(epoch)        
    H = {}
    for f in glob.glob(os.path.join(folder,'model_epoch*.pt')):
        epoch = parse_cktp_name(f)               
        H[epoch] = f
    
    return H
    

'''Train from scrath if epoch == 0, else try to load from chekpoint
'''
def train_model(start_epoch,learning_rate): 
    torch.cuda.init()        
    
    chunk  = 0
    train_batch_size = 128
    validation_batch_size = 128    
    
    valid_data = []
    valid_target = []
    fname = './data/x_train_merged.npy'
    data = np.load(fname)    
    fname = './data/x_val_merged.npy'
    valid_data = np.load(fname)
    valid_data = np.expand_dims(valid_data,axis=1)

    fname = './data/y_train_merged.npy'
    target = np.load(fname)
    fname = './data/y_val_merged.npy'
    valid_target = np.load(fname)
    
    unique_label = np.unique(valid_target)
    I = {}
    index = 0
    for l in unique_label:
        if not l in I:
            I[l] = index
            index += 1
        
    print('Train fix')        
    for i in range(len(valid_target)):        
        valid_target[i] = I[valid_target[i]]
        
    print('Val fix')        
    for i in range(len(target)):        
        target[i] = I[target[i]]        
    
    num_of_batches_in_validation = int(np.ceil(len(valid_target)/validation_batch_size))
    validation_index = list(range(len(valid_data)))
    num_of_batches_in_chunk = int(np.ceil(len(target)/train_batch_size))
    print('num_of_batches_in_validation:',num_of_batches_in_validation,', num_of_batches_in_chunk:',num_of_batches_in_chunk)
    print('Validation set size:{}, unique labels:{}'.format(len(valid_target),len(np.unique(valid_target))))
    TRAIN_LOSS = []
    VAL_LOSS = []
    
    device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    learning_rate = learning_rate  
    
    embedding_net = EmbeddingNetClassifier(num_of_clusses=len(unique_label))
    if start_epoch > 0:
        ckpt = get_ckpt()
        fname = ckpt[start_epoch]
        print('Load model from:',fname)
        embedding_net.load_state_dict(torch.load(fname))
        
    embedding_net.cuda()    
    optimizer = torch.optim.Adam(embedding_net.parameters(), 
        lr=learning_rate)    
    for epoch in range(start_epoch,MAX_TRAIN_EPOCH):        

        random_index = list(range(len(target)))
        np.random.shuffle(random_index)            
        
        batch_index = 0
        
        ######################## TRAIN ####################################
        embedding_net.train()
        train_loss = 0
        for batch_index in range(num_of_batches_in_chunk-1): 
            train_data_batch_index = random_index[batch_index*train_batch_size:batch_index*train_batch_size+train_batch_size]            
            batch_data = data[train_data_batch_index,:,:,:]
            batch_target = target[train_data_batch_index]                
            x = torch.tensor(batch_data,device='cuda')                
            y = Variable(torch.tensor(batch_target,device='cuda').long())
            optimizer.zero_grad()
            net_out = embedding_net(x)                        
            loss = F.nll_loss(net_out, y)
            loss.backward()                
            optimizer.step()                        
            train_loss += loss.item()
            del x
            del y                      
            del batch_data
            del batch_target            
            
        ######################## VALIDATION ###############################
        embedding_net.eval()            
        val_loss = 0
        correct = 0            
        
        with torch.no_grad():   
            for batch_index in range(num_of_batches_in_validation):
                validation_batch_index = validation_index[batch_index*validation_batch_size:batch_index*validation_batch_size+validation_batch_size]                                    
                batch_data = valid_data[validation_batch_index,:,:,:]
                batch_target = valid_target[validation_batch_index]                
                x_val = torch.tensor(batch_data,device='cuda')
                y_val = Variable(torch.tensor(batch_target,device='cuda').long())
                net_out = embedding_net(x_val)
                val_loss += F.nll_loss(net_out, y_val)
                pred = net_out.argmax(dim=1, keepdim=True) # get the index of the max log-probability                                                            
                correct += pred.eq(y_val.view_as(pred)).sum().item()
                del x_val
                del y_val
        
        train_loss/= num_of_batches_in_chunk            
        val_loss /= num_of_batches_in_validation 
        TRAIN_LOSS.append(train_loss)
        VAL_LOSS.append(val_loss)
        print('\nTest set: epoch:{},chunk:{},train loss: {:.4f},test loss:{:.4f},Accuracy: {}/{} ({:.0f}%)\n'.format(
                epoch,chunk,train_loss, val_loss,correct, 
                len(valid_target),100. * correct / len(valid_target)))
        fname = './stat/stat_epoch_{}_chunk_{}.bin'.format(epoch,chunk)
        pickle.dump([TRAIN_LOSS,VAL_LOSS],open(fname,'wb'))
            
        fname = './ckpt/model_epoch_{}_chunk_{}_val_loss_{:.4f}_acc_{:.0f}.pt'.format(epoch,chunk,val_loss,100. * correct / len(valid_target))
        torch.save(embedding_net.state_dict(),fname)                

    
if __name__ == '__main__':
    #train_matrix_generator()
    #merger_train(threshold=200)
    merger_embed_train(threshold1=0,threshold2=100,prefix='0to100')
    #train_model(start_epoch=14,learning_rate=1e-3)
    

    
    
    