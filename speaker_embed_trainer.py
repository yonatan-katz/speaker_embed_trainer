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
MAX_TRAIN_EPOCH = 1000
PRE_TRAINDEX_CLASSIFIER_MODEL = './ckpt/model_epoch_17_chunk_0_val_loss_0.1827_acc_98.pt'

'''Embeding net on top of classifier
'''
class EmbeddingNetClassifier(networks.EmbeddingNet):
    def __init__(self,num_of_clusses):
        super(EmbeddingNetClassifier, self).__init__()
        self.logits = networks.EmbeddingNet()        
        self.fc = nn.Linear(EMBEDDING_SIZE,num_of_clusses)
        
    def forward(self, x): 
        x = self.logits(x)        
        return x
    

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
def merger(threshold = 100):
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

def generate_diff_index(target):
    index1 = list(range(len(target)))   
    np.random.shuffle(index1)    
    
    while True:
        i1 = np.random.choice(index1)
        i2 = np.random.choice(index1)  
        if target[i1] != target[i2]:            
            yield i1,i2

def generate_same_index(target):    
    P = {}
    for t in target:
        i = np.where(target==t)[0]
        P[t] = i
        
    while True:
        t = np.random.choice(target,size=1)[0]              
        i1,i2 = np.random.choice(P[t],size=2)        
        yield i1,i2    


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) + (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive  
    
def generate_batch(index_generator,data,label,size):
    X1 = []        
    X2 = []        
    L = []        
    for i in range(size):
        i1,i2 = next(index_generator)            
        X1.append(data[i1])
        X2.append(data[i2])
        L.append(label)        
    return np.array(X1),np.array(X2),np.array(L)
    
def generate_epoch(SG,DG,data_set,epoch_size):
    X1_s,X2_s,L_s = generate_batch(SG,data_set,1.0,epoch_size)
    X1_d,X2_d,L_d = generate_batch(DG,data_set,0.0,epoch_size)
    X1 = np.concatenate([X1_s,X1_d])
    X2 = np.concatenate([X2_s,X2_d])
    L = np.concatenate([L_s,L_d])
    x_index = np.array(list(range(len(X1))))
    np.random.shuffle(x_index)
    X1 = X1[x_index]
    X2 = X2[x_index]
    L =  L[x_index]
    X1_v = Variable(torch.tensor(X1,device='cuda'))
    X2_v = Variable(torch.tensor(X2,device='cuda'))
    L_v = Variable(torch.tensor(L,device='cuda').float())        
    return X1_v,X2_v,L_v

'''Train from scrath if epoch == 0, else try to load from chekpoint
'''
def train_model(start_epoch,learning_rate): 
    torch.cuda.init()        
    
    chunk  = 0
    train_batch_size = 64
    
    valid_data = []
    valid_target = []
    fname = './data/x_embed_merged_100to200.npy'
    data = np.load(fname)    
    fname = './data/x_embed_merged_0to100.npy'
    valid_data = np.load(fname)    

    fname = './data/y_embed_merged_100to200.npy'
    target = np.load(fname)
    fname = './data/y_embed_merged_0to100.npy'
    valid_target = np.load(fname)  
    
    print('Validation set size:{}, unique labels:{}'.format(len(valid_target),len(np.unique(valid_target))))
    TRAIN_LOSS = []
    VAL_LOSS = []
    
    device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    learning_rate = learning_rate  
    
    embedding_net = networks.EmbeddingNet()        
    print('Load model from:',PRE_TRAINDEX_CLASSIFIER_MODEL)
    embedding_net.load_state_dict(torch.load(PRE_TRAINDEX_CLASSIFIER_MODEL),strict=False)
    embedding_net.eval()
    print('Pre trained model loaded')
    
    #equal pair index generator
    SG_train = generate_same_index(target)
    SG_valid = generate_same_index(valid_target)
    
    #diff pair index generator
    DG_train = generate_diff_index(target)
    DG_valid = generate_diff_index(valid_target)
        
        
    embedding_net.cuda()    
    optimizer = torch.optim.Adam(embedding_net.parameters(), 
        lr=learning_rate,weight_decay=0.01)
    loss_func = ContrastiveLoss() 
    for epoch in range(start_epoch,MAX_TRAIN_EPOCH):       
        batch_index = 0        
        ######################## TRAIN ####################################
        embedding_net.train()
        train_loss = 0
        X1_v,X2_v,L_v = generate_epoch(SG_train, DG_train, data, 640*4)        
        train_batch_size = 64
        num_of_batches_in_chunk = int(np.ceil(len(X1_v)/train_batch_size))            
        print('Train num of batches:',num_of_batches_in_chunk)
        
        for _index in range(num_of_batches_in_chunk-1):           
            batch_index = range(_index,_index+train_batch_size)
            x1, x2, l = X1_v[batch_index], X2_v[batch_index], L_v[batch_index]                        
            f1=embedding_net(x1)
            f2=embedding_net(x2)
            loss = loss_func(f1,f2,l)            
            optimizer.zero_grad()                      
            loss.backward()                
            optimizer.step()                        
            train_loss += loss.item()            
            del x1
            del x2
            del l        
        
        del X1_v
        del X2_v
        del L_v
        
        ######################## VALIDATION ###############################               
        num_of_batches_in_validation = 100
        embedding_net.eval()
        val_loss = 0        
        euclidean_distance_same = 0.0
        euclidean_distance_diff = 0.0
        with torch.no_grad():   
            for _ in range(num_of_batches_in_validation):
                X1,X2,L = generate_batch(SG_valid,valid_data,1.0,1)
                X1_v = Variable(torch.tensor(X1,device='cuda'))
                X2_v = Variable(torch.tensor(X2,device='cuda'))
                L_v = Variable(torch.tensor(L,device='cuda').float())                  
                
                f1=embedding_net(X1_v)
                f2=embedding_net(X2_v)
                val_loss += loss_func(f1,f2,L_v)     
                euclidean_distance_same += torch.pow(F.pairwise_distance(f1, f2),2).item()
                del X1_v
                del X2_v
                del L_v
                del X1
                del X2
                del L
                
                X1,X2,L = generate_batch(DG_valid,valid_data,0.0,1)
                X1_v = Variable(torch.tensor(X1,device='cuda'))
                X2_v = Variable(torch.tensor(X2,device='cuda'))
                L_v = Variable(torch.tensor(L,device='cuda').float()) 
                f1=embedding_net(X1_v)
                f2=embedding_net(X2_v)
                val_loss += loss_func(f1,f2,L_v)
                euclidean_distance_diff += torch.pow(F.pairwise_distance(f1, f2),2).item()
                del X1_v
                del X2_v
                del L_v
                del X1
                del X2
                del L                
        
        
        train_loss/= num_of_batches_in_chunk
        val_loss /= num_of_batches_in_validation
        euclidean_distance_same /=num_of_batches_in_validation
        euclidean_distance_diff /=num_of_batches_in_validation
        margin = euclidean_distance_diff-euclidean_distance_same
        
        TRAIN_LOSS.append(train_loss)
        VAL_LOSS.append(val_loss)
        
        print('Test set: epoch:{},chunk:{},train loss: {:.4f},val loss:{:.4f}, eucl diff:{}, eucl same:{},margin:{}\n'.
                  format(epoch,chunk,train_loss,val_loss,euclidean_distance_diff,euclidean_distance_same,margin))
        fname = './stat_embed/stat_epoch_{}_chunk_{}.bin'.format(epoch,chunk)
        pickle.dump([TRAIN_LOSS,VAL_LOSS],open(fname,'wb'))
            
        fname = './ckpt_embed/model_epoch_{}_chunk_{}_val_loss_{:.4f}_margin_{:.4f}.pt'.format(epoch,chunk,val_loss,margin)
        torch.save(embedding_net.state_dict(),fname)
        
        
def eval_model():
    MODEL_CKPT = './ckpt_embed/model_epoch_35_chunk_0_val_loss_0.5735_margin_0.9335.pt'    
    
    fname = './data/x_embed_merged_0to100.npy'
    valid_data = np.load(fname)
    
    fname = './data/y_embed_merged_0to100.npy'
    valid_target = np.load(fname)   
    print('Uniq labels:',len(np.unique(valid_target)), ', samples:',len(valid_target))
    
    
    torch.cuda.init()
    device = torch.cuda.current_device()
    torch.cuda.set_device(device)    
    
    embedding_net = networks.EmbeddingNet()        
    print('Load model from:',MODEL_CKPT)
    embedding_net.load_state_dict(torch.load(MODEL_CKPT),strict=False)
    embedding_net.eval()
    embedding_net.cuda()
    print('Pre trained model loaded')
    
    #diff pair index generator 
    SG_valid = generate_same_index(valid_target)
    DG_valid = generate_diff_index(valid_target)
    euclidean_distance_same = 0.0
    euclidean_distance_diff = 0.0
    num_of_batches_in_validation = 10000
    with torch.no_grad():   
        for _ in range(num_of_batches_in_validation):
            X1,X2,L = generate_batch(SG_valid,valid_data,1.0,1)
            X1_v = Variable(torch.tensor(X1,device='cuda'))
            X2_v = Variable(torch.tensor(X2,device='cuda'))
            L_v = Variable(torch.tensor(L,device='cuda').float())                              
            f1=embedding_net(X1_v)
            f2=embedding_net(X2_v)            
            euclidean_distance_same += torch.pow(F.pairwise_distance(f1, f2),2).item()            
            del X1_v
            del X2_v
            del L_v
            del X1
            del X2
            del L
            
            X1,X2,L = generate_batch(DG_valid,valid_data,0.0,1)
            X1_v = Variable(torch.tensor(X1,device='cuda'))
            X2_v = Variable(torch.tensor(X2,device='cuda'))
            L_v = Variable(torch.tensor(L,device='cuda').float()) 
            f1=embedding_net(X1_v)
            f2=embedding_net(X2_v)            
            euclidean_distance_diff += torch.pow(F.pairwise_distance(f1, f2),2).item()
            
            del X1_v
            del X2_v
            del L_v
            del X1
            del X2
            del L
            
    euclidean_distance_same /=num_of_batches_in_validation
    euclidean_distance_diff /=num_of_batches_in_validation 
    
    print('euclidean_distance_same:{},euclidean_distance_diff:{}'.format(euclidean_distance_same,euclidean_distance_diff))
    
    
    
    
    
    
if __name__ == '__main__':
    #train_matrix_generator()
    #merger(threshold=200)
    #train_model(start_epoch=0,learning_rate=1e-4)
    eval_model()
    

    
    
    