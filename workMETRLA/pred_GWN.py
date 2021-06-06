import sys
import os
import shutil
import math
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
import Metrics
from GWN import *
from Param import *
from Param_GWN12 import *

def getXSYS(data, mode, modelname):
    if modelname == 'GraphWaveNet' and GWN_CHANNEL==2:
        data_feature = data.reshape(data.shape[0],data.shape[1],1)
        feature_list = [data_feature]
        num_samples, num_nodes = data1.shape
        time_ind = (data.index.values - data.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
        data = np.concatenate(feature_list, axis=-1)
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :,:]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :,:]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN, data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :,:]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :,:]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    print('XS.SHAPE: ',XS.shape)
    if modelname == 'GraphWaveNet':
        if GWN_CHANNEL==1:
            XS, YS = XS[:, np.newaxis, :, :], YS[:, :, :, np.newaxis]
            XS = XS.transpose(0,3,2,1)
            YS = YS.transpose(0,3,2,1)
        if GWN_CHANNEL==2:
            XS = XS.transpose(0,3,2,1)
            YS = YS.transpose(0,3,2,1)
    return XS, YS

def getModel(name):
    # ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 16, 64], [64, 16, 64]], TIMESTEP_IN, N_NODE, 0
    adj_mx =load_adj(GWN_ADJPATH,GWN_ADJTYPE)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    print('Graph WaveNet Parameters:')
    print('GWN_ADJTYPE:',GWN_ADJTYPE,' ,GWN_GCN_BOOL:',GWN_GCN_BOOL,' ,GWN_APTONLY:',GWN_APTONLY,' ,GWN_ADDAPTADJ:',GWN_ADDAPTADJ,' ,GWN_RANDOWADJ:',GWN_RANDOWADJ)
    if GWN_RANDOWADJ:
        adjinit = None
    else:
        adjinit = supports[0]
    if GWN_APTONLY:
        supports = None
    model = gwnet(device, num_nodes = N_NODE, dropout = GWN_DROPOUT, supports=supports, gcn_bool=GWN_GCN_BOOL, addaptadj=GWN_ADDAPTADJ, aptinit=adjinit, in_dim=GWN_CHANNEL, out_dim=GWN_TIMESTEP_OUT, residual_channels=GWN_N_HID, dilation_channels=GWN_N_HID, skip_channels=GWN_N_HID * 8, end_channels=GWN_N_HID * 16,kernel_size=2,blocks=4,layers=2).to(device)
    return model

def evaluateModel(model, criterion, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            x = nn.functional.pad(x,(1,0,0,0))
            y_pred = model(x)
            y_pred = y_pred
            predict = scaler.inverse_transform(y_pred)
            real = torch.unsqueeze(y,dim=1)
            l = criterion(predict, real, 0.0)
            l_sum += l.item() * real.shape[0]
            n += real.shape[0]
        return l_sum / n

def predictModel(model, data_iter):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            x = nn.functional.pad(x,(1,0,0,0))
            YS_pred_batch = model(x)
            YS_pred_batch = YS_pred_batch
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred

def trainModel(name, mode, XS, YS):
#     YS = YS[:,0,:,:]
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    model = getModel(name)
    summary(model, (GWN_CHANNEL,N_NODE,GWN_TIMESTEP_OUT), device="cuda:{}".format(GPU))
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1-TRAINVALSPLIT))
    print('XS_torch.shape:  ', XS_torch.shape)
    print('YS_torch.shape:  ', YS_torch.shape)
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, BATCHSIZE, shuffle=True)
    if LOSS == "GraphWaveNetLoss":
        criterion = Metrics.masked_mae
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()
    if OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARN)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    
    min_val_loss = np.inf
    wait = 0
    clip = 5
    for epoch in range(EPOCH):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            x = nn.functional.pad(x,(1,0,0,0)).to(device)
            y = y[:,0,:,:].to(device)
            y_pred = model(x)
            real = torch.unsqueeze(y,dim=1)
            predict = y_pred
            loss = criterion(predict, real,0.0)
#             loss = Metrics.masked_mae(predict, real,0.0)
            print('loss ok')
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            loss_sum += loss.item() * real.shape[0]
            n += real.shape[0]
#         scheduler.step()
        train_loss = loss_sum / n
        val_loss = evaluateModel(model, criterion, val_iter)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), PATH + '/' + name + '.pt')
        else:
            wait += 1
            if wait == PATIENCE:
                print('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %.10f, %s, %.10f\n" % ("epoch", epoch, "time used", epoch_time, "train loss", train_loss, "validation loss:", val_loss))
            
    torch_score = evaluateModel(model, criterion, train_iter)
    YS_pred = predictModel(model, torch.utils.data.DataLoader(trainval_data, BATCHSIZE, shuffle=False))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = scaler.inverse_transform(np.squeeze(YS)), scaler.inverse_transform(np.squeeze(YS_pred))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())
        
def testModel(name, mode, XS, YS):
    YS = YS[:,0,:,:]
    print('Model Testing Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
    model = getModel(name)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))
    
    torch_score = evaluateModel(model, criterion, test_iter)
    YS_pred = predictModel(model, test_iter)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
#     YS, YS_pred = np.squeeze(YS), np.squeeze(YS_pred)
#     YS = scaler.inverse_transform(YS)
#     YS_pred = scaler.inverse_transform(YS_pred)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    for i in range(1):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = YS[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Testing Ended ...', time.ctime())
        
################# Parameter Setting #######################
MODELNAME = 'GraphWaveNet'
KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = '../' + KEYWORD
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)
torch.backends.cudnn.deterministic = True
###########################################################
param = sys.argv
if len(param) == 2:
    GPU = param[-1]
else:
    GPU = '0'
device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
###########################################################

data2 = pd.read_hdf(FLOWPATH)
scaler = StandardScaler()
data1 = scaler.fit_transform(data2)
print('data1.shape', data1.shape)
    
def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('Param_GWN12.py', PATH)
        
    print(KEYWORD, 'training started', time.ctime())
    trainXS, trainYS = getXSYS(data1, data2, 'TRAIN', MODELNAME)
    print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    trainModel(MODELNAME, 'train', trainXS, trainYS)
    
    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS = getXSYS(data1, data2, 'TEST', MODELNAME)
    print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
    testModel(MODELNAME, 'test', testXS, testYS)

    
if __name__ == '__main__':
    main()

