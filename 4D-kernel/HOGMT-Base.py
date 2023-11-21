import torch
import torch.nn as nn
import random
import math
import numpy as np
import cmath
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch.nn.utils as utils
import time
from collections import deque
import scipy.io as sio


np_kernel_split = np.load('kernal_4x5x4x5.npy')
kernel_split = torch.from_numpy(np_kernel_split)
TotalSampleSize = kernel_split.size(0)
print(TotalSampleSize)
NumberOfUsers = kernel_split.size(1)
TimeLength = kernel_split.size(2)
kernel_split = kernel_split.reshape((TotalSampleSize, NumberOfUsers * TimeLength * NumberOfUsers * TimeLength, 2))
kernel = torch.complex(kernel_split[:,:,0], kernel_split[:,:,1])
kernel = kernel.reshape((TotalSampleSize, NumberOfUsers * TimeLength, NumberOfUsers * TimeLength))
kernel_split = kernel_split.reshape((TotalSampleSize, NumberOfUsers * TimeLength * NumberOfUsers * TimeLength* 2))

penalty = 0.01
slope = 0.1
NumberOfEpochs = 500
BatchSize = 16
NumberOfEigenFun = 20


TrainSampleSize = int((TotalSampleSize/5)*4)
TestSampleSize = int((TotalSampleSize/5)*1)

NumberOfTrainBatches = int((TrainSampleSize/BatchSize))
NumberOfTestBatches = int((TestSampleSize/BatchSize))

EigneVectorSize = NumberOfUsers * TimeLength
VmatrixLocation = NumberOfEigenFun + NumberOfEigenFun*EigneVectorSize
NNOutputPerEigenFn = 2 * NumberOfUsers * TimeLength + 1
NumberOfOutputs = NumberOfEigenFun * NNOutputPerEigenFn

SizeOfInputLayer = NumberOfUsers * TimeLength * NumberOfUsers * TimeLength * 2
SizeOfHiddenLayer1 =  SizeOfInputLayer
SizeOfOutputLayer = 2 * NumberOfOutputs
SizeOfHiddenLayer2 =  (int)((SizeOfInputLayer + SizeOfOutputLayer)/2)


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

#print(device)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(0)

class NNetwork(nn.Module):
    def __init__(self):
        super(NNetwork, self).__init__()
        self.fc1 = nn.Linear(SizeOfInputLayer, SizeOfHiddenLayer1)
        self.relu1 = nn.LeakyReLU(slope)
        self.fc2 = nn.Linear(SizeOfHiddenLayer1, SizeOfHiddenLayer2)
        self.relu2 = nn.LeakyReLU(slope)
        self.fc3 = nn.Linear(SizeOfHiddenLayer2, SizeOfOutputLayer)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
model = NNetwork()
print(model)

def orthogonalityCheck(matrix, batch_Size):
    sumVal = 0
    for i in range(batch_Size):
        for j in range(NumberOfEigenFun):
            for k in range(NumberOfEigenFun-(j+1)):
                sumVal = sumVal + torch.abs(torch.sum(matrix[i,:,j] * matrix[i,:,(k+j+1)]))
    return sumVal/batch_Size

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, nnOutput, train_kernals_2d, batch_Size):
        nnOut_matrix_complex = torch.complex(nnOutput[:,0:NumberOfOutputs], nnOutput[:,NumberOfOutputs:SizeOfOutputLayer])

        diagonalMatrix = torch.diag_embed(nnOut_matrix_complex[:,0:NumberOfEigenFun])

        UVectorMatrixTmp = nnOut_matrix_complex[:, NumberOfEigenFun:VmatrixLocation]
        UVectorMatrix = UVectorMatrixTmp.view(batch_Size, EigneVectorSize, NumberOfEigenFun)
        VVectorMatrixTmp = nnOut_matrix_complex[:, VmatrixLocation:NumberOfOutputs]
        VVectorMatrix = VVectorMatrixTmp.view(batch_Size, EigneVectorSize, NumberOfEigenFun)
        transposedV = torch.transpose(VVectorMatrix, 1, 2)

        obj1 = orthogonalityCheck(UVectorMatrix, batch_Size)
        obj2 = orthogonalityCheck(VVectorMatrix, batch_Size)

        kernels_pred = UVectorMatrix @ diagonalMatrix @ transposedV
        constraint = penalty * obj1 + penalty * obj2 
        mse_norm = torch.pow(torch.norm((train_kernals_2d - kernels_pred),p=2),2)/torch.pow(torch.norm(train_kernals_2d,p=2),2)
        loss = mse_norm + constraint
        return loss , obj1, obj2

custom_loss = CustomLoss()

train_loss_train = torch.zeros((NumberOfEpochs,1))
test_avg_loss_train = torch.zeros((NumberOfEpochs,1))
test_loss_train = torch.zeros((NumberOfEpochs,1))
train_obj1_train = torch.zeros((NumberOfEpochs,1))
test_avg_obj1_train = torch.zeros((NumberOfEpochs,1))
test_obj1_train = torch.zeros((NumberOfEpochs,1))
train_obj2_train = torch.zeros((NumberOfEpochs,1))
test_avg_obj2_train = torch.zeros((NumberOfEpochs,1))
test_obj2_train = torch.zeros((NumberOfEpochs,1))


#obj1_wind = torch.ones((10 ,1))
#obj2_wind = torch.ones((10 ,1))

epochs = 0

def trainNN(train_kernel_split, train_kernel, test_kernel_split, test_kernel):
    lr = 0.00001
    train_kernel_split = train_kernel_split #.to(device)
    train_kernel = train_kernel #.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    testlossPerEpoch = 0
    testobj1PerEpoch = 0
    testobj2PerEpoch = 0
    testlossPer1 = 0
    testobj1Per1 = 0
    testobj2Per1 = 0
    
    for t in range(NumberOfEpochs):
#        print(f'Epoch {t+1}')
        model.train()
        trainlossPerEpoch = 0
        trainobj1PerEpoch = 0
        trainobj2PerEpoch = 0
        for b in range(NumberOfTrainBatches):
            nnOut = model(train_kernel_split[(b*BatchSize):(b+1)*BatchSize,:])
            loss , obj_1_tmp, obj_2_tmp = custom_loss(nnOut, train_kernel[(b*BatchSize):(b+1)*BatchSize, :, :], BatchSize)
            trainlossPerEpoch = trainlossPerEpoch + loss
            trainobj1PerEpoch = trainobj1PerEpoch + obj_1_tmp
            trainobj2PerEpoch = trainobj2PerEpoch + obj_2_tmp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        testlossPerEpoch = 0
        testobj1PerEpoch = 0
        testobj2PerEpoch = 0

        model.eval()
        with torch.no_grad():
            for v in range(NumberOfTestBatches):
                nnOut = model(test_kernel_split[(v*BatchSize):(v+1)*BatchSize,:])
                loss , obj_1, obj_2 = custom_loss(nnOut, test_kernel[(v*BatchSize):(v+1)*BatchSize, :, :], BatchSize)
                testlossPerEpoch = testlossPerEpoch + loss
                testobj1PerEpoch = testobj1PerEpoch + obj_1
                testobj2PerEpoch = testobj2PerEpoch + obj_2            
    
        train_loss_train[t] = trainlossPerEpoch/NumberOfTrainBatches
        test_avg_loss_train[t] = testlossPerEpoch/NumberOfTestBatches
        test_loss_train[t] = testlossPer1
        train_obj1_train[t] = trainobj1PerEpoch/NumberOfTrainBatches
        test_avg_obj1_train[t] = testobj1PerEpoch/NumberOfTestBatches
        test_obj1_train[t] = testobj1Per1
        train_obj2_train[t] = trainobj2PerEpoch/NumberOfTrainBatches
        test_avg_obj2_train[t] = testobj2PerEpoch/NumberOfTestBatches
        test_obj2_train[t] = testobj2Per1  

            
        if (t+1)%25 == 0:
            # Prepare data to save

            train_loss_train_tmp = train_loss_train.numpy()
            test_avg_loss_train_tmp = test_avg_loss_train.numpy()
            train_obj1_train_tmp = train_obj1_train.numpy()
            test_avg_obj1_train_tmp = test_avg_obj1_train.numpy()
            train_obj2_train_tmp = train_obj2_train.numpy()
            test_avg_obj2_train_tmp = test_avg_obj2_train.numpy()

            data_to_save = {
                'trained_epochs' : t,
                'train_loss': train_loss_train_tmp,
                'test_loss': test_avg_loss_train_tmp,
                'o1_train' : train_obj1_train_tmp,
                'o1_test' : test_avg_obj1_train_tmp,
                'o2_train' : train_obj2_train_tmp,
                'o2_test' : test_avg_obj2_train_tmp,                
            }

            # Save the data to a .mat file
            sio.savemat('N20.mat', data_to_save)
            torch.save(model.state_dict(), 'N20.pth')   
        

train_kernel_split = kernel_split[0:TrainSampleSize,:]
train_kernel_2d = kernel[0:TrainSampleSize,:,:]
test_kernel_split = kernel_split[TrainSampleSize:TotalSampleSize,:]
test_kernel_2d = kernel[TrainSampleSize:TotalSampleSize,:,:]

start_time = time.time()
trainNN( train_kernel_split, train_kernel_2d,test_kernel_split, test_kernel_2d)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time/60} minutes")


##import scipy.io as sio
### Prepare data to save
##
###x = x.numpy()
##train_loss_train_tmp = train_loss_train.numpy()
##test_avg_loss_train_tmp = test_avg_loss_train.numpy()
##train_obj1_train_tmp = train_obj1_train.numpy()
##test_avg_obj1_train_tmp = test_avg_obj1_train.numpy()
##train_obj2_train_tmp = train_obj2_train.numpy()
##test_avg_obj2_train_tmp = test_avg_obj2_train.numpy()
##
###penalty_2_train_tmp = penalty_2_train.numpy()
##
##
##
##data_to_save = {
##    'trained_epochs' : epochs,
##    'train_loss': train_loss_train_tmp,
##    'test_loss': test_avg_loss_train_tmp,
##    'o1_train' : train_obj1_train_tmp,
##    'o1_test' : test_avg_obj1_train_tmp,
##    'o2_train' : train_obj2_train_tmp,
##    'o2_test' : test_avg_obj2_train_tmp,
##}
##
### Save the data to a .mat file
##sio.savemat('N20.mat', data_to_save)
##torch.save(model.state_dict(), 'N20.pth')

##
##model.eval()
##
##with torch.no_grad():
##    nnOut = model(test_kernel_split[0:BatchSize,:])
##    loss , obj_1, obj_2 = custom_loss(nnOut, test_kernel_2d[0:BatchSize, :, :], BatchSize, 0, 0, 0)
##    print(obj_1/(NumberOfEigenFun*(NumberOfEigenFun-1)))
##    print(obj_2/(NumberOfEigenFun*(NumberOfEigenFun-1)))
##
##model2 = NNetwork()
##model2.load_state_dict(torch.load('N20.pth'))
##model2.eval()
##
##with torch.no_grad():
##    nnOut = model2(test_kernel_split[0:BatchSize,:])
##    loss , obj_1, obj_2 = custom_loss(nnOut, test_kernel_2d[0:BatchSize, :, :], BatchSize, 0, 0, 0)
##    print(obj_1/(NumberOfEigenFun*(NumberOfEigenFun-1)))
##    print(obj_2/(NumberOfEigenFun*(NumberOfEigenFun-1)))


    
