import torch
import torch.nn as nn
import random
import math
import numpy as np
import cmath
import numpy as np
import torch.nn.utils as utils
import time
from collections import deque
import scipy.io as sio


np_kernel_split = np.load('kernal_16x16.npy')
kernel_split = torch.from_numpy(np_kernel_split)
TotalSampleSize = kernel_split.size(0)
print(TotalSampleSize)
rw = kernel_split.size(1)
print(rw)
col = kernel_split.size(2)
print(col)
kernel_split = kernel_split.reshape((TotalSampleSize, rw * col, 2))
kernel = torch.complex(kernel_split[:,:,0], kernel_split[:,:,1])
kernel = kernel.reshape((TotalSampleSize, rw, col)) 
kernel_split = kernel_split.reshape((TotalSampleSize, 2, rw , col ))

NumberOfEpochs = 1
BatchSize = 16
BatchSize_Sqrt = 4
NumberOfEigenFun = 10


TrainSampleSize = int((TotalSampleSize/5)*4)
TestSampleSize = int((TotalSampleSize/5)*1)

NumberOfTrainBatches = int((TrainSampleSize/BatchSize))
NumberOfTestBatches = int((TestSampleSize/BatchSize))

EigneVectorSize = rw
VmatrixLocation = NumberOfEigenFun + NumberOfEigenFun*EigneVectorSize
NNOutputPerEigenFn = 2 * rw + 1
NumberOfOutputs = NumberOfEigenFun * NNOutputPerEigenFn

SizeOfInputLayer = rw * col * 2
SizeOfHiddenLayer1 =  SizeOfInputLayer
SizeOfOutputLayer = 2 * NumberOfOutputs
SizeOfHiddenLayer2 =  (int)((SizeOfInputLayer + SizeOfOutputLayer)/2)


identity_batch = torch.eye(NumberOfEigenFun).unsqueeze(0).expand(BatchSize, -1, -1)

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
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.elu1 = nn.ELU(alpha=1.0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.elu2 = nn.ELU(alpha=1.0)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 4 * SizeOfOutputLayer)
        self.elu3 = nn.ELU(alpha=1.0)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(4 * SizeOfOutputLayer, SizeOfOutputLayer)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.elu1(x)
        x = self.conv2(x)
        x = self.elu2(x)
        x = self.max_pool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.elu3(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = NNetwork()
print(model)

##def orthogonalityCheck(matrix, batch_Size):
##    sumVal = 0
##    for i in range(batch_Size):
##        for j in range(NumberOfEigenFun):
##            for k in range(NumberOfEigenFun-(j+1)):
##                constraint = torch.vdot(matrix[i,:,j] , matrix[i,:,(k+j+1)])
##                sumVal = sumVal + torch.abs(constraint)
##    return sumVal/batch_Size


def orthogonalityCheck(matrix, batch_Size):
    constraint_matrix = torch.transpose(matrix, 1, 2).conj() @ matrix - identity_batch
    constraint_vector = constraint_matrix.reshape(batch_Size, NumberOfEigenFun*NumberOfEigenFun)
    sumVal = torch.sum(torch.norm(constraint_vector, dim=1, p=2))
    return sumVal/batch_Size

def normalize(matrix):
    magnitudes = torch.norm(matrix, dim=1, p=2)
    columns_to_normalize = (magnitudes != 1.0)
    batch_normalized = torch.where(columns_to_normalize.unsqueeze(1), matrix / magnitudes.unsqueeze(1), matrix)
    return batch_normalized

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, nnOutput, train_kernals_2d, batch_Size, penalty):
        nnOut_matrix_complex = torch.complex(nnOutput[:,0:NumberOfOutputs], nnOutput[:,NumberOfOutputs:SizeOfOutputLayer])

        diagonalMatrix = torch.diag_embed(nnOut_matrix_complex[:,0:NumberOfEigenFun])

        UVectorMatrixTmp = nnOut_matrix_complex[:, NumberOfEigenFun:VmatrixLocation]
        UVectorMatrix = UVectorMatrixTmp.view(batch_Size, EigneVectorSize, NumberOfEigenFun)
        UVectorMatrixN = normalize(UVectorMatrix)

        VVectorMatrixTmp = nnOut_matrix_complex[:, VmatrixLocation:NumberOfOutputs]
        VVectorMatrix = VVectorMatrixTmp.view(batch_Size, EigneVectorSize, NumberOfEigenFun)
        VVectorMatrixN = normalize(VVectorMatrix)
        transposedV = torch.transpose(VVectorMatrixN, 1, 2).conj()

        constraint1_abs_sum = orthogonalityCheck(VVectorMatrixN, batch_Size)
        constraint2_abs_sum = orthogonalityCheck(UVectorMatrixN, batch_Size)
        
        kernels_pred = UVectorMatrixN @ diagonalMatrix @ transposedV
 
        constraint = penalty *(constraint1_abs_sum + constraint2_abs_sum)
        mse_norm = torch.norm((train_kernals_2d - kernels_pred),p=2)/torch.norm(train_kernals_2d,p=2)
        loss = mse_norm + constraint
        return loss, mse_norm , constraint1_abs_sum, constraint2_abs_sum 

custom_loss = CustomLoss()

train_loss_train = torch.zeros((NumberOfEpochs,1))
test_avg_loss_train = torch.zeros((NumberOfEpochs,1))
train_obj1_train = torch.zeros((NumberOfEpochs,1))
test_avg_obj1_train = torch.zeros((NumberOfEpochs,1))
train_obj2_train = torch.zeros((NumberOfEpochs,1))
test_avg_obj2_train = torch.zeros((NumberOfEpochs,1))

def trainNN(train_kernel_split, train_kernel, test_kernel_split, test_kernel):
    lr = 0.000001
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

    penalty = 0.01
    
    for t in range(NumberOfEpochs):
#        print(f'Epoch {t+1}')
        model.train()
        trainlossPerEpoch = 0
        trainobj1PerEpoch = 0
        trainobj2PerEpoch = 0
        for b in range(NumberOfTrainBatches):
            nnIn = train_kernel_split[(b*BatchSize):(b+1)*BatchSize,:,:,:]
            nnOut = model(nnIn)
            loss ,mse_norm, constraint1_abs_sum, constraint2_abs_sum = custom_loss(nnOut, train_kernel[(b*BatchSize):(b+1)*BatchSize, :, :], BatchSize, penalty)
            trainlossPerEpoch = trainlossPerEpoch + loss
            trainobj1PerEpoch = trainobj1PerEpoch + constraint1_abs_sum
            trainobj2PerEpoch = trainobj2PerEpoch + constraint2_abs_sum
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        testlossPerEpoch = 0
        testobj1PerEpoch = 0
        testobj2PerEpoch = 0
        

        model.eval()
        with torch.no_grad():
            for v in range(NumberOfTestBatches):
                nnIn = test_kernel_split[(v*BatchSize):(v+1)*BatchSize,:,:,:]
                nnOut = model(nnIn)
                loss ,mse_norm, constraint1_abs_sum, constraint2_abs_sum = custom_loss(nnOut, test_kernel[(v*BatchSize):(v+1)*BatchSize, :, :], BatchSize, penalty)
                testlossPerEpoch = testlossPerEpoch + loss
                testobj1PerEpoch = testobj1PerEpoch + constraint1_abs_sum
                testobj2PerEpoch = testobj2PerEpoch + constraint2_abs_sum            
    
        train_loss_train[t] = trainlossPerEpoch/NumberOfTrainBatches
        test_avg_loss_train[t] = testlossPerEpoch/NumberOfTestBatches
        train_obj1_train[t] = trainobj1PerEpoch/NumberOfTrainBatches
        test_avg_obj1_train[t] = testobj1PerEpoch/NumberOfTestBatches
        train_obj2_train[t] = trainobj2PerEpoch/NumberOfTrainBatches
        test_avg_obj2_train[t] = testobj2PerEpoch/NumberOfTestBatches

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
            sio.savemat('SVD_N10_2.mat', data_to_save)
            torch.save(model.state_dict(), 'SVD_N10_2.pth')

                    
train_kernel_split = kernel_split[0:TrainSampleSize,:,:,:]
train_kernel_2d = kernel[0:TrainSampleSize,:,:]
test_kernel_split = kernel_split[TrainSampleSize:TotalSampleSize,:,:,:]
test_kernel_2d = kernel[TrainSampleSize:TotalSampleSize,:,:]

start_time = time.time()
trainNN( train_kernel_split, train_kernel_2d,test_kernel_split, test_kernel_2d)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time/60} minutes")





    
