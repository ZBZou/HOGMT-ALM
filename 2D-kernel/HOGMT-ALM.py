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
kernel_split = kernel_split.reshape((TotalSampleSize, rw * col * 2))

slope = 0.1
NumberOfEpochs = 500
BatchSize = 16
BatchSize_Sqrt = 4
NumberOfEigenFun = 10


TrainSampleSize = int((TotalSampleSize/5)*4)
TestSampleSize = int((TotalSampleSize/5)*1)

NumberOfTrainBatches = int((TrainSampleSize/BatchSize))
NumberOfTestBatches = int((TestSampleSize/BatchSize))

EigneVectorSize = rw
VmatrixLocation = NumberOfEigenFun + NumberOfEigenFun*EigneVectorSize
NNOutputPerEigenFn = 2* rw + 1
NumberOfOutputs = NumberOfEigenFun * NNOutputPerEigenFn

SizeOfInputLayer = rw * col * 2
SizeOfHiddenLayer1 =  SizeOfInputLayer
SizeOfOutputLayer = 2 * NumberOfOutputs
SizeOfHiddenLayer2 =  (int)((SizeOfInputLayer + SizeOfOutputLayer)/2)

diagonal_mask = torch.eye(NumberOfEigenFun, dtype=torch.bool).unsqueeze(0).expand(BatchSize, -1, -1)

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
    constraint_matrix = torch.transpose(matrix, 1, 2).conj() @ matrix
    constraint_matrix[diagonal_mask] = 0
    constraint_vector = constraint_matrix.reshape(batch_Size, NumberOfEigenFun*NumberOfEigenFun)
    sumVal = torch.sum(torch.norm(constraint_vector, dim=1, p=2))
    constraint_vector = torch.stack((constraint_vector.real, constraint_vector.imag), dim=1)
    constraint_vector = constraint_vector.reshape(batch_Size, 2*NumberOfEigenFun*NumberOfEigenFun) 
    return sumVal/batch_Size, constraint_vector

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, nnOutput, train_kernals_2d, batch_Size, batchSize_Sqrt, penalty, lambda_1, lambda_2):
        nnOut_matrix_complex = torch.complex(nnOutput[:,0:NumberOfOutputs], nnOutput[:,NumberOfOutputs:SizeOfOutputLayer])

        diagonalMatrix = torch.diag_embed(nnOut_matrix_complex[:,0:NumberOfEigenFun])

        UVectorMatrixTmp = nnOut_matrix_complex[:, NumberOfEigenFun:VmatrixLocation]
        UVectorMatrix = UVectorMatrixTmp.view(batch_Size, EigneVectorSize, NumberOfEigenFun)

        VVectorMatrixTmp = nnOut_matrix_complex[:, VmatrixLocation:NumberOfOutputs]
        VVectorMatrix = VVectorMatrixTmp.view(batch_Size, EigneVectorSize, NumberOfEigenFun)
        transposedV = torch.transpose(VVectorMatrix, 1, 2).conj()
        
        constraint1_abs_sum, constraint_vector1 = orthogonalityCheck(VVectorMatrix, batch_Size)
        constraint_vector1_avg = torch.mean(constraint_vector1, dim=0)
        constraint2_abs_sum, constraint_vector2 = orthogonalityCheck(UVectorMatrix, batch_Size)
        constraint_vector2_avg = torch.mean(constraint_vector2, dim=0)

        constraint_vector1_avg = constraint_vector1_avg.reshape( 2*NumberOfEigenFun*NumberOfEigenFun,1)
        constraint_vector2_avg = constraint_vector2_avg.reshape( 2*NumberOfEigenFun*NumberOfEigenFun,1)
        
        kernels_pred = UVectorMatrix @ diagonalMatrix @ transposedV
        
 
        constraintP1 = torch.sum(lambda_1 * constraint_vector1_avg) + torch.sum(lambda_2 * constraint_vector2_avg)
        constraints_L2_norm = (torch.norm(constraint_vector1,p=2) + torch.norm(constraint_vector2,p=2))/ batchSize_Sqrt
        constraintP2 = (penalty/(2*batch_Size)) * (torch.pow(torch.norm(constraint_vector1,p=2),2) + torch.pow(torch.norm(constraint_vector2,p=2),2))
        constraint =  constraintP1 + constraintP2
        mse_norm = torch.pow(torch.norm((train_kernals_2d - kernels_pred),p=2),2)/torch.pow(torch.norm(train_kernals_2d,p=2),2)
        loss = mse_norm + constraint
        return loss , constraint1_abs_sum, constraint2_abs_sum, constraint_vector1_avg, constraint_vector2_avg, constraints_L2_norm

custom_loss = CustomLoss()

train_loss_train = torch.zeros((NumberOfEpochs,1))
test_avg_loss_train = torch.zeros((NumberOfEpochs,1))
train_obj1_train = torch.zeros((NumberOfEpochs,1))
test_avg_obj1_train = torch.zeros((NumberOfEpochs,1))
train_obj2_train = torch.zeros((NumberOfEpochs,1))
test_avg_obj2_train = torch.zeros((NumberOfEpochs,1))
penalty_train = torch.zeros((NumberOfEpochs,1))
lambda1_train = torch.zeros((NumberOfEpochs,1))
lambda2_train = torch.zeros((NumberOfEpochs,1))

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

    lambda1V = torch.full(( 2*NumberOfEigenFun*NumberOfEigenFun,1), 0.0)
    lambda2V = torch.full(( 2*NumberOfEigenFun*NumberOfEigenFun,1), 0.0)
    penalty = 1
    alpha = 1.01

    constraint_vector1_avg = torch.zeros( 2*NumberOfEigenFun*NumberOfEigenFun,1, dtype=torch.float)
    constraint_vector2_avg = torch.zeros( 2*NumberOfEigenFun*NumberOfEigenFun,1, dtype=torch.float)
    constraints_L2_norm = 0
    constraints_L2_norm_prev = 0
    
    for t in range(NumberOfEpochs):
        model.train()
        trainlossPerEpoch = 0
        trainobj1PerEpoch = 0
        trainobj2PerEpoch = 0
        for b in range(NumberOfTrainBatches):
            nnOut = model(train_kernel_split[(b*BatchSize):(b+1)*BatchSize,:])
            loss , constraint1_abs_sum, constraint2_abs_sum, constraint_vector1_avg, constraint_vector2_avg, constraints_L2_norm = custom_loss(nnOut, train_kernel[(b*BatchSize):(b+1)*BatchSize, :, :], BatchSize, BatchSize_Sqrt, penalty, lambda1V, lambda2V)
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
                nnOut = model(test_kernel_split[(v*BatchSize):(v+1)*BatchSize,:])
                loss , constraint1_abs_sum, constraint2_abs_sum, constraint_vector1_avg_test, constraint_vector2_avg_test, constraints_L2_norm_test = custom_loss(nnOut, test_kernel[(v*BatchSize):(v+1)*BatchSize, :, :], BatchSize, BatchSize_Sqrt, penalty, lambda1V, lambda2V)
                testlossPerEpoch = testlossPerEpoch + loss
                testobj1PerEpoch = testobj1PerEpoch + constraint1_abs_sum
                testobj2PerEpoch = testobj2PerEpoch + constraint2_abs_sum            
    
        train_loss_train[t] = trainlossPerEpoch/NumberOfTrainBatches
        test_avg_loss_train[t] = testlossPerEpoch/NumberOfTestBatches
        train_obj1_train[t] = trainobj1PerEpoch/NumberOfTrainBatches
        test_avg_obj1_train[t] = testobj1PerEpoch/NumberOfTestBatches
        train_obj2_train[t] = trainobj2PerEpoch/NumberOfTrainBatches
        test_avg_obj2_train[t] = testobj2PerEpoch/NumberOfTestBatches
        penalty_train[t] = penalty
        lambda1_train[t] = torch.norm(lambda1V.float(),p=2)
        lambda2_train[t] = torch.norm(lambda2V.float(),p=2)

        with torch.no_grad():
            lambda1V = lambda1V + penalty *  constraint_vector1_avg
            lambda2V = lambda2V + penalty *  constraint_vector2_avg
            if (constraints_L2_norm > 0.75 * constraints_L2_norm_prev):
                penalty = alpha * penalty
            constraints_L2_norm_prev = constraints_L2_norm

        if (t+1)%25 == 0:
            # Prepare data to save
            print(t+1)

            train_loss_train_tmp = train_loss_train.numpy()
            test_avg_loss_train_tmp = test_avg_loss_train.numpy()
            train_obj1_train_tmp = train_obj1_train.numpy()
            test_avg_obj1_train_tmp = test_avg_obj1_train.numpy()
            train_obj2_train_tmp = train_obj2_train.numpy()
            test_avg_obj2_train_tmp = test_avg_obj2_train.numpy()
            penalty_train_tmp = penalty_train.numpy()
            lambda_1 = lambda1_train.numpy()
            lambda_2 = lambda2_train.numpy()

            data_to_save = {
                'trained_epochs' : t,
                'train_loss': train_loss_train_tmp,
                'test_loss': test_avg_loss_train_tmp,
                'o1_train' : train_obj1_train_tmp,
                'o1_test' : test_avg_obj1_train_tmp,
                'o2_train' : train_obj2_train_tmp,
                'o2_test' : test_avg_obj2_train_tmp,
                'penalty' : penalty_train_tmp,
                'lamda1' : lambda_1,
                'lamda2' : lambda_2,
                
            }

            # Save the data to a .mat file
            sio.savemat('N10_alm.mat', data_to_save)
            torch.save(model.state_dict(), 'N10_alm.pth')
                    
train_kernel_split = kernel_split[0:TrainSampleSize,:]
train_kernel_2d = kernel[0:TrainSampleSize,:,:]
test_kernel_split = kernel_split[TrainSampleSize:TotalSampleSize,:]
test_kernel_2d = kernel[TrainSampleSize:TotalSampleSize,:,:]

start_time = time.time()
trainNN( train_kernel_split, train_kernel_2d,test_kernel_split, test_kernel_2d)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time/60} minutes")
