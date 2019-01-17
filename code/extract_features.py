from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from math import cos, sin
from PIL import Image
import encoding

# Initialise the constants
DATASET_ROOT = '../datasets/Fashion144k_stylenet_v1/'
FEATURES_ALL = '../features/'
MODEL_FILE = '../model/model_fashion_144k.pt'
BATCH_SIZE = 64
LABEL_SIZE = 59
LEARNING_RATE = 0.00001

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.drop1 = nn.Dropout2d(0.25)
        self.pool1 = nn.MaxPool2d(4, 4, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.drop2 = nn.Dropout2d(0.25)
        self.pool2 = nn.MaxPool2d(4, 4, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
    def forward(self, x):
        x = self.bn1(self.pool1(F.relu(self.conv1_2(F.relu(self.conv1_1(x))))))
        x = self.bn2(self.pool2(F.relu(self.conv2_2(F.relu(self.conv2_1(x))))))
        x = self.conv3_2(F.relu(self.conv3_1(x)))
        return x

class STLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(STLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.weight_fx = nn.Linear(input_size, hidden_size)
        self.weight_xi = nn.Linear(hidden_size, hidden_size)
        self.weight_hi = nn.Linear(hidden_size, hidden_size)
        self.weight_xg = nn.Linear(hidden_size, hidden_size)
        self.weight_hg = nn.Linear(hidden_size, hidden_size)
        self.weight_xo = nn.Linear(hidden_size, hidden_size)
        self.weight_ho = nn.Linear(hidden_size, hidden_size)
        self.weight_xm = nn.Linear(hidden_size, hidden_size)
        self.weight_hm = nn.Linear(hidden_size, hidden_size)
        self.weight_hz = nn.Linear(hidden_size, hidden_size)
        self.weight_zs = nn.Linear(hidden_size, output_size)
        self.weight_zm = nn.Linear(hidden_size, 6)

    def forward(self, f_k, M_k, h_k, c_k):
        x_k = F.relu(self.weight_fx(f_k))
        i_k = F.sigmoid(self.weight_xi(x_k) + self.weight_hi(h_k))
        g_k = F.sigmoid(self.weight_xg(x_k) + self.weight_hg(h_k))
        o_k = F.sigmoid(self.weight_xo(x_k) + self.weight_ho(h_k))
        m_k = F.tanh(self.weight_xm(x_k) + self.weight_hm(h_k))
        cx = g_k * c_k + i_k * m_k
        hx = o_k * cx
        z_k = F.relu(self.weight_hz(hx))
        sx = self.weight_zs(z_k)
        Mx = self.weight_zm(z_k)
        return sx, Mx, hx, cx

    def init_hidden(self, batch_size):
        M_0 = Variable(torch.FloatTensor([1, 0, 0, 0, 1, 0]).repeat(batch_size, 1)).cuda()
        h_0 = Variable(torch.zeros(batch_size, self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros(batch_size, self.hidden_size)).cuda()
        return M_0,h_0,c_0

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer = encoding.nn.Encoding(D=256,K=32)
        self.cnn = CNN()
        self.rnn = STLSTMCell(256*32, 256 , LABEL_SIZE)
        print ("init")

    # Spatial transformer network forward function
    def stn(self, f_I, M_curr):
        f_I = f_I.view(-1, 256, 24, 16)
        M_curr = M_curr.view(-1, 2, 3)
        grid = F.affine_grid(M_curr, f_I.size())
        f_curr = F.grid_sample(f_I, grid)
        f_curr = f_curr.view(-1, 256*24*16)
        return f_curr

    def scale_constraint(self,M):
        M = M.view(-1,2,3)
        scale_loss = np.fmax(M.data.cpu().numpy()[:,0,0]-0.5,0)**2 + np.fmax(M.data.cpu().numpy()[:,1,1]-0.5,0)**2
        return scale_loss

    def pos_constraint(self,M):
        M = M.view(-1,2,3)
        pos_c = np.fmax(0.1-M.data.cpu().numpy()[:,0,0],0) + np.fmax(0.1-M.data.cpu().numpy()[:,1,1],0)
        return pos_c

    def anchor_constraint(self, M_list):
        anch_c = 0
        anchor_points = []
        x0 = 0
        y0 = 0
        lst = []
        for i in range(10):
            x = x0 + 0.5 * cos(2 * 22/7.0 * i / 10)
            y = y0 + 0.5 * sin(2 * 22/7.0 * i / 10)
            anchor_points.append([x,y])
        anch_loc = 0
        for M in M_list:
            M = M.view(-1,2,3)
            anch_c += 0.5 * ((M.data.cpu().numpy()[:,0,2]-anchor_points[anch_loc][0])**2 + (M.data.cpu().numpy()[:,1,2]-anchor_points[anch_loc][1])**2)
        return anch_c

    def forward(self,x):
        # transform the input
        f_I = self.cnn(x)

        M_curr, h_curr, c_curr = self.rnn.init_hidden(BATCH_SIZE)
        f_curr = self.stn(f_I, M_curr)
        f_curr = f_curr.view(BATCH_SIZE,256,384)
        f_curr = self.layer(f_curr)
        f_curr = f_curr.view(BATCH_SIZE,32*256)
        s_curr, M_curr, h_curr, c_curr = self.rnn(f_curr, M_curr, h_curr, c_curr)
        scores = []
        M_list = []
        features = []

        for i in range(1,10):
            f_curr = self.stn(f_I, M_curr)
            f_curr = f_curr.view(BATCH_SIZE,256,384)
            f_curr = self.layer(f_curr)
            f_curr = f_curr.view(BATCH_SIZE,32*256)
            features.append(f_curr)
            s_curr, M_curr, h_curr, c_curr = self.rnn(f_curr, M_curr, h_curr, c_curr)
            scores.append(s_curr)
            M_list.append(M_curr)
        features = torch.stack(features)
        scores = torch.stack(scores)
        scores = scores.permute(1,0,2)
        scores = scores.max(dim=1)[0]
        scores = F.softmax(scores)
        anch_c = self.anchor_constraint(M_list)
        scale_c = self.scale_constraint(M_curr)
        pos_c = self.pos_constraint(M_curr)
        return scores,scale_c,pos_c,anch_c,features


def get_features():

    # Load the model
    model = Net()
    model.load_state_dict(torch.load(MODEL_FILE))
    model.cuda()
    mean = [0.5657177752729754, 0.5381838567195789, 0.4972228365504561]
    std = [0.29023818639817184, 0.2874722565279285, 0.2933830104791508]

    # Store all the features
    features_all = []
    file = open(DATASET_ROOT + 'female_online_offline_images.txt','rb')
    lines = file.readlines()

    inputs = []

    i = 1

    # Loop over all the files
    for line in lines:
        img = Image.open(DATASET_ROOT + line.split('\n')[0]).convert('RGB')
        img = img.resize((256, 384))
        img.load()
        img = np.asarray(img, dtype=np.float32)
        img /= 255.
        img = np.add(img, mean)
        img = np.divide(img, std)
        img = np.transpose(img, (2,0,1))
        inputs.append(img)

        if i%64 == 0:
            inputs_ = np.asarray(inputs, dtype=np.float32)
            scores, features = model(Variable(torch.from_numpy(inputs_).cuda()))
            features = features.cpu().data.numpy()

            for j in range(64):
                features_all.append(features[-1][j])
            inputs = []
        i += 1
    print('done')
    features_all = np.array(features_all)
    np.save(FEATURES_ALL,features_all)
get_features()
