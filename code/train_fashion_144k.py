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

##Initialise the constants
DATASET_ROOT = '../datasets/Fashion144k_stylenet_v1/'
MODEL_FILE = '../models/saved_model_final.pt'
BATCH_SIZE = 32
LABEL_SIZE = 59

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

def train():
    torch.backends.cudnn.enabled = False
    ##Initialise the model
    model = Net()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.BCEWithLogitsLoss()
    mean = [0.5657177752729754, 0.5381838567195789, 0.4972228365504561]
    std = [0.29023818639817184, 0.2874722565279285, 0.2933830104791508]

    trainids = np.load(DATASET_ROOT + 'trainids.npy')


    TRAIN_SIZE = trainids.shape[0]

    #Map ID to filenames
    id_to_name = []

    r = csv.reader(open(DATASET_ROOT + 'photos.txt'))
    for row in r:
        id_to_name.append(DATASET_ROOT + row[0])

    ##Load the labels
    labels_all = np.load(DATASET_ROOT + 'feat/singles_at_59.npy')

    trainids = np.load(DATASET_ROOT + 'trainids.npy')

    r = csv.reader(open(DATASET_ROOT + 'photos.txt'))
    id_to_name = []
    for row in r:
        id_to_name.append(DATASET_ROOT + row[0])


    # loop over the dataset multiple times
    for epoch in range(40):
        for i in range(0, TRAIN_SIZE, BATCH_SIZE):
            inputs = []
            labels = []

            for j in range(i, i+BATCH_SIZE):
                try:
                    img = Image.open(id_to_name[trainids[j]]).convert('RGB')
                except:
                    continue
                img = img.resize((256, 384))
                img.load()
                img = np.asarray(img, dtype=np.float32)
                img /= 255.
                img = np.add(img, mean)
                img = np.divide(img, std)
                img = np.transpose(img, (2,0,1))
                inputs.append(img)
                labels.append(labels_all[trainids[j]])

            inputs = np.asarray(inputs, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.float32)
            inputs, labels = Variable(torch.from_numpy(inputs).cuda()), Variable(torch.from_numpy(labels).cuda())

            if len(labels) == BATCH_SIZE:
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs, scale_c, pos_c, anch_c,features = model(inputs)

                features = features.permute(1,0,2)

                features_t = features.permute(0,2,1)

                ##Calculate divergence loss
                corr = torch.bmm(features,features_t)
                div_loss = corr.sum()/(384*32*100000)
                norm = []
                for j in labels:
                    norm.append(j.norm(1).repeat(LABEL_SIZE))
                norm = torch.stack(norm)

                ##Calculate the probabilities for each label
                p_bar = torch.FloatTensor(BATCH_SIZE,LABEL_SIZE)
                p_bar = labels/norm

                for j in range(p_bar.size()[0]):
                    for k in range(p_bar.size()[1]):
                        if(math.isnan(p_bar.data[j][k])):
                            p_bar.data[i][j] = 0.0

                ##Classification loss
                classification_loss = ((outputs - p_bar).pow(2).sum())/BATCH_SIZE

                ##Localisation loss
                localisation_loss = np.sum(scale_c)/BATCH_SIZE + 0.1*np.sum(pos_c) /BATCH_SIZE + 0.01 * np.sum(anch_c) /BATCH_SIZE

                ##Combine all losses
                loss = classification_loss + 0.1*localisation_loss + div_loss
                loss.backward()
                optimizer.step()

                ##Sort the labels
                sorted1 = Variable(torch.IntTensor(BATCH_SIZE, LABEL_SIZE).zero_())
                sorted1 = outputs.sort(1,True)
                predict = Variable(torch.IntTensor(BATCH_SIZE, LABEL_SIZE).zero_())

                for j in range(BATCH_SIZE):
                    for k in range(LABEL_SIZE):
                        a = (outputs[j][k].data[0])
                        b = sorted1[0].data[j][6]
                        if(a > b):
                            predict.data[j][k] = 1
                running_loss = loss.data[0]
                predicts = predict.data.cpu().numpy()
                true = labels.data.cpu().numpy()
                # For each class
                precision = dict()
                recall = dict()
                average_precision = dict()
                for nc in range(LABEL_SIZE):
                    precision[nc], recall[nc], _ = precision_recall_curve(true[:, nc], predicts[:, nc])
                    average_precision[nc] = average_precision_score(true[:, nc], predicts[:, nc])
                # A "micro-average": quantifying score on all classes jointly
                print ('divergence loss %3f' % div_loss.data[0])
                precision["micro"], recall["micro"], _ = precision_recall_curve(true.ravel(), predicts.ravel())
                average_precision["micro"] = average_precision_score(true, predicts, average="micro")
                print('[%d, %5d] loss: %.13f Average precision score: %13f' % (epoch + 1, i + 1, running_loss, average_precision["micro"]))
        np.random.shuffle(trainids)
        torch.save(model.state_dict(), "/media/Drive2/Staq/fashion_recommendation/models/model_fashion_144k.pt")
        print("Epoch complete")
    print('Finished Training')
train()
