import torch
import numpy as np
import argparse
import torchvision
import torch.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import glog as log
import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from tensorboardX import SummaryWriter
import time

from squeezenet import SqueezeNet
#from SqueezeNet_classic import SqueezeNet
from vgg import vgg19_bn

from knn_pebble import calcAsync
from model import Model
from load import getHCO

def test_net(args, net, test_loader):
    test_batch_iterator = iter(test_loader)
    losses = []
    for iteration in tqdm(range(int(len(test_loader)))):
        try:
            images, targets = next(test_batch_iterator)
        except RuntimeError:
            break
        with torch.no_grad():
            targets = targets.unsqueeze(-1).unsqueeze(-1)
            out = net(images)
            if args.physical == 'False':
                losses.append(torch.mean((out-targets)**2, dim = 0).detach().numpy())
            else:
                # out = torch.sinh(out)
                # targets = torch.sinh(targets)
                if args.reparametrization == 'log':
                    out = torch.exp(out) - 0.9#look up in train for exact reparametrization!
                    targets = torch.exp(targets) - 0.9
                if args.reparametrization == 'arcsinh':
                    out = torch.sinh(out)
                    targets = torch.sinh(targets)
                if args.reparametrization == 'log_arcsinh':
                    out[1:9] = torch.sinh(out[1:9])
                    out[0] = torch.sinh(out[0])
                    out[1] = torch.exp(y_train[1])
                    targets[1:9] = torch.sinh(targets[1:9])
                    targets[0] = torch.sinh(targets[0])
                    targets[1] = torch.exp(targets[1])
                # print(out.numpy().shape, targets.numpy().shape)
                losses.append(torch.mean(torch.sqrt((out-targets)**2), dim = 0).numpy())
    targets = np.squeeze(targets.numpy())
    #targets = targets.numpy()
    losses = np.asarray(losses)
    losses = np.mean(losses, axis = 0)
    losses = np.squeeze(losses)
    out = np.squeeze(out.numpy())
    if args.physical == 'False':
        print('Mean target value on each axis (last batch): \n', np.mean(targets, axis = 0))
        print('Mean out value on each axis (last batch): \n', np.mean(out, axis = 0))
        print('Mse on each axis (reparametrizated value): \n',losses/np.mean(targets, axis = 0))
        print('Total absolute mse: \n',np.mean(abs(losses)))
    else:
        print('Mean target value on each axis (last batch): \n', np.mean(targets, axis = 0))
        print('Mean out value on each axis (last batch): \n', np.mean(out, axis = 0))
        print('Mse on each axis (physical value): \n',losses)
        print('Relative error on each axis (physical value): \n',losses/np.mean(targets, axis = 0))
        print('Total absolute mse: \n',np.mean(abs(losses)))
        print('Total physical relative mse: \n',np.mean(losses/np.mean(np.abs(targets), axis = 0)))

    return np.mean(losses)

if __name__ =='__main__':
    torch.set_printoptions(edgeitems=12)
    np.set_printoptions(precision=4)
    parser = argparse.ArgumentParser(description='SqueezeNet')
    parser.add_argument('--num_workers', default=24, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--model_name', default='SqueezeNet', help='SqueezeNet, vgg19_bn')
    parser.add_argument('--physical', default=False, help='Reparametrization')
    parser.add_argument('--reparametrization', default='arcsinh', help='Reparametrization')
    parser.add_argument('--lr', default=None, help='Don\'t need here')
    parser.add_argument('--start_iter', default=300, type=int, help='iter state')
    args = parser.parse_args()

    # x_train = torch.from_numpy(np.load('/datastore/analysis/models/X_train_net.npy'))
    # y_train = torch.from_numpy(np.arcsinh(np.load('/datastore/analysis/models/y_train_net.npy')))
    # x_test = torch.from_numpy(np.load('/datastore/analysis/models/X_test_net.npy'))
    # y_test = torch.frolossesm_numpy(np.arcsinh(np.load('/datastore/analysis/models/y_test_net.npy')))

    # y0 = np.load('/datastore/analysis/models/y_hco_77_rand_0_all.npy')
    # y1 = np.load('/datastore/analysis/models/y_hco_77_rand_1_all.npy')
    y2 = np.load('/datastore/analysis/models/y_hco_77_pca_1_all.npy')
    y3 = np.load('/datastore/analysis/models/y_hco_77_pca_0_all.npy')
    y4 = np.load('/datastore/analysis/models/y_hco_77_pca_2_all.npy')
    y5 = np.load('/datastore/analysis/models/y_hco_77_pca_3_all.npy')

    # y =np.concatenate((y0, y1, y2, y3, y4, y5), axis = 1).astype(np.float32)
    # y =np.concatenate((y2, y3, y4, y5), axis = 1).astype(np.float32)
    y = np.asarray(y5, dtype = np.float32)
    if args.reparametrization == 'log':
        y = np.log(y+0.9)
        # y_test = torch.from_numpy(np.log(y_test+0.9))
    if args.reparametrization == 'arcsinh':
        y = np.arcsinh(y)
        # y_test = torch.from_numpy(np.arcsinh(y_test))
    if args.reparametrization == 'log_arcsinh':
        y[1:12] = np.arcsinh(y[1:12])
        y[0] = np.arcsinh(y[0])
        y[1] = np.log(y[1])
    y = y.transpose(1,0)
    print('original mean: \n ', np.mean(y, axis = 0))
    print('original std: \n', np.std(y, axis = 0))
    axis4 = np.mean(y[4])
    axis9 = np.mean(y[9])
    axis10 = np.mean(y[10])
    mean = np.mean(y, axis = 0)
    y -= mean
    std = np.std(y, axis = 0)
    y /=std
    y = y.transpose(1,0)

    y = np.delete(y, (4,9,10),0)
    # y[4]=np.zeros(len(y[4]))#zero out dimensions with std=0
    # y[10]=np.zeros(len(y[4]))
    # y[9]=np.zeros(len(y[4]))
    #y = y.transpose(1,0)


    #data0 = np.load('/datastore/analysis/models/data_hco_77_rand_0_all.npy')
    #data1 = np.load('/datastore/analysis/models/data_hco_77_rand_1_all.npy')
    data2 = np.load('/datastore/analysis/models/data_hco_77_pca_1_all.npy')
    data3 = np.load('/datastore/analysis/models/data_hco_77_pca_0_all.npy')
    data4 = np.load('/datastore/analysis/models/data_hco_77_pca_2_all.npy')
    data5 = np.load('/datastore/analysis/models/data_hco_77_pca_3_all.npy')
    # x = np.concatenate((data0, data1, data2, data3, data4, data5), axis = 0).astype(np.float32)
    # x = np.concatenate((data2, data3,data4, data5), axis = 0).astype(np.float32)
    x = np.asarray(data5, dtype = np.float32)

    del data2, y2,data3, y3, data4, y4, data5, y5#data0 ,y0, data1, y1#,
    y = y.transpose(1,0)

    x_train, x_test, y_train, y_test = train_test_split(x, y,  test_size=0.3, random_state=42)

    del x_train, y_train

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    # y_test = torch.from_numpy(np.log(y_test+0.9))
    if args.model_name == 'SqueezeNet':
        net = SqueezeNet()
    if args.model_name == 'vgg19_bn':
        net = vgg19_bn()

    net.load_state_dict(torch.load('/datastore/analysis/models/weights/'+args.model_name + repr(args.start_iter)+args.reparametrization + '.pth'))
    #print('x_test: ', x_test.shape,' y_test: ', y_test.shape)
    #log.info('Network model:')
    #log.info(net)
    test_dataset = data.TensorDataset(x_test,y_test)
    test_loader = data.DataLoader(test_dataset, num_workers = args.num_workers, batch_size = args.batch_size,shuffle = True)
    net.eval()
    #mse = test_net(args, net, test_loader)

    target = '77'
    line = 'hco'
    isotope = '12'
    cube, v, pos = getHCO(target)
    hco = Model(line)
    hco.loadConsts(target,line,isotope)
    hco.setCube(cube,v,pos)
    out = net(torch.from_numpy(np.asarray(hco.cube, dtype = np.float32)).unsqueeze(0)).detach().numpy()
    out = out.squeeze()
    out = np.insert(out, 4, axis4)#debug it
    out = np.insert(out, 9, axis9)
    out = np.insert(out, 10, axis10)
    out*=std
    out+=mean
    names = ['DENS0s','X0s','RDENSs','RMAXs','R0s','VTURB0s','vsys0s','rvsyss','rvs','posX','posY']

    print('Prediction on cube: \n', out)
    if args.reparametrization == 'log':
        out = np.exp(out)-0.9
    if args.reparametrization == 'arcsinh':
        out = np.sinh(out)
    dictionary = dict(zip(names, out))
    print('Prediction on cube of physical parameters: \n', dictionary)
    print('original mean: \n ', dict(zip(names, mean)))

    par = [out,'name']
    # if os.path.exists('./name'):
    #     os.remove('./name')
    # out_out = calcAsync(par).squeeze()
    # print('Model prediction on net prediction: \n', out_out.shape)
    # print(cube.shape, out_out.shape)#shapes must be equal but they are not
    # print('Error: \n', ((cube-out_out)**2).sum(axis=(1,2,3)))
