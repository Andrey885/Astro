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


from test import test_net

torch.set_printoptions(edgeitems=12,threshold=12)
parser = argparse.ArgumentParser(description='SqueezeNet')
parser.add_argument('--num_workers', default=24, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--lr_drops', nargs='+', type=int, default=[100, 250, 350, 450], help='LR step sizes')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for SGD')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--model_name', default='vgg19_bn', help='SqueezeNet, vgg19_bn')

parser.add_argument('--start_iter', default=0, type=int, help='Resume training at this iter')
parser.add_argument('--val_step', type=int, default=150, help='Evaluate model each val_step.')
parser.add_argument('--max_iter', default=800, type=int, help='Max steps to train')
parser.add_argument('--physical', default=True, help='Reverse test reparametrization')
parser.add_argument('--reparametrization', default='log_arcsinh', help='Reparametrization')

args = parser.parse_args()

torch.set_num_threads(args.num_workers)
np.set_printoptions(edgeitems = 12)
# x_train = torch.from_numpy(np.load('/datastore/analysis/models/X_train_net.npy'))
# y_train = torch.from_numpy(np.arcsinh(np.load('/datastore/analysis/models/y_train_net.npy')))
# x_test = torch.from_numpy(np.load('/datastore/analysis/models/X_test_net.npy'))
# y_test = torch.from_numpy(np.arcsinh(np.load('/datastore/analysis/models/y_test_net.npy')))

y0 = np.load('/datastore/analysis/models/y_hco_77_rand_0_all.npy')
# y1 = np.load('/datastore/analysis/models/y_hco_77_rand_1_all.npy')
y2 = np.load('/datastore/analysis/models/y_hco_77_pca_1_all.npy')
y3 = np.load('/datastore/analysis/models/y_hco_77_pca_0_all.npy')
y4 = np.load('/datastore/analysis/models/y_hco_77_pca_2_all.npy')
y5 = np.load('/datastore/analysis/models/y_hco_77_pca_3_all.npy')

# y =np.concatenate((y0, y1, y2, y3, y4, y5), axis = 1).astype(np.float32)
#y =npstd.concatenate((y2, y3, y4), axis = 1).astype(np.float32)


y = np.asarray(y5, dtype = np.float32)

# print(np.mean(y, axis = 1))
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
    # y_test[0,2:11] = np.arcsinh(y_test[0,2:11])
    # y_test[1] = torch.from_numpy(np.arcsinh(y_test[1]))
    # print(np.mean(y, axis = 1))

    # y_test = torch.from_numpy(y_test)
print(y.shape)
#print(np.mean(y, axis = 1))
y = y.transpose(1,0)
print('original mean: \n ', np.mean(y, axis = 0))
print('original std: \n', np.std(y, axis = 0))

y -= np.mean(y, axis = 0)
y /=np.std(y, axis = 0)
y = y.transpose(1,0)
y = np.delete(y, (4,9,10),0)
# y[4]=np.zeros(len(y[4]))#zero out dimensions with std=0
# y[10]=np.zeros(len(y[4]))
# y[9]=np.zeros(len(y[4]))
y = y.transpose(1,0)
print('set mean: \n', np.mean(y, axis = 0))
# print(y)
#data0 = np.load('/datastore/analysis/models/data_hco_77_rand_0_all.npy')
#data1 = np.load('/datastore/analysis/models/data_hco_77_rand_1_all.npy')
data2 = np.load('/datastore/analysis/models/data_hco_77_pca_1_all.npy')
data3 = np.load('/datastore/analysis/models/data_hco_77_pca_0_all.npy')
data4 = np.load('/datastore/analysis/models/data_hco_77_pca_2_all.npy')
data5 = np.load('/datastore/analysis/models/data_hco_77_pca_3_all.npy')

# x = np.concatenate((data0, data1, data2, data3, data4, data5), axis = 0).astype(np.float32)
#x = np.concatenate((data2, data3,data4), axis = 0).astype(np.float32)
x = np.asarray(data5, dtype = np.float32)
del data2, y2,data3, y3, data4, y4, data5, y5#data0 ,y0, data1, y1#,

x_train, x_test, y_train, y_test = train_test_split(x, y,  test_size=0.3, random_state=42)
x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
if args.model_name == 'SqueezeNet':
    from squeezenet import SqueezeNet
    net  = SqueezeNet()
if args.model_name == 'vgg19_bn':
    from vgg import vgg19_bn
    net = vgg19_bn()
if args.model_name == 'SqueezeNet_classic':
    from SqueezeNet_classic import SqueezeNet
    net = SqueezeNet()
if args.model_name == 'some_net':
    from some_net import Net
    net = Net()

net.train()

#net = torch.nn.DataParallel(ssd_net, device_ids=args.gpus)
if args.start_iter != 0:
    net.load_state_dict(torch.load('weights/'+args.model_name + repr(args.start_iter) + '.pth'))
print('x_train', x_train.shape, 'y_train', y_train.shape, 'x_test', x_test.shape,' y_test', y_test.shape)
log.info('Network model:')
log.info(net)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
train_dataset = data.TensorDataset(x_train,y_train)
test_dataset = data.TensorDataset(x_test,y_test)

train_loader = data.DataLoader(train_dataset, num_workers = args.num_workers, batch_size = args.batch_size,
                                 shuffle = True)
test_loader = data.DataLoader(test_dataset, num_workers = args.num_workers, batch_size = args.batch_size,
                                  shuffle = True)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drops)

batch_iterator = iter(train_loader)
test_batch_iterator = iter(test_loader)
#print(images.shape, targets.shape)

log_path = './logs/{:%Y_%m_%d_%H_%M}'.format(datetime.datetime.now())
writer = SummaryWriter(log_path)

mse_loss = nn.MSELoss()
for iteration in range(args.start_iter, args.max_iter):
    scheduler.step()
    try:
        images, targets = next(batch_iterator)
    except StopIteration:
        batch_iterator = iter(train_loader)
        images, targets = next(batch_iterator)

    with torch.no_grad():
        images = Variable(images)
        targets = targets.unsqueeze(-1).unsqueeze(-1)
        targets = Variable(targets)

    out = net(images)
    optimizer.zero_grad()
    #print(out, targets)
    loss = mse_loss(out, targets)
    loss.backward()
    #print(loss.item())
    optimizer.step()
#net = torch.nn.DataParallel(ssd_net, device_ids=args.gpus)

    if iteration % 10 == 0:
        log.info('Iter ' + repr(iteration) + ' || LR: %.6f || Loss: %.4f'
                 % (scheduler.get_lr()[0], loss.item()))

        writer.add_scalar('Losses/Total_loss', loss.item(), iteration)
        writer.add_scalar('LR/Learning_rate', scheduler.get_lr()[0], iteration)

    if iteration % args.val_step == 0 and iteration != args.start_iter:
        log.info('Saving state, iter: {}'.format(iteration))
        net.eval()

        mse = test_net(args, net, test_loader)
        net.train()
        log.info('mse: {0:.4f}'.format(float(mse)))
        writer.add_scalar('MSE', mse, iteration)
        torch.save(net.state_dict(), '/datastore/analysis/models/weights/'+args.model_name +
                   repr(iteration)+args.reparametrization + '.pth')
torch.save(net.state_dict(),
          '/datastore/analysis/models/weights/'+ args.model_name + 'Final.pth')


# print(x.shape, y.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# np.save('/datastore/analysis/models/X_train_net.npy',X_train)
# np.save('/datastore/analysis/models/X_test_net.npy',X_test)
# np.save('/datastore/analysis/models/y_train_net.npy',y_train)
# np.save('/datastore/analysis/models/y_test_net.npy',y_test)
# exit()
