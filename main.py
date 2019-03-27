import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from model import Model
from load import getHCO
from sklearn.model_selection import train_test_split
#from mlHCO.fitting import calculateErrors
y0 = np.load('analysis/y_hco_77_rand_0_all.npy')
data0 = np.load('analysis/data_hco_77_rand_0_all.npy')
# print('y0',y0.shape,'data0', data0.shape)

target = '77'
line = 'hco'
isotope = '12'
cube, v, pos = getHCO(target)
hco = Model(line)
hco.loadConsts(target,line,isotope)
hco.setCube(cube,v,pos)
print('cube', hco.cube.shape, 'data', data0.shape)
chis = ((data0-hco.cube)**2)
chis = chis.sum(axis=(1,2,3))
#np.set_printoptions(edgeitems  = 100)
# k = np.argmin(chis)
y0 = y0.transpose(1,0)
print(y0.shape)
# #print(hco.cube, data0[k])
# print(np.min(chis)/np.mean(data0[k]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = linear_model.Lasso(alpha=0.1)
clf.fit(y0, chis)
#научиться считать ошибку
