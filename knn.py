import math

def distance(x1, x2):
    t = 0
    t += (x1 - float(x2)) ** 2
    return t ** 0.5

def readData(fp):
    with open(fp, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.split(',')
        line = [float(x) for x in line]
    return lines

def getDistances(p, x):
    dists = list()
    for item in x:
        dists.append(distance(p, item))
    return dists

def kNN(p, k, x, y):    
    dists = getDistances(p, x)
    temp = zip(dists, y)
    temp = sorted(temp)
    res = 0
    for i in range(0, k):
        res += float(temp[i][1])
    return res / k

def score(x, y, k, xval, yval):
    pred = list()
    for item in xval:
        pred.append(kNN(float(item), k, x, y))
    res = 0
    for i in range(0, len(yval)):
        res += distance(float(yval[i]), pred[i])
    return res

x = readData('data/xdata.txt')
xval = readData('data/xdata_val.txt')
y = readData('data/ydata.txt')
yval = readData('data/ydata_val.txt')

k = 6

mse = score(x, y, k, xval, yval)
print("k = {}\nValidation MSE: {}".format(k, mse))