import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#мощность двигателя 69 + 23 = 92
#крутящий момент
#объем двигателя
rand = np.random.RandomState(0)
data = [[310, 2000, 10],[320, 2000, 10.4],[300, 1900, 9],[250, 1800, 10],[330, 2100, 10],[340, 2200, 10],[350, 2000, 10],[360, 2100, 10],
                 [370, 2300, 11],[380, 2400, 11],[390, 2000, 11],[400, 2000, 11],[410, 2100, 10],[420, 2100, 10],[430, 2300, 10],[440,2300,9],
                 [290,1800,9],[280,1700,9],[270,1800, 10],[300, 1700, 10],


                 [310, 2000, 10],[320, 2000, 10.4],[300, 1900, 9],[250, 1800, 10],[330, 2100, 10],[340, 2200, 10],
                 [350, 2000, 10],[360, 2100, 10],
                 [370, 2300, 11],[380, 2400, 11],[390, 2000, 11],[400, 2000, 11],[410, 2100, 10],[420, 2100, 10],
                 [430, 2300, 10],[440, 2300, 9],
                 [290, 1800, 9],[280, 1700, 9],[270, 1800, 10],[300, 1700, 10],

                 [200,1000,6],[210,900,7],[190,800,6],[180,700,6],[170,600,7],[200,1100,7],[160,600,6],[195,850,7],
                 [210,800,7],[220,900,7],[230,1000,7],[240,1100,6],[200,950,7],[195,930,5],[215,1050,7],[225,1150,7],
                 [260,500,7],[250,600,6],[245,800,6],[215,950,7],

                 [150,400,4],[160,400,5],[140,350,5],[155,450,4],[160,500,4],[170,400,4],[165,355,5],[145,335,5],
                 [135,370,5],[130,330,5],[150,375,4],[155,345,5],[165,365,5],[170,320,4],[154,315,4],[160,300,5],
                 [135,325,5],[125,370,5],[143,330,4],[160,360,5],

                 [130,280,3],[140,250,3],[130,250,3],[120,275,3],[110,230,3],[120,240,3],[130,260,3],[100,220,3],
                 [90,210,2],[140,220,3],[150,245,3],[160,210,3],[125,200,3],[135,225,3],[155,215,2],[165,200,3],
                 [125,190,2],[115,175,2],[134,250,3],[145,245,3]
                 ]

tData = []
num = []


for i in data:
    count = 0
    for j in range(3):
        count += i[j]
    num.append(count / 3)


#663 300 255 165 100

for i in num:
    if i < 100:
        tData.append([0, 0, 255])
        #tData.append([0,0,random.randint(0,255)])
    elif i > 100 and i < 150:
        tData.append([0, 255, 100])
    elif i > 150 and i < 200:
        tData.append([255, 100, 0])
    elif i > 200 and i < 250:
        tData.append([255, 100, 0])
    elif i > 250 and i < 300:
        tData.append([255, 0, 50])
    elif i > 300 and i < 350:
        tData.append([50, 80, 255])
    elif i > 350 and i < 400:
        tData.append([0, 135, 255])
    elif i > 400 and i < 450:
        tData.append([30, 100, 255])
    elif i > 450 and i < 500:
        tData.append([120, 40, 255])
    elif i > 500 and i < 550:
        tData.append([200, 155, 0])
    elif i > 550 and i < 600:
        tData.append([200, 200, 155])
    elif i > 600 and i < 650:
        tData.append([255, 100, 100])
    elif i > 650 and i < 700:
        tData.append([255, 50, 80])
    elif i > 700 and i < 750:
        tData.append([255, 70, 255])
    elif i > 750 and i < 800:
        tData.append([0, 80, 100])
    elif i > 800 and i < 850:
        tData.append([20, 255, 30])
    elif i > 900 and i < 950:
        tData.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
    else:
        tData.append([0,0,0])
        #tData.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])





trainData = np.array(tData)


def find_BMU(SOM, x):
    distSq = (np.square(SOM - x)).sum(axis=2)
    return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)

def update_weights(SOM, train_ex, learn_rate, radius_sq, BMU_coord, step=3):
    g,h = BMU_coord
    if radius_sq < 1e-3:
        SOM[g,h,:] += learn_rate * (train_ex * SOM[g,h,:])
        return SOM
    for i in range(max(0, g-step), min(SOM.shape[0], g+step)):
        for j in range(max(0, h-step), min(SOM.shape[1], h+step)):
            distSq = np.square(i - g) + np.square(j - h)
            distFunc = np.exp(-distSq / 2 / radius_sq)
            SOM[i,j,:] += learn_rate * distFunc * (train_ex - SOM[i,j,:])
    return SOM


def train_SOM(SOM, trainData, learn_rate = .1, radiusSq = 1, lr_decay = .1, radiusDecay = .1, epochs = 10):
    learn_rate_0 = learn_rate
    radius_0 = radiusSq
    for epoch in np.arange(0, epochs):
        random.shuffle(trainData)
        for train_ex in trainData:
            g,h = find_BMU(SOM, train_ex)
            SOM = update_weights(SOM, train_ex, learn_rate, radiusSq, (g,h))

        learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
        radiusSq = radius_0 * np.exp(-epoch * radiusDecay)
    return SOM

#-----------------



m = 6
n = 6
n_x = 3000

#trainData = rand.randint(0,255,(n_x,3))
#print(trainData)

SOM = rand.randint(0,255,(m,n,3)).astype(float)
print(SOM)

fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(12,3.5),subplot_kw=dict(xticks=[], yticks=[]))

ax[0].imshow(trainData.reshape(10, 10, 3))
ax[0].title.set_text('Training Data')
ax[1].imshow(SOM.astype(int))
ax[1].title.set_text('Randomly Initialized SOM Grid')

fig, ax = plt.subplots(
    nrows=1, ncols=4, figsize=(15, 3.5),
    subplot_kw=dict(xticks=[], yticks=[]))
total_epochs = 0

for epochs, i in zip([1, 4, 5, 10], range(0,4)):
    total_epochs += epochs
    SOM = train_SOM(SOM, trainData, epochs=epochs)
    print(SOM)
    ax[i].imshow(SOM.astype(int))
    ax[i].title.set_text('Epochs = ' + str(total_epochs))


plt.show()