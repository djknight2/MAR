# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 00:33:47 2019

@author: dtwiz
"""

import requests
import pprint as pp
import sys
import pandas
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
import numpy as np
import os

done = False
first = True
while not done:

    seed = 4#normally 4
    np.random.seed(seed)


    for p in sys.path:
        print(p)

    length = 1296#NORMALLY 1296
    reservePeriod = 56
    predictionPeriod = 21
    mode = 1
    ribbonDays = 200
    priceDays = 200
    stop = False
    asset = 'BTC'
    quote = 'USD'
    if first:
        numEpochs = 2000
    modelName = asset + quote + '10-14-19!' + str(length) + '-' + str(reservePeriod) + '-' + str(predictionPeriod) + '-' + str(mode)
    url = "https://api.kraken.com/0/public/OHLC"
    # if len(sys.argv) > 1:
    #     first = False
    #     print(sys.argv[1])
    #     numEpochs = int(sys.argv[1])


    url = "https://min-api.cryptocompare.com/data/histoday?fsym=" + asset +  "&tsym=" + quote + "&limit=" + str(length)
    resp = requests.get(url)
    resp = resp.json()
    resp = resp['Data']
    resp = list(resp)
    fib = {
        0: 1,
        1: 2,
        2: 3,
        3: 5,
        4: 8,
        5: 13,
        6: 21,
        7: 34,
        8: 55,
        9: 89,
        10: 144,
        11: 233}
    # pp.pprint(resp)
    closes = []
    highs = []
    lows = []
    for day in range(len(resp)-1):  # makes closes into an array of only the close prices, most recent last
            closes.append(resp[day]['close'])
            highs.append(resp[day]['high'])
            lows.append(resp[day]['low'])
    resp.pop()
    np.save('closes',closes)
    

    activeDays = {}  # filled with closes of the last 200 days close prices. 'circular' dictionary. starts at 0 - 199
    maRibbon = {}  # filled with 0-199 day moving averages for each day
    oneDayGains = []  # difference between close of previous day and current day bool
    # TRY TO DO A RIBBON OF VALUES WITH PERCENT DIFFERENCE OF PRICE TO EACH MA
    for day in range(length - 200):
        maRibbon[day] = {}
    dayMAs = {}  # filled with 0-199 day moving averages for a single day

    for day in range(200):
        activeDays[day] = closes[day]



    X = []
    Y = []

    def defYRange(days):#how many days forward
        dayRanges = []#ultimately 2d list with [day][high/low]0 for low and 1 for high
        for date in range(len(highs)-(days+reservePeriod)):
            rng = []
            tempHigh = highs[date+days]
            tempLow = lows[date+days]
            for day in range(predictivePeriod):
                if highs[date+days+day] > tempHigh:
                    tempHigh = highs[date+days+day]
                if lows[date+days+day] < tempLow:
                    tempLow = lows[date+days+day]
            rng.append(tempLow)
            rng.append(tempHigh)
            dayRanges.append(rng)
        return dayRanges

    def defXMAR(days):#days in ribbon
        maRibbon = {}
        activeDays = {}
        for day in range(length - days):
            maRibbon[day] = {}
        dayMAs = {}  # filled with 0-199 day moving averages for a single day

        for day in range(days):
            activeDays[day] = closes[day]
        index = 0  # index in active days to be replaced
        for date in range(len(closes)-days):  # for each day in the data
            sum = 0
            activeDays[index] = closes[date + (days-1)]
            print(closes[date+days])
            for day in range(days):  # to calculate the ma for each day interval
                key = day
                if (index - key) < 0:
                    key = index - key + days
                else:
                    key = index - key

                sum += activeDays[key]
                # print(sum)
                maRibbon[date][day] = (sum / (day + 1))# - activeDays[index]
                sum += 1
            index += 1
            if index > days-1:
                index = 0
        return maRibbon
    
    def defYRibbon(days):
        ribbons = []
        for date in range(len(highs)-(days+reservePeriod)):
            ribbon = []
            for x in range(18):
                ribbon.append(0)
            price = closes[date + days]
            tempHigh = highs[date+days]
            tempLow = lows[date+days]
            for day in range(predictionPeriod):
                if highs[date+days+day] > tempHigh:
                    tempHigh = highs[date+days+day]
                if lows[date+days+day] < tempLow:
                    tempLow = lows[date+days+day]
            unit = .025
            if tempLow < price:
                if tempLow > price - price*unit:#-0-2.5%   
                    ribbon[8] = 1
                elif tempLow > price - price*unit*2:#-2.5-5%
                    ribbon[7] = 1
                elif tempLow > price - price*unit*3:#...
                    ribbon[6] = 1
                elif tempLow > price - price*unit*4:#...
                    ribbon[5] = 1
                elif tempLow > price - price*unit*5:
                    ribbon[4] = 1
                elif tempLow > price - price*unit*6:
                    ribbon[3] = 1
                elif tempLow > price - price*unit*7:
                    ribbon[2] = 1
                elif tempLow > price - price*unit*8:#-17.5-20%
                    ribbon[1] = 1
                else:# < -20% (-20% or less)
                    ribbon[0] = 1
            if tempHigh > price:
                if tempHigh < price + price*unit:#+0-2.5%   
                    ribbon[9] = 1
                elif tempHigh < price + price*unit*2:#+2.5-5%
                    ribbon[10] = 1
                elif tempHigh < price + price*unit*3:#...
                    ribbon[11] = 1
                elif tempHigh < price + price*unit*4:#...
                    ribbon[12] = 1
                elif tempHigh < price + price*unit*5:
                    ribbon[13] = 1
                elif tempHigh < price + price*unit*6:
                    ribbon[14] = 1
                elif tempHigh < price + price*unit*7:
                    ribbon[15] = 1
                elif tempHigh < price + price*unit*8:#+17.5-20%
                    ribbon[16] = 1
                else:# > +20% (+20% or greater)
                    ribbon[17] = 1
                ribbons.append(ribbon)
        return ribbons

    def defYHigher(days):
        oneDayGains = []
        for date in range(len(closes)-(days+reservePeriod)):
            oneDayGains.append(1 if closes[date + days + predictionPeriod] >= (closes[date + days]) else 0)
        return oneDayGains


    def defXMARnCloses(daysMAR,daysClose):
        return []

    if mode == 0:
        X = defXMAR(ribbonDays)
        Y = defYHigher(ribbonDays)
    elif mode == 1:
        X = defXMAR(ribbonDays)
        Y = defYRibbon(ribbonDays)
        

    dict = {}
    #Scale for cases you want to test must be fixed here

    # #will be defXMAR
    # index = 0  # index in active days to be replaced
    # for date in range(len(closes)-200):  # for each day in the data
    #     sum = 0
    #     activeDays[index] = closes[date + 199]
    #     print(closes[date+200])
    #     if date < len(closes)-(200 + reservePeriod):
    #         oneDayGains.append(1 if closes[date + 199 + predictionPeriod] >= (closes[date + 199]) else 0)
    #     for day in range(200):  # to calculate the ma for each day interval
    #         key = day
    #         if (index - key) < 0:
    #             key = index - key + 200
    #         else:
    #             key = index - key
    #
    #         sum += activeDays[key]
    #         # print(sum)
    #         maRibbon[date][day] = (sum / (day + 1))# - activeDays[index]
    #         sum += 1
    #     index += 1
    #     if index > 199:
    #         index = 0
    #     # pp.pprint(maRibbon)

    maRibbon = X
    oneDayGains = Y

    maRibbonArray = []

    dict = {}

    for date in range(len(closes)-200):  # for each date
        mas = []
        for day in range(200):  # for each day timeframe
            mas.append(0)
        for day in range(200):
            mas[day] = maRibbon[date][day]
        if date < len(closes)-(200+reservePeriod):
            dict[str(mas[0])+str(mas[100])+str(mas[199])] = date
        maRibbon[date] = mas
        maRibbonArray.append(mas)

    # pp.pprint(maRibbon)

    # for date in range(len(closes)-(200+predictionPeriod+reservePeriod)):
    #     dict[str(maRibbonArray[date][0])+str(maRibbonArray[date][100]) + str(maRibbonArray[date][199])] = date

    maRibbonArray = np.array(maRibbonArray)
    oneDayGains = np.array(oneDayGains)

    # pp.pprint(maRibbonArray)

    # print(xTrain)
    # print(yTrain)
    # print(xTest)
    # print(yTest)


    optim = SGD(lr=.1)
    activation = 'relu'

    data = []

    # for date in range(len(maRibbonArray)):  # adds answers to each date
    #     tempData = maRibbonArray[date]
    #     np.append(tempData, oneDayGains[date])
    #     data.append(tempData)
    #
    # data = np.array(data)
    #
    # x, y = data[:, :-1], data[:, -1]
    #
    # split = int(4 * len(maRibbonArray) / 5)
    #
    # train = data[:split, :]
    # test = data[split:, :]
    #
    # ribbonTrain = maRibbonArray[:int(4 * len(maRibbonArray) / 5)]
    # trainAnswer = oneDayGains[:int(4 * len(oneDayGains) / 5)]
    #
    # wins = 0
    # losses = 0
    #
    # for date in range(len(trainAnswer)):
    #     if (trainAnswer[date] == 0):
    #         losses += 1
    #     else:
    #         wins += 1
    #
    # ribbonTest = maRibbonArray[int(4 * len(maRibbonArray) / 5):]
    # testAnswer = oneDayGains[int(4 * len(oneDayGains) / 5):]

    print(maRibbonArray[0])
    maRibbonArray = preprocessing.scale(maRibbonArray)
    print(maRibbonArray[0])
    dict = {}
    resMAR = maRibbonArray[-reservePeriod:][:]
    maRibbonArray = maRibbonArray[:-reservePeriod][:]

    for date in range(len(closes)-(200+reservePeriod)):
        dict.update({(str(maRibbonArray[date][0]) + str(maRibbonArray[date][100]) + str(maRibbonArray[date][199])) : date})
    print(dict)
    np.save('dict',dict)

    np.save('scaledMAR',maRibbonArray)
    np.save('resMAR',resMAR)

    #oneDayGains = preprocessing.scale(oneDayGains)
    #oneDayGains = to_categorical(oneDayGains)


    print(str(len(maRibbon)))
    print(str(len(oneDayGains)))


    xTrain, xTest, yTrain, yTest = train_test_split(maRibbonArray, oneDayGains, test_size=0.2)
    xTrain = np.array(xTrain)
    xTest = np.array(xTest)
    yTrain = np.array(yTrain)
    yTest = np.array(yTest)

    np.save('xTest',xTest)
    np.save('yTest',yTest)
    
    if stop:
        pause()
    # pp.pprint(ribbonTest)
    pp.pprint(xTest)
    pp.pprint(yTest)

    def baseline_model(i):
        model = Sequential()
        model.add(Dense(units=50, activation=activation, input_shape=(200,)))
        model.add(Dense(units=50, activation=activation))
        model.add(Dense(units=50, activation=activation))
        model.add(Dense(units=1, activation = 'sigmoid'))
        model.compile(SGD(lr=i), 'binary_crossentropy', metrics= ['accuracy'])
        return model
    
    def rangeModel(i):
        model = Sequential()
        model.add(Dense(units=50, activation=activation, input_shape=(200,)))
        model.add(Dense(units=50, activation=activation))
        model.add(Dense(units=50, activation=activation))
        model.add(Dense(units=2, activation = activation))
        model.compile(SGD(lr=i, nesterov = True), 'mean_squared_error')
        return model
    
    def ribbonOutputModel(i):
        model = Sequential()
        model.add(Dense(units=50, activation=activation, input_shape=(200,)))
        model.add(Dense(units=50, activation=activation))
        model.add(Dense(units=50, activation=activation))
        model.add(Dense(units=18, activation = 'sigmoid'))
        model.compile(SGD(lr=i, nesterov = True), 'binary_crossentropy', metrics=['accuracy'])
        return model


    LR = [0.001, 0.01, 0.1]

    #for i in LR:
    # Defines linear regression model and its structure
    if mode == 0:
        model = baseline_model(.1)
    elif mode == 1:
        model = ribbonOutputModel(.1)
        

    # Fits model
    history = model.fit(xTrain, yTrain, epochs=numEpochs, validation_split=0.2, verbose=1,batch_size=100)
    history_dict = history.history
    if not first:

        # Plots model's training cost/loss and model's validation split cost/loss
        pp.pprint(history_dict)
        #acc = history_dict['acc']
        val_acc = history_dict['val_acc']
        plt.figure()
        #plt.plot(acc, 'bo', label='training loss')
        plt.plot(val_acc, 'r+', label='val training loss')
        plt.show(block=False)

        # model = baseline_model(.01)
        # history = model.fit(xTrain, yTrain, epochs=80, validation_split=0.2, verbose=1, batch_size=32)
        predictions = model.predict_classes(x=xTest, batch_size=1, verbose=1)  # 1 is buy and 0 is sell
        probs = model.predict(x=xTest, batch_size=1, verbose=1)
        surePreds = []#[dataNum][0 or 1]0 for prediction, 1 for probabilty, 2 for answer
        sum = float(0.0)
        divisor = float(0.0)
        cash = 600.0
        btc = 0.0
        closeB = -1
        closeS = -1

#        for x in range(len(predictions)):
#            if probs[x] > .65:
#                temp = []
#                temp.append(predictions[x])
#                temp.append(probs[x])
#                temp.append(yTest[x])
#                surePreds.append(temp)
#                if(predictions[x] == 0 and x > closeS and x > closeB and btc > 0.000001):#open short
#                    cash += btc*closes[200+x]
#                    btc = 0
#                    closeS=x+7
#                if(predictions[x] == 1 and x > closeS and x > closeB and cash > 0.000001):#open long
#                    btc += cash/closes[200+x]
#                    cash = 0
#                    closeB = x+7
#                if(int(predictions[x]) == int(yTest[x])):
#                    print('TEST')
#                    sum += 1
#            if x == closeB:
#                cash = btc*closes[200+x]
#                btc = 0
#            elif x == closeS:
#                btc = cash/closes[200+x]
#                cash = 0
#        sum = 0
#        mas = []
#        for day in range(200):
#            sum+= closes[-1-day]
#            mas.append(sum/(day+1))
#
#        print(predictions)
#        pp.pprint(probs)
#
#        finalAcc = sum / int(len(surePreds))
#        print(str(finalAcc))
#        print(len(surePreds))
#        print(len(predictions))
#        print('\n')
#        netWorth = btc*closes[len(closes)-1] + cash
#        print('Net Worth = ' + str(netWorth))
        model.save(modelName)
        #nextWeek = model.predict_classes(preprocessing.scale(np.array([mas])),batch_size=1,verbose=1)
        #nwProb = model.predict(preprocessing.scale(np.array([mas])),batch_size=1,verbose=1)
        #print('Big Predicto = ' + str(nextWeek[0]))
        #print(nwProb[0])


        # pp.pprint(history_dict)
        # max = 0
        # maxIndex = 0
        # accs = history_dict['val_accuracy']
        # print(len(accs))
        # for x in range(len(accs)):
        #     if accs[x] > max:
        #         max = accs[x]
        #         maxIndex = x
        # print(maxIndex+1)
        # print(max)
        # seed = 4
        # np.random.seed(seed)
        # model = baseline_model(.1)
        # history = model.fit(xTrain, yTrain, epochs=maxIndex+1, validation_split=0.2, verbose=1,batch_size=8)
        # model.save(modelName)
        plt.show()
        done = True
    else:
        max = 0
        maxIndex = 0
        if mode == 0:
            accs = history_dict['val_loss']
        if mode == 1:
            accs = history_dict['val_acc']
        print(len(accs))
        for x in range(len(accs)):
            if accs[x] > max:
                max = accs[x]
                maxIndex = x
        first = False
        numEpochs = maxIndex
        
#%%

import requests
import pprint as pp
import sys
import pandas
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
import numpy as np

modelPath = 'BTCUSD10-14-19!1296-56-21-1'
model = load_model(modelPath)#latestModel is the best BTC model as of now
xTest = np.load('xTest.npy')
yTest = np.load('yTest.npy')
resp = np.load('closes.npy')
dict = np.load('dict.npy',allow_pickle=True)
resMAR = np.load('resMAR.npy')
dict = dict.item()



# seed = 4
# np.random.seed(seed)

pp.pprint(dict)
print(len(dict))


predictions = []
probs = []
for x in range(len(xTest)):
    prediction = model.predict_classes(x=np.array([xTest[x]]), batch_size=1, verbose=1)  # 1 is buy and 0 is sell
    prob = model.predict(x=np.array([xTest[x]]), batch_size=1, verbose=1)
    predictions.append(prediction)
    probs.append(prob[0])

surePreds = []#[dataNum][0 or 1]0 for prediction, 1 for probabilty, 2 for answer
sum = float(0.0)
divisor = float(0.0)
cash = 600.0
btc = 0.0
closeB = -1
closeS = -1
sures = 0

tempStr = modelPath[modelPath.find('!')+1:]
tempStr = tempStr[tempStr.find('-')+1:]

samples = int(tempStr[:tempStr.find('-')])
tempStr = tempStr[tempStr.find('-')+1:]

predictivePeriod = int(tempStr[:tempStr.find('-')])
tempStr = tempStr[tempStr.find('-')+1:]

mode = int(tempStr)

sureness = .2#for mode one sureness ranges from 0 to 1. for mode zero, sureness ranges from 0 -.5



buySum = 0.0
sellSum = 0.0
buySures = 0.0
sellSures = 0.0

def blankProbs():
    temp = []
    for x in range(18):
        temp.append(None)
    return temp

if mode == 1:
    surePreds = []
    for x in range(len(probs)):
        tempPreds = blankProbs()
        for y in range(len(probs[x])):
            if (float(probs[x][y]) > 1-sureness):
                tempPreds[y] = probs[x][y]
        surePreds.append(tempPreds)
        
elif mode == 0:     
    for x in range(len(predictions)):
        if probs[x] > 1-sureness or probs[x] < sureness:
            temp = []
            temp.append(predictions[x])
            temp.append(probs[x])
            temp.append(yTest[x])
            surePreds.append(temp)
    #        if(predictions[x] == 0 and x > closeS and x > closeB and btc > 0.000001):#open short
    #            cash += btc*resp[200+dict[str(xTest[x][0])+str(xTest[x][100]) + str(xTest[x][199])]]
    #            btc = 0
    #            closeS=x+7
    #        if(predictions[x] == 1 and x > closeS and x > closeB and cash > 0.000001):#open long
    #            btc += cash/resp[200+dict[str(xTest[x][0])+str(xTest[x][100]) + str(xTest[x][199])]]
    #            cash = 0
    #            closeB = x+7
            if(int(predictions[x]) == int(yTest[x]) and int(predictions[x]) == 1):
                print('TEST')
                buySum += 1
            if(int(predictions[x]) == int(yTest[x]) and int(predictions[x]) == 0):
                print('TEST')
                sellSum += 1
            if int(predictions[x]) == 1:
                buySures+= 1
            if int(predictions[x]) == 0:
                sellSures+= 1
        if x == closeB:
            cash = btc*resp[200+dict[str(xTest[x][0])+str(xTest[x][100]) + str(xTest[x][199])]]
            btc = 0
        elif x == closeS:
            btc = cash/resp[200+dict[str(xTest[x][0])+str(xTest[x][100]) + str(xTest[x][199])]]
            cash = 0

#testAccuracy = sum/sures

print('RESPONSE')
pp.pprint(resp)

# lastTest = []

# for date in range(samples):
#     sum = 0
#     mas = []
#     print(resp[-(date+1)])
#     for day in range(200):
#         sum+= resp[-(date+1+day)]
#         mas.append(sum/(day+1))
#     lastTest.append(mas)
#
# pp.pprint(lastTest)

def rangeVal(price,case):
    unit = .025
    if case == 0:
        return price - price*unit*8
    elif case == 1:
        return price - price*unit*7
    elif case == 2:
        return price - price*unit*6
    elif case == 3:
        return price - price*unit*5
    elif case == 4:
        return price - price*unit*4
    elif case == 5:
        return price - price*unit*3
    elif case == 6:
        return price - price*unit*2
    elif case == 7:
        return price - price*unit*1
    elif case == 8:
        return price
    elif case == 9:
        return price
    elif case == 10:
        return price *(1+unit)
    elif case == 11:
        return price *(1+unit*2)
    elif case == 12:
        return price *(1+unit*3)
    elif case == 13:
        return price *(1+unit*4)
    elif case == 14:
        return price *(1+unit*5)
    elif case == 15:
        return price *(1+unit*6)
    elif case == 16:
        return price *(1+unit*7)
    elif case == 17:
        return price *(1+unit*8)

if mode == 0:
    print(predictions)
    pp.pprint(probs)
    
    finalAcc = sum / int(len(surePreds))
    print(len(surePreds))
    print(len(predictions))
    print('\n')
    buyAcc = buySum/buySures
    sellAcc = sellSum/sellSures
    netWorth = btc*resp[len(resp)-1] + cash
    resDownPreds = []
    resUpPreds = []
    resDownDates = []
    resUpDates = []
    resPrices = []
    for x in range(samples+predictivePeriod):
        resUpPreds.append(None)
        resDownPreds.append(None)
        resDownDates.append(None)
        resUpDates.append(None)
    for x in range(samples):
        nextWeek = model.predict_classes(np.array([resMAR[x]]),batch_size=1,verbose=1)
        nwProb = model.predict_proba(np.array([resMAR[x]]),batch_size=1,verbose=1)
        print(str(samples-x) + ' days ago')
        print(nextWeek)
        print(nwProb)
        print('price: ' + str(resp[-(samples-x)]))
        if nwProb[0] > 1-sureness or nwProb[0] < sureness:
            if nwProb[0] > 1-sureness:
                resUpPreds[x+predictivePeriod] = float(resp[-(samples-x)])
                resUpDates[x] = float(resp[-(samples-x)])
            elif nwProb[0] < sureness:
                resDownPreds[x+predictivePeriod] = float(resp[-(samples-x)])
                resDownDates[x] = float(resp[-(samples-x)])
        resPrices.append(float(resp[-(samples-x)]))
        print(resp[-1])
    print('buy acc: ' + str(buyAcc) + ' buy samples: ' + str(buySures))
    print('sell acc: ' + str(sellAcc) + ' sell samples: ' + str(sellSures))
    print('samples: ' + str(len(predictions)))
    
elif mode == 1:
    lowHighs = []
    for x in range(samples+predictivePeriod):
        lowHighs.append(None)
    for x in range(samples):
        tempProbs = model.predict_proba(np.array([resMAR[x]]),batch_size=1,verbose=1)
        surePreds = []
        surePreds.append(None)
        surePreds.append(None)
        print(tempProbs)
        for y in range(len(tempProbs[0])):
            if tempProbs[0][y] > 1-sureness:
                if y < 9:
                    surePreds[0] = y
                else:
                    surePreds[1] = y
            lowHighs.append(surePreds)
        
#        for y in range()temp:
    for x in range(len(yTest)):
        pp.pprint(yTest[x])
        
        
        

#print(len(resUpPreds))
#print(len(resPrices))
#
#plt.figure()
#plt.plot(resPrices)
#plt.plot(resUpPreds,'go')
#plt.plot(resDownPreds,'ro')
#plt.plot(resUpDates,'g+')
#plt.plot(resDownDates,'r+')
#plt.show()
#print('ok')

    
