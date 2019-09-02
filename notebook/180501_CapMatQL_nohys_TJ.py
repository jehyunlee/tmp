#-*-coding:utf-8-*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
import random
import copy
from sys import argv

time_start = time.time()    # Running time estimation
#######################################################################################################################
# Wind data
vCutIn = 3.5                # Cut-in Velocity
vRated = 13                  # Rated Velocity
block_size = 10               # Data unit block size

# Q-Learning
q_learning = 1              # Q-Learning ON. 0 = 'off', others = 'on'
alpha = 0.5                   # Learning rate
gamma = 0.9                 # Discounting factor
iterationQ = 30000            # 최적의 index set 을 찾기 위한 반복 횟수
q_method = 3                # 1 : 'freq' : Data Pre-processing 결과 frequency 가 낮은 bin 우선
# 2 : 'random' : 동일 Q 중 random 선택
# 3 : 'e-random greedy' : decaying epsilon-greedy.
#                       1. 선택 가능한 Action 중 Q가 가장 큰 값 선택
#                       2. 가장 큰 Q가 복수일 경우 이들 중 랜덤으로 하나 선택
#                       * iteration num.에 반비례하는 수 e의 확률로 random walk
# 4 : 'e-freq greedy' : decaying epsilon-greedy.
#                       1. 선택 가능한 Action 중 Q가 가장 큰 값 선택
#                       2. 가장 큰 Q가 복수일 경우 남아있는 빈도가 가장 적은 bin 선택
#                       * iteration num.에 반비례하는 수 e의 확률로 random walk

# File Input & Output
noutfile = 'output'                 # Name of output file series
ninfile = 'TminData.xlsx'     # Excel format input file.

# Column names should be: "indexMinutes" and "vWind",
windvis = 0                 # vWind vs indexMinutes Graph 출력.
qlvis = 0                   # Q-Learning result visualization

######################################################################################################################


class DataInfo:                     # 입력 data와 1차 처리 결과.
    vMax = int(vRated + 4)          # Maximum Velocity
    vMin = int(np.floor(vCutIn))    # CutIn Velocity를 버림한 값. bin 작성에 활용.
    nbin = vMax-vMin                # number of bins

    # Input data pre-processing
    data = pd.read_excel(ninfile)       # pandas data frame 형식으로 data load

    print('number of raw data = ', data.vWind.size)
    data_valid = data.loc[vCutIn <= data.vWind].loc[data.vWind < vMax]   # Valid data: vCutIn, vMax 외 data 버림.
    data_valid = data_valid.reset_index(drop=True)
    data_valid_size = data_valid.index.size
    data_invalid = pd.concat([data.loc[vCutIn > data.vWind], data.loc[data.vWind >= vMax]]) # Invalid data
    print('number of valid data = ', data_valid_size)

    # Starting point of a day: 날짜별 첫 data 추출해서 모음
    indexStart = pd.DataFrame(columns=['dateAndTime', 'indexMinutes', 'data_valid_index',
                                       'indexMinutesFinal', 'indexFinal', 'dateAndTimeFinal'])

    # Preparation of Q-Learning
    data_valid['nextRelIndex'] = pd.np.empty((data_valid_size, 0)).tolist()      # action candidates at the state
    data_valid['Q'] = pd.np.empty((data_valid_size, 0)).tolist()                 # action-values (reward)
    data_valid['binIndex'] = pd.np.empty((data_valid_size, 0)).tolist()          # bin of the action-state (new_state)

    # Q-Learning parameter 초기화
    for i in range(data_valid_size):
        tmp_indexMin = data_valid.indexMinutes.loc[i]
        temp1 = data_valid.loc[data_valid.indexMinutes >= tmp_indexMin + block_size]
        if temp1.indexMinutes.size > 0:
            temp1 = temp1.loc[data_valid.indexMinutes <= temp1.indexMinutes.iloc[0] + (block_size-1)]
            temp = temp1.indexMinutes.tolist()
            templen = len(temp)
            temp_wind = (np.floor(temp1.vWind.tolist())-vMin).astype(int)
            data_valid.at[i, 'nextRelIndex'] = [0]*templen

            for j in range(templen):
                data_valid.nextRelIndex.loc[i][j] = temp1.axes[0][j] - data_valid.axes[0][i]
            data_valid.at[i, 'Q'] = [0]*templen
            data_valid.at[i, 'binIndex'] = temp_wind

    # frequency 기반 greedy 알고리즘 적용을 위한 data evaluation matrix 작성
    dataEval = np.zeros((6, nbin))          # 5x14 Matrix
    dataEval[0] = np.arange(vMin, vMax)     # Row1~2 = CaptureMatrix,
    dataEval[0][0] = vCutIn
    dataEval[1] = np.arange(vMin+1, vMax+1)

    CM_target = np.array([20]*(nbin-2) + [10]*2)          # Capture Matrix Target
    print('Capture Matrix Target = ', CM_target)

    dateAndTimeFinal = '0'
    errorflag = 0

    # optimized CM
    CM_sequence_opt = []
    CapMat_opt = []

#======================================================
# Functions


def set_start_index():        # 하루 중 첫 data 설정
    date = 'arbitrary'
    for i in range(DataInfo.data_valid_size):
        date_old = date
        date_time = DataInfo.data_valid.dateAndTime.iloc[i]
        date = date_time[1:11]
        if date != date_old:
            DataInfo.indexStart = DataInfo.indexStart.append({'dateAndTime': date_time,
                                                              'indexMinutes': DataInfo.data_valid.indexMinutes.loc[i],
                                                              'data_valid_index': i}, ignore_index=True)


def get_start_index(date_order):         # 하루 중 첫 data 가져오기.
    if date_order >= DataInfo.indexStart.size:
        print("date_order is out of range!")
        exit(1)
    else:
        #print(DataInfo.indexStart.data_valid_index.iloc[date_order])
        #print(DataInfo.data_valid.loc[DataInfo.indexStart.data_valid_index.iloc[date_order]])
        return DataInfo.indexStart.data_valid_index.iloc[date_order]


# Valid data 분석
def analyze_data(noutfile, date_order, windvis):   # Input data (vWind) Analysis
    start_index_min = str(DataInfo.indexStart.indexMinutes.iloc[date_order])
    print('starting Data analysis from @indexMinutes = ' + start_index_min + ', '
          + str(DataInfo.indexStart.dateAndTime.loc[date_order]))
    noutlog = noutfile + '_JS_' + str(date_order) + '.txt'
    outlog = open(noutlog, 'w')

    # Data Analysis for Capture Matrix
    # dataEval Row4 = No. of valid elements in the wind velocity bin
    indexMin = 0
    DataInfo.dataEval[3] = np.zeros(DataInfo.vMax - DataInfo.vMin)

    CM_complete = 0
    binarray = DataInfo.dataEval[0:2]

    date = 'arbitrary'

    # Just Stacking 으로 Capture Matrix 작성 : data 앞에서부터 block_size 만큼의 data를 골라 Compact Matrix 작성
    CM_tmp = []
    state = get_start_index(date_order)
    for i in range(get_start_index(date_order), DataInfo.data_valid_size):

        v_wind = DataInfo.data_valid.vWind.loc[i]
        bin = np.nonzero((binarray[0] <= v_wind) == (binarray[1] > v_wind))[0][0]

        # dataEval[3] : Just Stacking으로 Capture Matrix를 완성할때까지 등장하는 bin 별 원소의 수
        if DataInfo.data_valid.indexMinutes.loc[i] - indexMin >= block_size:
            DataInfo.dataEval[3][int(np.floor(DataInfo.data_valid.vWind.loc[i])-DataInfo.vMin)] += 1

        outlog.write(str(DataInfo.data_valid.loc[i].vWind) + '\tbin= ' + str(bin) + '\tindexMin= '
                     + str(DataInfo.data_valid.loc[i].indexMinutes) + '\tdateAndTime= '
                     + str(DataInfo.data_valid.dateAndTime.loc[i]) + '\t')
        outlog.write(str(DataInfo.dataEval[3].astype(int)))

        if DataInfo.data_valid.indexMinutes.loc[i] - indexMin >= block_size:
            new_state = i
            delta = new_state - state
            #action = np.nonzero(np.array(DataInfo.data_valid.nextRelIndex.loc[state]) == delta)[0][0]
            #print('state = ', state, 'new_state = ', new_state, 'delta = ', delta, 'action = ', action, DataInfo.data_valid.nextRelIndex.loc[state])
            outlog.write(' < ' + str(DataInfo.dataEval[0][int(bin)]))
            indexMin = DataInfo.data_valid.indexMinutes.loc[i]
            CM_tmp.append([i, 0, DataInfo.data_valid.indexMinutes.loc[i], DataInfo.data_valid.dateAndTime.loc[i]]) # state, action, indexMinutes[state], dateAndTime[state]
            state = new_state

        outlog.write('\n')

        # dataEval[2] : Capture Matrix
        DataInfo.dataEval[2][bin] += 1

        # Capture Matrix 작성 완료시 indexStart 입력
        if False not in (DataInfo.dataEval[3] >= DataInfo.CM_target):
            DataInfo.dataEval[4] = DataInfo.dataEval[2]/DataInfo.CM_target # Row5 = Data density @CM is filled by just data stacking
            DataInfo.indexStart.at[date_order, 'indexFinal' ] = i                   # Final Index
            DataInfo.indexStart.at[date_order, 'indexMinutesFinal'] = DataInfo.data_valid.indexMinutes.loc[i]     # final indexMinutes
            DataInfo.indexStart.at[date_order, 'dateAndTimeFinal'] = DataInfo.data_valid.dateAndTime.loc[i]       # final dateAndTime
            #print(DataInfo.dataEval[2:5])
            print('CM will full at most @indexMinutes = '
                  + str(DataInfo.indexStart.indexMinutesFinal.loc[date_order]) +' ,' + str(DataInfo.indexStart.dateAndTimeFinal.loc[date_order])+'\n')
            CM_complete = 1
            DataInfo.CM_sequence_opt = CM_tmp
            DataInfo.CapMat_opt = DataInfo.dataEval[3]
            #print('CM_sequence_opt @JS =', DataInfo.CM_sequence_opt)
            return DataInfo.dataEval[2]

    # data 부족시 에러메시지 출력, 현재 data로 Capture Matrix 작성 시도.
    if CM_complete == 0:
        print('\n### Error! Data is insufficient!!!\n')
        print('### building CM_sequence_opt @JS Failed')
        print('### Capture Matrix generation will be tried anyway.........')
        DataInfo.dataEval[4] = DataInfo.dataEval[2]/DataInfo.CM_target # Row5 = Data density @CM is filled by just data stacking
        DataInfo.indexStart.at[date_order, 'indexFinal' ] = i                    # Final Index
        DataInfo.indexStart.at[date_order, 'indexMinutesFinal'] = DataInfo.data_valid.indexMinutes.loc[i]     # final indexMinutes
        DataInfo.indexStart.at[date_order, 'dateAndTimeFinal'] = DataInfo.data_valid.dateAndTime.loc[i]       # final dateAndTime
        print('CM will full at most @indexMinutes = '
              + str(DataInfo.indexStart.indexMinutesFinal.loc[date_order]) +' ,' + str(DataInfo.indexStart.dateAndTimeFinal.loc[date_order])+'\n')

    #--------------------------------------------------------
    # Wind Data 분석, 출력
    #-- Save Log
    outlog.close()
    #-- Save Figure
    if windvis != 0:
        noutfig_time = noutfile+'.png'
        font_time = fm.FontProperties(size=4)
        fig_time = plt.figure(figsize=(100, 4), frameon=False)
        plot_time = fig_time.add_subplot(1, 1, 1)
        plot_time.xaxis.get_label().set_fontproperties(font_time)
        plot_time.yaxis.get_label().set_fontproperties(font_time)
        plot_time.tick_params(axis='both', labelsize=4)

        plt.scatter(DataInfo.data_valid.indexMinutes, DataInfo.data_valid.vWind, c=np.floor(DataInfo.data_valid.vWind),
                    cmap='jet', linewidths=0.1, s=2, edgecolor='k', vmin=vCutIn, vmax=DataInfo.vMax, zorder=1)
        plt.scatter(DataInfo.data_invalid.indexMinutes, DataInfo.data_invalid.vWind, color='w', linewidths=0.1, s=2,
                    zorder=1)
        plt.plot(DataInfo.data.indexMinutes, DataInfo.data.vWind, color='grey', lw=0.3, zorder=0)

        plt.xticks(np.arange(0, int(np.array(DataInfo.data.tail(1).indexMinutes)[0]), 200), rotation='vertical')
        plt.yticks(np.arange(DataInfo.vMin, DataInfo.vMax+1, 1))
        plt.grid(color='gray', linestyle='-', linewidth='0.2')
        plt.xlabel('indexMinutes')
        plt.ylabel('vWind')
        plt.xlim(np.array(DataInfo.data.head(1).indexMinutes)[0], np.array(DataInfo.data.tail(1).indexMinutes)[0])

        plt.savefig(noutfig_time, dpi=300, facecolor='w', transparent=True, bbox_inches='tight', pad_inches=0.0)


# Action by Random
def rargmax(vector):
    # Argmax that choose randomly among eligible maximum indices
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)

# Action by Greedy algorithm : 1. Q가 큰 순, 2. Just Stacking 범위 내 빈도가 적은 순으로 action 선택.
def sargmax(tempdata, tempeval, state):
    vector = tempdata.Q.loc[state]
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    freq = [0] * indices.size
    for i in range(indices.size):
        freq[i] = tempeval[4][tempdata.binIndex.loc[state][i]]

    freq_pstv = []
    freq_pstv_flag = 0
    for x in freq:
        if x > 0:
            freq_pstv.append(abs(x))
            freq_pstv_flag = 1          # useful element presents
        else:
            freq_pstv.append(np.inf)

    if freq_pstv_flag != 0:
        action = indices[np.argmin(freq_pstv)]
    else:
        action = indices[np.argmax(freq)]

    return action


# Action by Greedy + random algorithm : 1. Q가 큰 순, 2. 동일 Q 중에서는 random 선택
def srargmax(tempdata, state):
    vector = tempdata.Q.loc[state]
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    action = indices[np.random.randint(0, len(indices))]
    return action


# Initial state selection by frequency greedy
def sargmax_init(tempdata, state):
    vector = tempdata.vWind.loc[state:state + block_size-1].tolist()
    freq = [0] * block_size
    for i in range(block_size):
        freq[i] = (DataInfo.dataEval[4][int(np.floor(vector[i]) - DataInfo.vMin)])
    m = np.amin(freq)
    state_init = state + np.nonzero(freq == m)[0][0]
    print('Initial state was chosen as @indexMinutes=', tempdata.indexMinutes.loc[state_init] , ', by frequency.')
    return state_init


def updateEval(tempdata, tempeval, state, new_state, initRow):

    margin = 0
    if new_state >= tempdata.index.size + initRow:
        print("Capture Matrix failed!!!")
        DataInfo.errorflag = 1
    else:
        margin = initRow + tempdata.index.size - new_state
        if margin > block_size:
            margin = block_size

        temp = tempdata.loc[tempdata.indexMinutes < tempdata.indexMinutes.loc[new_state] + margin]
        #print('tempdata.index.size = ', tempdata.index.size, '\tnew_state = ', new_state, '\tmargin = ', margin)
        #print('updateEval temp = ', temp)

        updateboundary = temp.axes[0][-1]
        ##print('#indexMinutes @updateboundary = ', tempdata.indexMinutes.loc[updateboundary])

        if False not in (tempeval == DataInfo.dataEval): # if the first run at the iteration
            state = state-1                             #Run following update include 'state', the first data of the date.
        else:                                           # if not the first run, the 'updateboundary' of the previous update should be considered.
            temp = tempdata.loc[tempdata.indexMinutes < tempdata.indexMinutes.loc[state] + block_size]
            state = temp.axes[0][-1]

        for i in range(state+1, updateboundary+1):
            state_vWind = tempdata.vWind.loc[i]
            if state_vWind >= 0:            # avoid the case of 'index = -1' to select 1st data of the day
                state_bin = np.nonzero((tempeval[0] <= state_vWind) == (tempeval[1] > state_vWind))[0][0]
                tempeval[2][state_bin] -= 1
                tempeval[4][state_bin] = tempeval[2][state_bin]/DataInfo.CM_target[state_bin]

    return tempeval

#======================================================
def qLearning(startindex, alpha, gamma, iterationQ, trial):
    #------------------------------------------------------
    # Q Iteration
    initRow = get_start_index(startindex)
    finalRow = DataInfo.indexStart.indexFinal.loc[startindex]

    ql_range = (finalRow - initRow) * 1.2 + initRow # 20% margin
    print('initRow, finalRow, ql_range = ', initRow, finalRow, ql_range)
	
    tempdata = DataInfo.data_valid.loc[DataInfo.data_valid.index <= ql_range]
    tempdata = tempdata.loc[tempdata.index >= initRow]

    nextReltemp = tempdata.loc[tempdata.indexMinutes < DataInfo.indexStart.indexMinutes.loc[startindex] + block_size]	
    nextRelIndex = np.arange(1, nextReltemp.index.size+1).tolist()

    tempdata.loc[initRow-1] = ['-', 0, -1.0, 0.0, nextRelIndex, [0]*len(nextRelIndex), [0]*len(nextRelIndex)]
    tempdata.index = tempdata.index + 1
    tempdata = tempdata.sort_index()

    cList = []      # List to contain total counts per episode
    iList = []      # List to contain final indexMinutes per episode
    Q_occu = []      # Occupation of Q Matrix

    nouttail = '_'+str(gamma)+'_'+str(iterationQ) + '_'+str(q_method)+'_ql_try'+ str(trial)
    noutql = noutfile + '_' + str(startindex) + nouttail + '.txt'
    nCMql = noutfile + '_' + str(startindex) + '_CM'+nouttail + '.txt'
    outql = open(noutql, 'w')
    CMql = open(nCMql, 'w')

    for i in range(iterationQ):
        CMql.write('\ni='+str(i)+' ')

        ### Reset environment and get first new observation
        count = 0
        #state = get_start_index(startindex)          # 시작점 설정
        state = initRow
        #tempdata = tempdata.loc[tempdata.index >= state]
        #if count == 0:
        #    print(tempdata.head(3))

        tempeval = copy.deepcopy(DataInfo.dataEval)
        #new_state = srargmax(tempdata, state)
        #new_state = tempdata.nextRelIndex.loc[state][srargmax(tempdata, state)] # Q 기준 initial state update
        new_state = initRow

        DataInfo.errorflag = 0
        rAll = 0                        # reward = 0
        done = False                    # Capture Matrix full check
        CM_current = np.zeros(DataInfo.nbin, dtype=int)       # reset building Capture Matrix

        # for e-greedy
        e = 1./ ((i//100)+1)

        # reduce local minimum problem
        CM_tmp = []

        action = 0
        while not done:
            #         updateEval(tempdata, tempeval, state, new_state, initRow)
            if DataInfo.errorflag == 1:
                #CMql.write('\n')
                break
            state = new_state
            count += 1
            CMql.write(str(tempdata.indexMinutes.loc[state])+' ')

            bin_current = int(np.floor(tempdata.vWind.loc[state])-DataInfo.vMin)
            if bin_current >= 0:
                CM_current[bin_current] += 1

            reward = 0

            if CM_current[bin_current] == DataInfo.CM_target[bin_current]:
                tempeval[5][bin_current] = 1
                tempeval[4][bin_current] = tempeval[4][bin_current] + 1000
                #reward = np.exp(DataInfo.indexStart.indexMinutesFinal.loc[startindex] - tempdata.indexMinutes.loc[state])

            if 0 not in tempeval[5]:
                try:
                    print('###i = ', i, ' !!! Capture Matrix Complete !!!, indexMinutesFinal = ', tempdata.indexMinutes.loc[state])
                except OSError as e:
                    pass
                reward = np.exp(DataInfo.indexStart.indexMinutesFinal.loc[startindex] - tempdata.indexMinutes.loc[state])
                #reward = 100
                cList.append([i, count])
                iList.append([i, tempdata.indexMinutes.loc[state]])
                outql.write('i='+str(i)+' Capture Matrix = ' + str(CM_current) + '\t' + 'finalIndexMinutes = ' + str(tempdata.indexMinutes.loc[state]) + '\n')
                #CMql.write('\n')
                tempQ = check_Q_occu(initRow, finalRow, tempdata)
                Q_occu.append([i, tempQ[0], tempQ[1], tempQ[2]])
                done = True

                if len(DataInfo.CM_sequence_opt) != 0:
                    if tempdata.indexMinutes[state] <= DataInfo.CM_sequence_opt[-1][2]:
                        DataInfo.CM_sequence_opt = CM_tmp
                        #tempdata = Q_fill_history(CM_tmp, tempdata, reward, 'D')

                        #saveQ(initRow, tempdata, i)
                else:
                    DataInfo.CM_sequence_opt = CM_tmp

                DataInfo.CapMat_opt = CM_current
                #tempdata = Q_fill_history(CM_tmp, tempdata, reward, 'ND')

            tempQ = tempdata.Q.loc[state]
            #tempQlen = len(tempdata.Q.loc[state])
            tempQlen = len(tempQ)
            if q_method == 2 : #'random'
                action = rargmax(tempdata.Q.loc[state])
            elif q_method == 1: # 'freq'
                action = sargmax(tempdata, tempeval, state)
            elif q_method == 3: #'egreedy-random'
                if (np.random.rand(1) < e) and (tempQlen > 1):
                    CMql.write('R')
                    action = np.random.randint(0, tempQlen)
                else:
                    action = srargmax(tempdata, state)
            elif q_method == 4: #'egreedy-frequency'
                if (np.random.rand(1) < e) and (tempQlen > 1):
                    CMql.write('R')
                    action = np.random.randint(0, tempQlen)
                else:
                    action = sargmax(tempdata, state)

            try:
                new_state = state + tempdata.nextRelIndex.loc[state][action]
            except:
                print('Error at "new_state = state + tempdata.nextRelIndex.loc[state][action]", state = ', state, '\taction = ', action)
                saveQ(initRow, tempdata, i, startindex)
                break


            if new_state > ql_range:
                #print('###i = ', i, ' !!! Exceeds expected finalRow !!! Exit !!!')
                CMql.write(' *** Fail ***')
                break

            if len(tempdata.Q.loc[new_state]) != 0:
                #tempdata.Q.loc[state][action] = reward + gamma*np.max(tempdata.Q.loc[new_state])
                #saveQ(initRow, tempdata, i, startindex)
                tempdata.Q.loc[state][action] = (1-alpha)*tempdata.Q[state][action] + alpha * (reward + gamma * np.max(tempdata.Q.loc[new_state])) # non-deterministic
                #pass # temporary

            else:
                print('### Error! Q[new_state] is empty!')
                print('Capture Matrix = ', CM_current)
                print('state = ', state, '\taction = ', action)
                print('indexMinutes @state= ', tempdata.indexMinutes.loc[state])
                print('nextRelIndex @state = ', tempdata.nextRelIndex.loc[state])
                print('new_state = ', new_state)
                print('indexMinutes @new_state= ', tempdata.indexMinutes.loc[new_state])
                print('nextRelIndex @new_state = ', tempdata.nextRelIndex.loc[new_state])
                exit(1)

            rAll += reward
            # state, action, indexMinutes[state], dateAndTime[state]
            CM_tmp.append([state, action, tempdata.indexMinutes.loc[state], tempdata.dateAndTime.loc[state]])


    outql.close()
    CMql.close()

    if qlvis != 0:
        cList = np.array(cList)
        plt.scatter(cList[:,0], cList[:,1], color='orange')
        plt.xlim(0, iterationQ)
        plt.savefig('bin_count' + '_' + str(startindex) + nouttail + '.png')
        plt.close()

        iList = np.array(iList)
        plt.scatter(iList[:,0], iList[:,1], color='red')
        plt.xlim(0, iterationQ)
        plt.savefig('indexMinutes' + '_' + str(startindex) + nouttail + '.png')
        plt.close()

        Q_occu = np.array(Q_occu)
        plt.scatter(Q_occu[:, 0], Q_occu[:, 2], color='green')
        plt.savefig('Q_occupation_num' + '_' + str(startindex) + nouttail + '.png')
        plt.close()

        plt.scatter(Q_occu[:, 0], Q_occu[:, 3], color='blue')
        plt.savefig('Q_occupation_rate' + '_' + str(startindex) + nouttail + '.png')
        plt.close()

    # Procedure file
    if len(Q_occu) > 0:
        nproc = 'procedure_' + str(startindex) + nouttail + '.csv'
        proc = open(nproc, 'w')
        proc.write('iteration, bin_count,indexMinutes, num_Q_occu(total=' + str(Q_occu[0][1]) + '), Q_occu_rate\n')
        for i in range(len(cList)):
            proc.write(str(cList[i][0]) + ',' + str(cList[i][1]) + ',' + str(iList[i][1]) + ',' + str(Q_occu[i][2]) + ',' + str(Q_occu[i][3])+'\n')
        proc.close()

        saveQ(initRow, tempdata, iterationQ, startindex)

    CaptureMatrix(tempdata, startindex, trial)


def Q_fill_history(CM_array, data, reward, option):
    arraylen = len(CM_array)
    # Non-Deterministric for normal case
    if option == 'ND':
        for i in range(arraylen-1, -1, -1):
            data.Q.loc[CM_array[i][0]][CM_array[i][1]] = (1-alpha) * data.Q.loc[CM_array[i][0]][CM_array[i][1]] + alpha * (np.power(gamma, (arraylen-i-1)) * reward) # Non-deterministic
            #print(data.Q.loc[CM_array[i][0]][CM_array[i][1]])
    # Deterministic for optimum case
    elif option == 'D':
        for i in range(arraylen-1, -1, -1):
            data.Q.loc[CM_array[i][0]][CM_array[i][1]] = np.power(gamma, (arraylen-i-1)) * reward  # Deterministic
    return data

def saveQ(initRow, data, iteration, startindex):
    nQ = 'Q_'+ '_' + str(startindex) + '_' + str(iteration)+'.txt'
    Q = open(nQ, 'w')
    for i in range(initRow, initRow + data.index.size):
        Q.write(str(data.indexMinutes.loc[i])+'\t'+str(data.nextRelIndex.loc[i])+'\t'+str(data.Q.loc[i])+'\n')
    Q.close()

def check_Q_occu(initRow, finalRow, tempdata):
    nQ_total = 0
    nQ_occupied = 0
    for i in range(initRow+1, int(finalRow)):
        nQ_total += len(tempdata.Q.loc[i])
        nQ_occupied += len(np.nonzero(tempdata.Q.loc[i])[0])
    Occu_rate = nQ_occupied/nQ_total*100
    return [nQ_total, nQ_occupied, Occu_rate]

def CaptureMatrix(tempdata, startindex, trial):
    
    noutfinal = noutfile + '_' + str(startindex) + '_try'+str(trial)+'.txt'
    outfinal = open(noutfinal, 'w')

    ### by saved optimum
    outfinal.write('\n\n### by saved optimum CM ###\n')
    for i in range(len(DataInfo.CM_sequence_opt)):
        outfinal.write(str(DataInfo.CM_sequence_opt[i][2])+' ')
    outfinal.write('\n')
    outfinal.write('Capture Matrix = [ ')
    for i in range(len(DataInfo.CapMat_opt)):
        outfinal.write(str(DataInfo.CapMat_opt[i]) + ' ')
    outfinal.write(']\tfinalIndexMinutes = ' + str(DataInfo.CM_sequence_opt[-1][2]) + '\n')


    ### by Q Matrix
    count = 0
    state = get_start_index(startindex)          # 시작점 설정

    new_state = state # frequency 기준 initial state update
    tempeval = copy.deepcopy(DataInfo.dataEval)

    #rAll = 0                        # reward = 0
    done = False                    # Capture Matrix full check
    CM_current = np.zeros(DataInfo.nbin, dtype=int)       # reset Capture Matrix

    outfinal.write('### by Q values ###\n')

    while not done:
        #updateEval(tempdata, tempeval, state, new_state)
        state = new_state

        if state >= tempdata.tail(1).index[0]:
            outfinal.write('\n Capture Matrix building by Q matrix failed!!! \n')
            done = True
            break

        outfinal.write(str(tempdata.indexMinutes.loc[state]) + ' ')

        count += 1

        bin_current = int(np.floor(tempdata.vWind.loc[state])-DataInfo.vMin)
        if bin_current >= 0:
            CM_current[bin_current] += 1

        action = srargmax(tempdata, state)
        new_state = state + tempdata.nextRelIndex.loc[state][action]

        if CM_current[bin_current] == DataInfo.CM_target[bin_current]:
            tempeval[5][bin_current] = 1
            tempeval[4][bin_current] = tempeval[4][bin_current] + 1000

        if 0 not in tempeval[5]:
            print('### Capture Matrix Optimization Complete !!!, indexMinutesFinal = ', tempdata.indexMinutes.loc[state])
            outfinal.write('\nCapture Matrix = ' + str(CM_current) + '\t' + 'finalIndexMinutes = ' + str(tempdata.indexMinutes.loc[state]) + '\n')
            done = True

    outfinal.close()

#======================================================
# Main

print(argv)
set_start_index()
i = int(argv[1])

analyze_data(noutfile, i, windvis)
get_start_index(i)
qLearning(i, alpha, gamma, iterationQ, 1)

print('--- ' + str("%.0f" %(time.time()-time_start)) + ' seconds ---')     # print Running time


