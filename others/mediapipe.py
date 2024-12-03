import cv2
import mediapipe as mp
import time
import traceback
import socket
import sys


commandDict = {'right':'10','left':'01','forward':'11','backforward':'00','unKnown':'-1'}

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

# 全局变量
frameNum = 0
commandLst = []
commandSending = ''

flag = 0


while True:
    # 初始化0关键点的坐标
    lst = [0, 0, 0]
    # 初始化字典
    distanceDict = {}

    success, img = cap.read()
    imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)


    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                ## 存储0关键点的三个坐标
                if id == 0:
                    lst = [lm.x,lm.y,lm.z]
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                ## 绘制手部关键结点
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

                ## 分别检测4，8，12，20四个关键结点与0结点间的距离判断手指指向
                if id == 4 or id == 8 or id == 12 or id == 20:
                    distanceSum = 0
                    distanceSum = (lst[0]-lm.x)**2 + (lst[1]-lm.y)**2 + (lst[2]-lm.z)**2
                    distanceDict[id] = distanceSum

            ## mediapipe中首部关键结点的连线
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    command = 'UnKnown'
    if distanceDict == {}:
        MaxId = 0
    else:
        MaxId = [key for key,value in distanceDict.items() if value == max(distanceDict.values())][0]
    ## 判断哪个结点距离根节点最远，并由此给出相应的命令
    if MaxId == 4:
        command = 'right'
    if MaxId == 8:
        command = 'left'
    if MaxId == 12:
        command = 'forward'
    if MaxId == 20:
        command = 'backforward'
    if MaxId == 0:
        command = 'unKnown'
    cv2.putText(img,command,(10,150), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    ## 计算并显示帧率
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)

    ## 连续判断十帧图像
    frameNum += 1
    commandLst.append(command)
    if frameNum == 5:
        if max(commandLst) == min(commandLst):
            commandSending = commandDict[command]
        else:
            commandSending = commandDict['unKnown']

        ## 数据重新置零
        frameNum = 0
        commandLst = []

        print(commandSending)
    else:
        continue

    ## 此处设置退出键esc,按下esc退出窗口
    key = cv2.waitKey(1)
    if key == 27:
        #通过esc键退出摄像
        cv2.destroyAllWindows()
        break
