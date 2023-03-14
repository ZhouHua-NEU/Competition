import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

#极坐标->直角坐标
def to_xy(l,a):
    a = a * math.pi /180
    return (l * math.cos(a),l * math.sin(a))

#直角坐标->极坐标
def to_pa(pj):
    pt,at= 0,0

    if(pj[0] >= 0 and pj[1] >= 0):
        pt,at= (pj[0] ** 2 + pj[1] ** 2)**(1/2),math.atan(pj[1]/pj[0])/math.pi *180.0

    if(pj[0] < 0 and pj[1] > 0):
        pt,at= (pj[0] ** 2 + pj[1] ** 2)**(1/2), math.atan(pj[1]/pj[0])/math.pi *180.0 + 180

    if(pj[0] <= 0 and pj[1] <= 0):
        pt,at= (pj[0] ** 2 + pj[1] ** 2)**(1/2), math.atan(pj[1]/pj[0])/math.pi *180.0 + 180 

    if(pj[0] > 0 and pj[1] < 0):
        pt,at= (pj[0] ** 2 + pj[1] ** 2)**(1/2), math.atan(pj[1]/pj[0])/math.pi *180.0 + 360
    return pt,at

#用直角坐标系求两点的距离
def distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

#用直角坐标系求夹角
def angle(p1, p2, p3):
    a=math.sqrt((p2[0]-p3[0])*(p2[0]-p3[0])+(p2[1]-p3[1])*(p2[1] - p3[1]))
    b=math.sqrt((p1[0]-p3[0])*(p1[0]-p3[0])+(p1[1]-p3[1])*(p1[1] - p3[1]))
    c=math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))
    A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
    return A
 
#是否在一个圆周上
def is_circle():
    return (location[:,0] == 100).all() and (location[:,1] % 40 ==0).all()

#原始的极坐标
location_angle =  np.array([[0,0],[100,0],[98,40.10],[112,80.21],[105,119.75],[98,159.86],[112,199.96],[105,240.07],[98,280.17],[112,320.28]])
real_angle  =     np.array([[0,0],[100,0],[100,40.],[100,80],[100,120.],[100,160],[100,200],[100,240.0],[100,280.],[100,320.]])

#装换为直角坐标系
location =  np.array([[0,0],[100,0],[98,40.10],[112,80.21],[105,119.75],[98,159.86],[112,199.96],[105,240.07],[98,280.17],[112,320.28]])
real  =     np.array([[0,0],[100,0],[100,40.],[100,80],[100,120.],[100,160],[100,200],[100,240.0],[100,280.],[100,320.]])


for i in range(len(location)): 
    location[i] = to_xy(location[i,0],location[i,1])


for i in range(len(real)): 
    real[i] = to_xy(real[i,0],real[i,1]) 


def scatter():
    theta = []
    for i in [0,40,80,120,160,200,240,280,320,360]:
        theta.append(math.radians(i))
    r = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    plt.figure(figsize=(20,20),dpi=80)

    ax = plt.subplot(111,projection='polar')      #极坐标

    loc_l = location_angle[:,0]
    loc_a = location_angle[:,1]

    theta2 = []
    for i in loc_a:
        theta2.append(math.radians(i))

    ax.plot(theta, r, '+')
    ax.scatter(theta2,loc_l,color='orange') 
    plt.show()



#求1架无人机发射信号的所有可能的角度
angle_1 = set()
for i in range(len(real)):
    for j in range(len(real)):
        if(i==j or i==0 or j==0):
            continue
        angle_1.add(round(angle(real[j],real[i],real[0]),0))

#求2架无人机发射信号的所有可能的角度
angle_2 = set()
for i in range(len(real)):
    for j in range(len(real)):
        for k in range(len(real)):
            if(i==j or i==k or j==k or i==0 or j==0 or k==0):
                continue
            '''
            angle_2.add(round(angle(real[k],real[j],real[i]),2))
            angle_2.add(round(angle(real[k],real[j],real[0]),2))
            angle_2.add(round(angle(real[k],real[i],real[0]),2))
            '''
            angle_2.add(min(round(angle(real[k],real[j],real[i]),2),round(angle(real[k],real[j],real[0]),2)))
            angle_2.add(min(round(angle(real[k],real[i],real[0]),2),round(angle(real[k],real[j],real[0]),2)))
               

#求3架无人机发射信号的所有可能的角度
angle_3 = set()
for i in range(len(real)):
    for j in range(len(real)):
        for k in range(len(real)):
            for l in range(len(real)):
                if(i==j or i==k or i==l or j==k or j==l or k==l or i==0 or j==0 or k==0 or l==0):
                    continue
                a=round(angle(real[l],real[k],real[i]),2)
                b=round(angle(real[l],real[k],real[j]),2)
                c=round(angle(real[l],real[k],real[0]),2)
                d=round(angle(real[l],real[j],real[i]),2)
                e=round(angle(real[l],real[j],real[0]),2)
                f=round(angle(real[l],real[i],real[0]),2)
                l = sorted([a,b,c,d,e])
                
            
        
from sympy import symbols,nsolve
x, y = symbols('x, y')
lst = []

def adjust(pj,p1,p2,p3):
    p,a = to_pa(pj)
    b1 = angle(pj,p1,p2)
    b2 = angle(pj,p1,p3)
    b3 = angle(pj,p2,p3)

    if(max(b1,b2,b3) == b3 ):
        a1 = b1
        a1_opposite = distance(p1,p2)
        a2 = b2
        a2_opposite = distance(p1,p3)
        p = distance(pj,p1)
        l_j1 = ((x-p1[0])**2+(y-p1[1])**2)**(1/2)
        l_j2 = ((x-p2[0])**2+(y-p2[1])**2)**(1/2)
        l_j3 = ((x-p3[0])**2+(y-p3[1])**2)**(1/2)


    if(max(b1,b2,b3) == b2 ):
        a1 = b1
        a1_opposite = distance(p1,p2)
        a2 = b3
        a2_opposite = distance(p2,p3)
        p = distance(pj,p2)
        l_j1 = ((x-p2[0])**2+(y-p2[1])**2)**(1/2)
        l_j2 = ((x-p1[0])**2+(y-p1[1])**2)**(1/2)
        l_j3 = ((x-p3[0])**2+(y-p3[1])**2)**(1/2)

    if(max(b1,b2,b3) == b1 ):
        a1 = b3
        a1_opposite = distance(p3,p2)
        a2 = b2
        a2_opposite = distance(p1,p3)
        p = distance(pj,p3)
        l_j1 = ((x-p3[0])**2+(y-p3[1])**2)**(1/2)
        l_j2 = ((x-p2[0])**2+(y-p2[1])**2)**(1/2)
        l_j3 = ((x-p1[0])**2+(y-p1[1])**2)**(1/2)

    if(a1 % 10 > 5):
        diff_a1 =  (a1 % 10 ) - 10
        a1_just = a1 -  diff_a1
    else:
        diff_a1 =  (a1 % 10 )
        a1_just = a1 -  diff_a1

    if(a2 % 10 > 5):
        diff_a2 = (a2 % 10 ) - 10 
        a2_just = a2 -  diff_a2
    else:
        diff_a2 =  (a2 % 10 )
        a2_just = a2 -  diff_a2

    cos_b1 = math.cos(a1_just/180*math.pi)
    cos_b2 = math.cos(a2_just/180*math.pi)

    initial = 104
    aa = nsolve([((l_j2**2+l_j1**2-a1_opposite**2)/(2*l_j2*l_j1))-cos_b1,
                ((l_j1**2+l_j3**2-a2_opposite**2)/(2*l_j1*l_j3)-cos_b2)],
               [x, y],[initial * math.cos(a /180 * math.pi) ,initial * math.sin(a /180 * math.pi)])

    print("初始直角坐标：",tuple(pj.round(2)),"初始极坐标：",tuple([round(i,2) for i in to_pa(pj)]),
        "调整前的角度1：",round(a1,2),"需要调整的度数：",round(diff_a1,2),"调整后的角度1：",round(a1_just,2),
        "调整前的角度2：",round(a2,2),"需要调整的度数：",round(diff_a2,2),"调整后的角度2：",round(a2_just,2),
        "调整后的极坐标：",tuple([round(i,2) for i in to_pa(aa)]))
    lst.append([pj.round(2),[round(i,2) for i in to_pa(pj)],round(a1,2),round(diff_a1,2),
        round(a1_just,2),round(a2,2),round(diff_a2,2),round(a2_just,2),[round(i,2) for i in to_pa(aa)]])
    return aa


print("第一次调整")
for i in range(len(location)):
    if(i==0 or i==4 or i==7):
        continue
    print("编号",i)
    location[i,0],location[i,1]=adjust(location[i],location[0], location[4], location[7])
lst = []

'''
for i in [5,6,8]:
    for j in [1,2,3,4,7,9]:
        for k in [1,2,3,4,7,9]:
            if(i==j or i==k or k==j or i==0):
                continue
            print(i,j,k) 
            _=adjust(location[i],location[0], location[j], location[k])
''' 
print()
print("第二次调整")
for i in range(len(location)):
    if(i==0 or i==1 or i==2):
        continue
    print("编号",i) 
    location[i,0],location[i,1]=adjust(location[i],location[0], location[1], location[2])

df = pd.DataFrame(lst)
df.columns=['初始直角坐标','初始极坐标','调整前的角度1','需要调整的度数',
'调整后的角度1','调整前的角度2','需要调整的度数','调整后的角度2','调整后的极坐标']
print(df)
