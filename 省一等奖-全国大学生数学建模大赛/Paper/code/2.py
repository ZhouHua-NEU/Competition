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

    if(pj[0]==0 and pj[1]==0):
        pa,at=0,0

    if(pj[0] > 0 and pj[1] > 0):
        pt,at= (pj[0] ** 2 + pj[1] ** 2)**(1/2),math.atan(pj[1]/pj[0])/math.pi *180.0

    if(pj[0] < 0 and pj[1] > 0):
        pt,at= (pj[0] ** 2 + pj[1] ** 2)**(1/2), math.atan(pj[1]/pj[0])/math.pi *180.0 + 180

    if(pj[0] < 0 and pj[1] < 0):
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
    v=math.acos((a*a-b*b-c*c)/(-2*b*c))
    A=math.degrees(v)
    return A



#直角坐标系
q=math.sqrt(3)
location =   np.array([[0,-100],[0,-50],[0,0],[0,50],[0,100],[25*q,-75],[25*q,-25],[25*q,25],[25*q,75],[50*q,-50],[50*q,0],[50*q,50],[75*q,-25],[75*q,25],[100*q,0]])
real  =     np.array([[0,-100],[0,-50],[0,0],[0,50],[0,100],[25*q,-75],[25*q,-25],[25*q,25],[25*q,75],[50*q,-50],[50*q,0],[50*q,50],[75*q,-25],[75*q,25],[100*q,0]])
real_zhui_angle = real
np.random.seed(0)
zaoyin =  np.random.randint(low= -60 ,high= 60,size=(15,2)) * 0.1
location = location + zaoyin
location[[6,7,10]]=real[[6,7,10]]
print(location)

  
from sympy import symbols,nsolve
x, y = symbols('x, y')
lst=[]

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

    if(a1< 20):
        
        if(a1 < 10):
            a1_just = 0
            diff_a1 = a1
        else:
            a1_just = 19.10660535086906
            diff_a1 =19.10660535086906- a1
    if(a2< 20):
        if(a2 < 10):
            a2_just = 0
            diff_a2 = a2
        else:
            a2_just = 19.10660535086906
            diff_a2 =19.10660535086906 - a2
            
    if(a1>=20):
        if(a1 % 10 > 5):
            diff_a1 =  (a1 % 10 ) - 10
            a1_just = a1 -  diff_a1
        else:
            diff_a1 =  (a1 % 10 )
            a1_just = a1 -  diff_a1
    if(a2>=20):
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
               [x, y],(pj[0] ,pj[1]))

    print("初始直角坐标：",tuple(pj.round(2)),"初始极坐标：",tuple([round(i,2) for i in to_pa(pj)]),
        "调整前的角度1：",round(a1,2),"需要调整的度数：",round(diff_a1,2),"调整后的角度1：",round(a1_just,2),
        "调整前的角度2：",round(a2,2),"需要调整的度数：",round(diff_a2,2),"调整后的角度2：",round(a2_just,2),
        "调整后的极坐标：",tuple([round(i,2) for i in to_pa(aa)]))

    lst.append([pj.round(2),[round(i,2) for i in to_pa(pj)],round(a1,2),round(diff_a1,2),
        round(a1_just,2),round(a2,2),round(diff_a2,2),round(a2_just,2),[round(i,2) for i in to_pa(aa)]])
    return aa

print(angle(real[6],real[7],real[4]))

for i in range(len(location)):
    print(i)
    if(i==6 or i==7 or i==10):
        print(location[i,0],location[i,1])
    else:
        location[i,0],location[i,1]=adjust(location[i],location[6], location[7], location[10])

'''
for i in [5,6,8]:
    for j in [1,2,3,4,7,9]:
        for k in [1,2,3,4,7,9]:
            if(i==j or i==k or k==j or i==0):
                continue
            print(i,j,k) 
            _=adjust(location[i],location[0], location[j], location[k])


for i in range(len(location)):
    if(i==0 or i==1 or i==2):
        continue
    print(i) 
    location[i,0],location[i,1]=adjust(location[i],location[0], location[1], location[2])
''' 


df = pd.DataFrame(lst)
df.columns=['初始直角坐标','初始极坐标','调整前的角度1','需要调整的度数',
'调整后的角度1','调整前的角度2','需要调整的度数','调整后的角度2','调整后的极坐标']
print(df)
