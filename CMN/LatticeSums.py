import numpy as np
import csv
import math
pi = np.pi
#from scipy.optimize import root
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#define rotation matrices
def Rx(theta):
    return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
def Ry(theta):
    return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
def Rz(theta):
    return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    
def find_translation_vectors(a,b,c,alpha,beta,gamma):
    global pos1, pos2, pos3
    pos1 = [a,0,0]
    pos2 = Rz(pi-gamma)@[b,0,0]
    pos3 = Rz(pi-gamma)@ Rx(pi/2-beta)@ Ry(alpha-pi)@[c,0,0]
    print(pos1,pos2,pos3)
  
# find_translation_vectors(9.61,6.49,7.87,pi/2,93.65*pi/180,pi/2)



def generate_lattice(a,b,c,R,pos):
    global vertices
    Na = int(np.ceil(R/np.sqrt(a[0]**2+a[1]**2+a[2]**2)))
    Nb = int(np.ceil(R/np.sqrt(b[0]**2+b[1]**2+b[2]**2)))
    Nc = int(np.ceil(R/np.sqrt(c[0]**2+c[1]**2+c[2]**2)))
    
    vertices=np.zeros(((2*Na+1)*(2*Nb+1)*(2*Nc+1),3)) 
    
    n=0
    for i in np.arange(-Na,Na+1):   
        for j in np.arange(-Nb,Nb+1):   
            for k in np.arange(-Nc,Nc+1):                  
                vertices[n]=[i*a[0]+j*b[0]+k*c[0],i*a[1]+j*b[1]+k*c[1],i*a[2]+j*b[2]+k*c[2]]
                n+=1    
    vertices = vertices+pos
    distance=np.sqrt(np.sum(np.power(vertices,2),axis=1))
    vertices=vertices[(distance < R) & (distance != 0.0)]
    

#define the lattice positions        
pos1 = np.array([0,0,0])
pos2 = np.array([+11.0/np.sqrt(3)*(np.sqrt(3)/2),11.0/np.sqrt(3)*(1/2),17.3/3])
pos3 = np.array([-11.0/np.sqrt(3)*(np.sqrt(3)/2),11.0/np.sqrt(3)*(1/2),17.3/3])
pos4 = pos2 + pos3
pos5 = np.array([0,-11.0/np.sqrt(3),17.3/3])
pos6 = pos2 + pos5
pos7 = pos3 + pos5
pos8 = pos2 + pos3 + pos5
positions = [pos1,pos2,pos3,pos4,pos5,pos6,pos7,pos8]


with open('MatrixValuesCMN.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, lineterminator = '\n')  
    for pos in positions:              
        generate_lattice(2*pos2,2*pos3,2*pos5,20,pos)
        x = vertices[:,0]/20
        # print(x)
        y = vertices[:,1]/20
        z = vertices[:,2]/20
        r = np.sqrt(np.sum(np.power(vertices/20,2),axis=1))
        
        xx = sum(((r**2-3*x**2)/r**5))
        yy = sum(((r**2-3*y**2)/r**5))
        zz = sum(((r**2-3*z**2)/r**5))
        xy = sum(((-3*x*y)/r**5))
        xz = sum(((-3*x*z)/r**5))
        yz = sum(((-3*y*z)/r**5)) 
                   
        writer.writerow([xx,xy,xz])
        writer.writerow([xy,yy,yz])
        writer.writerow([xz,yz,zz])

gpara = 0
gperp = 1.84
g = {}
g[0,0] = gperp**2
g[0,1] = gperp**2
g[0,2] = gperp*gpara
g[1,0] = gperp**2
g[1,1] = gperp**2
g[1,2] = gperp*gpara
g[2,0] = gperp*gpara
g[2,1] = gperp*gpara
g[2,2] = gpara**2

with open('MatrixValuesCMN.csv') as csvfile:
    spamreader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
    A = {}
    n = 0
    m = 11    
    for row in spamreader:
        if n == 0:
            A[m]=np.zeros((3,3))
            A[m][0,0]=row[0]*gperp**2
            A[m][0,1]=row[1]*gperp**2
            A[m][0,2]=row[2]*gperp*gpara
            n+=1
        elif n == 1:
            A[m][1,0]=row[0]*gperp**2
            A[m][1,1]=row[1]*gperp**2
            A[m][1,2]=row[2]*gperp*gpara
            n+=1
        else:
            A[m][2,0]=row[0]*gperp*gpara
            A[m][2,1]=row[1]*gperp*gpara
            A[m][2,2]=row[2]*gpara*gpara
            n=0
            m+=1

rho = 2.73
rho = 1/(np.dot(2*pos2/20,np.cross([2*pos3/20],[2*pos5/20])[0]))
# print(rho)
rho = 1/((3*np.sqrt(3.0)/2)*(11./20)**2*(17.3/20))
# print(rho)

def set_L_matrices():
    global L
    L = {}
    L[1] = A[11]+A[12]+A[13]+A[14]+A[15]+A[16]+A[17]+A[18]+[[8*2*pi/3*rho*gperp**2,0,0],[0,8*2*pi/3*rho*gperp**2,0],[0,0,8*-4*pi/3*rho*gpara**2]]
    L[2] = A[11]-A[12]-A[13]+A[14]-A[15]+A[16]+A[17]-A[18]
    L[3] = A[11]+A[12]-A[13]-A[14]+A[15]+A[16]-A[17]-A[18]
    L[4] = A[11]+A[12]+A[13]+A[14]-A[15]-A[16]-A[17]-A[18]
    L[5] = A[11]-A[12]+A[13]-A[14]+A[15]-A[16]+A[17]-A[18]
    L[6] = A[11]+A[12]-A[13]-A[14]-A[15]-A[16]+A[17]+A[18]
    L[7] = A[11]-A[12]-A[13]+A[14]+A[15]-A[16]-A[17]+A[18]
    L[8] = A[11]-A[12]+A[13]-A[14]-A[15]+A[16]-A[17]+A[18]
             
set_L_matrices()   

#SI UNITS
mB=9.274*10**(-24) 
k=1.380*10**(-23)
NA=6.022*10**23 
pi=3.14159
mu0=4*pi*10**(-7) 
print(np.linalg.eig(L[1]*(1000*mu0*mB**2)/(k*4*4*2*pi)/((20*10**(-10))**3))[0]*8)
print(np.linalg.eig(L[2]*(1000*mu0*mB**2)/(k*4*4*2*pi)/((20*10**(-10))**3))[0]*8)
print(np.linalg.eig(L[3]*(1000*mu0*mB**2)/(k*4*4*2*pi)/((20*10**(-10))**3))[0]*8)
print(np.linalg.eig(L[4]*(1000*mu0*mB**2)/(k*4*4*2*pi)/((20*10**(-10))**3))[0]*8)
print(np.linalg.eig(L[5]*(1000*mu0*mB**2)/(k*4*4*2*pi)/((20*10**(-10))**3))[0]*8)
print(np.linalg.eig(L[6]*(1000*mu0*mB**2)/(k*4*4*2*pi)/((20*10**(-10))**3))[0]*8)
print(np.linalg.eig(L[7]*(1000*mu0*mB**2)/(k*4*4*2*pi)/((20*10**(-10))**3))[0]*8)
print(np.linalg.eig(L[8]*(1000*mu0*mB**2)/(k*4*4*2*pi)/((20*10**(-10))**3))[0]*8)

def nearest_neighbours(a,b,c,pos):
    neighbours=np.zeros(((5*5*5),3))  
    n=0   
    for i in np.arange(-2,3):   
        for j in np.arange(-2,3):   
            for k in np.arange(-2,3):                  
                neighbours[n]=[i*a[0]+j*b[0]+k*c[0],i*a[1]+j*b[1]+k*c[1],i*a[2]+j*b[2]+k*c[2]]
                n+=1
    neighbours = neighbours+pos    
    distance=np.sqrt(np.sum(np.power(neighbours,2),axis=1))
    print(np.count_nonzero(distance == min(distance[np.nonzero(distance)])))
    
# nearest_neighbours(2*pos2,2*pos3,2*pos5,pos1)

