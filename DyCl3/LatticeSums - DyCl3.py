import numpy as np
import csv
import matplotlib
pi = np.pi

path = '/home/matthew/Dropbox/University/Ordering Temperature Model/'
# path = '/Users/siobhantobin/Desktop/Matt_work/Ordering Temperature Model/'

#define rotation matrices
def Rx(theta):
    return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
def Ry(theta):
    return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
def Rz(theta):
    return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])

#I rotate everything by 67 degrees so that the axial component of the g tensor is now aligned with the z-axis
def cell_translation_vectors(a,b,c,beta):
    global vect_a,vect_b,vect_c
    vect_a = Ry(67*pi/180) @ np.array([a,0,0])
    vect_b = Ry(67*pi/180) @ np.array([0,-b,0])
    vect_c = Ry(67*pi/180) @ Ry(-beta) @ np.array([c,0,0])
           
def generate_lattice(a,b,c,R,posi,posj):
    global vertices
    #calculate how many ions you need in each direction of the unit cell you need to fill a sphere of radius R
    Na = int(np.ceil(R/np.sqrt(2*a[0]**2+2*a[1]**2+2*a[2]**2)))
    Nb = int(np.ceil(R/np.sqrt(2*b[0]**2+2*b[1]**2+2*b[2]**2)))
    Nc = int(np.ceil(R/np.sqrt(2*c[0]**2+2*c[1]**2+2*c[2]**2)))

    #create an empty array that we'll fill with the locations of the lattice vertices upon translations of the unit cell
    vertices=np.zeros(((2*Na+1)*(2*Nb+1)*(2*Nc+1),3)) 
    #fill the vertice s array with the locations
    n=0
    for i in np.arange(-Na,Na+1):   
        for j in np.arange(-Nb,Nb+1):   
            for k in np.arange(-Nc,Nc+1):                  
                vertices[n]=[i*2*a[0]+j*2*b[0]+k*2*c[0],i*2*a[1]+j*2*b[1]+k*2*c[1],i*2*a[2]+j*2*b[2]+k*2*c[2]]
                n+=1

    #translate the whole lattice so that it now lines up with locations of the j ion sites (j is basically always 1)
    vertices = vertices + posj

    #calculate the distances of each j ion from the i ion
    distance=np.sqrt(np.sum(np.power(vertices-posi,2),axis=1))
    #only keep the locations of j ions which are within a distance R from ion i
    vertices=vertices[(distance < R) & (distance != 0.0)]


#define the 16 ion positions in a magnetic unit cell
def IonPositions(a,b,c,i1,i2):
    global pos
    pos = {}
    pos[1] = np.array([0,0,0]) + i1
    pos[2] = a + i1
    pos[3] = b + i1
    pos[4] = a + b + i1
    pos[5] = c + i1
    pos[6] = a + c + i1
    pos[7] = b + c + i1
    pos[8] = a + b + c + i1
    pos[9] = np.array([0,0,0]) + i2
    pos[10] = a + i2
    pos[11] = b + i2
    pos[12] = a + b + i2
    pos[13] = c + i2
    pos[14] = a + c + i2
    pos[15] = b + c + i2
    pos[16] = a + b + c + i2

def WriteLatticeSums(letter,jPositions,iPosition,SumRadius):
    with open(path+letter+'MatrixValuesDyCl3.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, lineterminator = '\n')
        if letter == 'A':
            shifter = 0
        elif letter == 'B':
            shifter = 8
        else:
            print("Error! Make sure its 'A' or 'B'")

        for key in np.arange(1+shifter,9+shifter):   
            
            generate_lattice(vect_a,vect_b,vect_c,SumRadius,jPositions[key],iPosition)
            
            x = vertices[:,0] - jPositions[key][0]
            
            y = vertices[:,1] - jPositions[key][1]
            z = vertices[:,2] - jPositions[key][2]
            r = np.sqrt(np.sum(np.power(vertices-jPositions[key],2),axis=1))
            
            xx = sum(((r**2-3*x**2)/r**5))
            yy = sum(((r**2-3*y**2)/r**5))
            zz = sum(((r**2-3*z**2)/r**5))
            xy = sum(((-3*x*y)/r**5))
            xz = sum(((-3*x*z)/r**5))
            yz = sum(((-3*y*z)/r**5)) 
                        
            writer.writerow([xx,xy,xz])
            writer.writerow([xy,yy,yz])
            writer.writerow([xz,yz,zz])

def CreateAMatrix(gx,gy,gz,factor,demagxx,demagyy,demagzz):
    with open(path+'AMatrixValuesDyCl3.csv') as csvfile:
        spamreader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
        global A
        A = {}
        n = 0
        m = 11    
        for row in spamreader:
            if n == 0:
                A[m]=np.zeros((3,3))
                A[m][0,0]=row[0]*gx*gx*factor + demagxx
                A[m][0,1]=row[1]*gx*gy*factor
                A[m][0,2]=row[2]*gx*gz*factor
                n+=1
            elif n == 1:
                A[m][1,0]=row[0]*gy*gx*factor
                A[m][1,1]=row[1]*gy*gy*factor + demagyy
                A[m][1,2]=row[2]*gy*gz*factor
                n+=1
            else:
                A[m][2,0]=row[0]*gz*gx*factor
                A[m][2,1]=row[1]*gz*gy*factor
                A[m][2,2]=row[2]*gz*gz*factor + demagzz
                n=0
                m+=1

def CreateAMatrixPaper():
    with open(path+'AMatrixValuesDyCl3PaperValues.csv') as csvfile:
        spamreader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
        global APaper
        APaper = {}
        n = 0
        m = 11    
        for row in spamreader:
            if n == 0:
                APaper[m]=np.zeros((3,3))
                APaper[m][0,0]=row[0]
                APaper[m][0,1]=row[1]
                APaper[m][0,2]=row[2]
                n+=1
            elif n == 1:
                APaper[m][1,0]=row[0]
                APaper[m][1,1]=row[1]
                APaper[m][1,2]=row[2]
                n+=1
            else:
                APaper[m][2,0]=row[0]
                APaper[m][2,1]=row[1]
                APaper[m][2,2]=row[2]
                n=0
                m+=1

def CreateBMatrixPaper():
    with open(path+'BMatrixValuesDyCl3PaperValues.csv') as csvfile:
        spamreader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
        global BPaper
        BPaper = {}
        n = 0
        m = 11    
        for row in spamreader:
            if n == 0:
                BPaper[m]=np.zeros((3,3))
                BPaper[m][0,0]=row[0]
                BPaper[m][0,1]=row[1]
                BPaper[m][0,2]=row[2]
                n+=1
            elif n == 1:
                BPaper[m][1,0]=row[0]
                BPaper[m][1,1]=row[1]
                BPaper[m][1,2]=row[2]
                n+=1
            else:
                BPaper[m][2,0]=row[0]
                BPaper[m][2,1]=row[1]
                BPaper[m][2,2]=row[2]
                n=0
                m+=1

def CreateBMatrix(gx,gy,gz,factor,demagxx,demagyy,demagzz):
    with open(path+'BMatrixValuesDyCl3.csv') as csvfile:
        spamreader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
        global B
        B = {}
        n = 0
        m = 11    
        for row in spamreader:
            if n == 0:
                B[m]=np.zeros((3,3))
                B[m][0,0]=row[0]*gx*gx*factor + demagxx
                B[m][0,1]=row[1]*gx*gy*factor
                B[m][0,2]=row[2]*gx*gz*factor
                n+=1
            elif n == 1:
                B[m][1,0]=row[0]*gy*gx*factor
                B[m][1,1]=row[1]*gy*gy*factor + demagyy
                B[m][1,2]=row[2]*gy*gz*factor
                n+=1
            else:
                B[m][2,0]=row[0]*gz*gx*factor
                B[m][2,1]=row[1]*gz*gy*factor
                B[m][2,2]=row[2]*gz*gz*factor + demagzz
                n=0
                m+=1

def set_L_matrices():
    global L
    LA = {}
    LA[1] = A[11]+A[12]+A[13]+A[14]+A[15]+A[16]+A[17]+A[18]
    LA[2] = A[11]-A[12]-A[13]+A[14]-A[15]+A[16]+A[17]-A[18]
    LA[3] = A[11]+A[12]-A[13]-A[14]+A[15]+A[16]-A[17]-A[18]
    LA[4] = A[11]+A[12]+A[13]+A[14]-A[15]-A[16]-A[17]-A[18]
    LA[5] = A[11]-A[12]+A[13]-A[14]+A[15]-A[16]+A[17]-A[18]
    LA[6] = A[11]+A[12]-A[13]-A[14]-A[15]-A[16]+A[17]+A[18]
    LA[7] = A[11]-A[12]-A[13]+A[14]+A[15]-A[16]-A[17]+A[18]
    LA[8] = A[11]-A[12]+A[13]-A[14]-A[15]+A[16]-A[17]+A[18]
    LB = {}
    LB[1] = B[11]+B[12]+B[13]+B[14]+B[15]+B[16]+B[17]+B[18]
    LB[2] = B[11]-B[12]-B[13]+B[14]-B[15]+B[16]+B[17]-B[18]
    LB[3] = B[11]+B[12]-B[13]-B[14]+B[15]+B[16]-B[17]-B[18]
    LB[4] = B[11]+B[12]+B[13]+B[14]-B[15]-B[16]-B[17]-B[18]
    LB[5] = B[11]-B[12]+B[13]-B[14]+B[15]-B[16]+B[17]-B[18]
    LB[6] = B[11]+B[12]-B[13]-B[14]-B[15]-B[16]+B[17]+B[18]
    LB[7] = B[11]-B[12]-B[13]+B[14]+B[15]-B[16]-B[17]+B[18]
    LB[8] = B[11]-B[12]+B[13]-B[14]-B[15]+B[16]-B[17]+B[18]
    L = {}
    for key in LA:
        L[key] = LA[key] + LB[key]
    for key in LA:
        L[key+8] = LA[key] - LB[key]

def set_L_matrices_Paper():
    global LPaper
    LAPaper = {}
    LAPaper[1] = APaper[11]+APaper[12]+APaper[13]+APaper[14]+APaper[15]+APaper[16]+APaper[17]+APaper[18]
    LAPaper[2] = APaper[11]-APaper[12]-APaper[13]+APaper[14]-APaper[15]+APaper[16]+APaper[17]-APaper[18]
    LAPaper[3] = APaper[11]+APaper[12]-APaper[13]-APaper[14]+APaper[15]+APaper[16]-APaper[17]-APaper[18]
    LAPaper[4] = APaper[11]+APaper[12]+APaper[13]+APaper[14]-APaper[15]-APaper[16]-APaper[17]-APaper[18]
    LAPaper[5] = APaper[11]-APaper[12]+APaper[13]-APaper[14]+APaper[15]-APaper[16]+APaper[17]-APaper[18]
    LAPaper[6] = APaper[11]+APaper[12]-APaper[13]-APaper[14]-APaper[15]-APaper[16]+APaper[17]+APaper[18]
    LAPaper[7] = APaper[11]-APaper[12]-APaper[13]+APaper[14]+APaper[15]-APaper[16]-APaper[17]+APaper[18]
    LAPaper[8] = APaper[11]-APaper[12]+APaper[13]-APaper[14]-APaper[15]+APaper[16]-APaper[17]+APaper[18]
    LBPaper = {}
    LBPaper[1] = BPaper[11]+BPaper[12]+BPaper[13]+BPaper[14]+BPaper[15]+BPaper[16]+BPaper[17]+BPaper[18]
    LBPaper[2] = BPaper[11]-BPaper[12]-BPaper[13]+BPaper[14]-BPaper[15]+BPaper[16]+BPaper[17]-BPaper[18]
    LBPaper[3] = BPaper[11]+BPaper[12]-BPaper[13]-BPaper[14]+BPaper[15]+BPaper[16]-BPaper[17]-BPaper[18]
    LBPaper[4] = BPaper[11]+BPaper[12]+BPaper[13]+BPaper[14]-BPaper[15]-BPaper[16]-BPaper[17]-BPaper[18]
    LBPaper[5] = BPaper[11]-BPaper[12]+BPaper[13]-BPaper[14]+BPaper[15]-BPaper[16]+BPaper[17]-BPaper[18]
    LBPaper[6] = BPaper[11]+BPaper[12]-BPaper[13]-BPaper[14]-BPaper[15]-BPaper[16]+BPaper[17]+BPaper[18]
    LBPaper[7] = BPaper[11]-BPaper[12]-BPaper[13]+BPaper[14]+BPaper[15]-BPaper[16]-BPaper[17]+BPaper[18]
    LBPaper[8] = BPaper[11]-BPaper[12]+BPaper[13]-BPaper[14]-BPaper[15]+BPaper[16]-BPaper[17]+BPaper[18]
    LPaper = {}
    for key in LAPaper:
        LPaper[key] = LAPaper[key] + LBPaper[key]
    for key in LAPaper:
        LPaper[key+8] = LAPaper[key] - LBPaper[key]


#define location of ions within unit cell
cell_translation_vectors(9.61,6.49,7.87,93.65*pi/180)
# print(vect_a)
# print(vect_b)
# print(vect_c)
ion1 = (1/4)*vect_a + (0.1521)*vect_b + (1/4)*vect_c
ion2 = (3/4)*vect_a + (0.8479)*vect_b + (3/4)*vect_c
IonPositions(vect_a,vect_b,vect_c,ion1,ion2)
#print(np.linalg.norm(vect_b+vect_c-(0.5*vect_a+1.3042*vect_b+0.5*vect_c)))
#print(np.linalg.norm(vect_b-(0.5*vect_a+1.3042*vect_b+0.5*vect_c)))
#print(np.linalg.norm((0.5*vect_a+1.3042*vect_b+0.5*vect_c)-(0.5*vect_a+0.3042*vect_b+0.5*vect_c)))

WriteLatticeSums('A',pos,pos[1],100)
WriteLatticeSums('B',pos,pos[1],100)

gpara = 16.52
gperp = 1.76
# ### DELETE NEXT TWO LINES WHEN NOT USING *PaperValues.csv
# # gpara = 1
# # gperp = 1
gx = gperp
gy = gperp
gz = gpara

#SI UNITS
mB=9.274*10**(-24) 
k=1.380*10**(-23)
NA=6.022*10**23 
mu0=4*pi*10**(-7) 
factor = (mu0*mB**2)/(k*8*4*pi)/((10**(-10))**3)
# # factor=1
# # generate_lattice(vect_a,vect_b,vect_c,20,pos[2],pos[1])

# # generate_lattice(vect_a,vect_b,vect_c,100,pos[13],pos[1])




CreateBMatrix(gx,gy,gz,factor,0,0,0)
CreateAMatrix(gx,gy,gz,factor,0,0,0)
set_L_matrices()
CreateBMatrixPaper()
CreateAMatrixPaper()
set_L_matrices_Paper()   
eigvals = {}
for key in L:
    eigvals[key] = np.linalg.eigvals(L[key])
with open(path+'/EigVals.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, lineterminator = '\n')  
    for key in eigvals:                   
        writer.writerow(eigvals[key])
    
# # dist = np.zeros((360,360,360))
# # i=0
# # j=0
# # k=0
# # for alpha in np.linspace(0,359,360):
# #     for beta in np.linspace(0,179.5,360):
# #         for gamma in np.linspace(0,359,360):
        
# #             dist[i,j,k]=(np.sum(np.square(Rz(alpha)@Ry(beta)@Rz(gamma)@L[5]@Rz(-gamma)@Ry(-beta)@Rz(-alpha)-LPaper[5])))
# #             k += 1
# #         j += 1
# #         k=0
# #     j=0
# #     i+=1
# #     print(i)
# # print(np.min(dist))
# # print(np.average(dist))

# # # def nearest_neighbours(a,b,c,pos):
# # #     neighbours=np.zeros(((5*5*5),3))  
# # #     n=0   
# # #     for i in np.arange(-2,3):   
# # #         for j in np.arange(-2,3):   
# # #             for k in np.arange(-2,3):                  
# # #                 neighbours[n]=[i*a[0]+j*b[0]+k*c[0],i*a[1]+j*b[1]+k*c[1],i*a[2]+j*b[2]+k*c[2]]
# # #                 n+=1
# # #     neighbours = neighbours+pos    
# # #     distance=np.sqrt(np.sum(np.power(neighbours,2),axis=1))
# # #     print(np.count_nonzero(distance == min(distance[np.nonzero(distance)])))
# # #     
# # # # nearest_neighbours(2*pos2,2*pos3,2*pos5,pos1)
