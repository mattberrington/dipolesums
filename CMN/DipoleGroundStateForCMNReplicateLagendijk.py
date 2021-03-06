import time
import numpy as np
import csv
import matplotlib
pi = np.pi

#define rotation matrices
def Rx(theta):
    return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
def Ry(theta):
    return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
def Rz(theta):
    return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])


class Lattice:
	def __init__(self,a_len,b_len,c_len,beta):
		self.a = np.array([+11.0/2,11.0/np.sqrt(3)*(1/2),17.3/3])
		self.b = np.array([-11.0/2,11.0/np.sqrt(3)*(1/2),17.3/3])
		self.c = np.array([0,-11.0/np.sqrt(3),17.3/3])

	def Ion1Position(self,x,y,z):
		self.ion1 = x*self.a + y*self.b + z*self.c

	def Position(self,ith_ion):
		pos = {1 : np.array([0,0,0]) + self.ion1,
			2 : self.a + self.ion1,
			3 : self.b + self.ion1,
			4 : self.a + self.b + self.ion1,
			5 : self.c + self.ion1,
			6 : self.a + self.c + self.ion1,
			7 : self.b + self.c + self.ion1,
			8 : self.a + self.b + self.c + self.ion1}
		return pos[ith_ion]

	
	def LatticePoints(self,R):
		
		Na = int(np.ceil(R/np.sqrt(2*self.a[0]**2+2*self.a[1]**2+2*self.a[2]**2)))
		Nb = int(np.ceil(R/np.sqrt(2*self.b[0]**2+2*self.b[1]**2+2*self.b[2]**2)))
		Nc = int(np.ceil(R/np.sqrt(2*self.c[0]**2+2*self.c[1]**2+2*self.c[2]**2)))
		
		vertices = np.zeros(((2*Na+1)*(2*Nb+1)*(2*Nc+1),3))
		
		n=0
		for i in np.arange(-Na,Na+1):   	
			for j in np.arange(-Nb,Nb+1):   
				for k in np.arange(-Nc,Nc+1):                  
					vertices[n]=[i*2*self.a[0]+j*2*self.b[0]+k*2*self.c[0],i*2*self.a[1]+j*2*self.b[1]+k*2*self.c[1],i*2*self.a[2]+j*2*self.b[2]+k*2*self.c[2]]
					n+=1
		# for i in np.arange(-Na,Na+1):   	
		# 	for j in np.arange(-Nb,Nb+1):   
		# 		for k in np.arange(-Nc,Nc+1):                  
		# 			vertices[n]=np.dot([[i,j,k]],[[self.a[0],self.a[1],self.a[2]],[self.b[0],self.b[1],self.b[2]],[self.c[0],self.c[1],self.c[2]]])
		# 			n+=1

		
		return vertices/2.0

	def ShiftAndTrim(self,vertices,R,iPosition,jPosition):
		vertices = vertices + self.Position(jPosition)
		# print(self.Position(jPosition))
		# print(jPosition)
		# print(vertices)

		distance = np.sqrt(np.sum(np.power(vertices-self.Position(iPosition),2),axis=1)) 

		#only keep the locations of j ions which are within a distance R from ion i
		vertices = vertices[(distance < R) & (distance != 0.0)]
		return vertices

	def S_terms(self,SumRadius):
	
			
		verticesbase = self.LatticePoints(SumRadius)

				
		vertices = self.ShiftAndTrim(verticesbase,SumRadius,1,1)				
			
		x = vertices[:,0] 			
		y = vertices[:,1] 
		z = vertices[:,2] 
		r = np.sqrt(np.sum(np.power(vertices,2),axis=1))
		 
		S1 = sum(1/r**6)
		S2 = sum(z**2/r**8)
		S3 = sum(z**4/r**10)
		S4 = sum(1/r**9)


		print(S1*10**8)
		print(S2*10**8)
		print(S3*10**8)
		print(S4*10**8)



	def gTensor(self,gx,gy,gz):
		self.gx = gx
		self.gy = gy
		self.gz = gz

	def CreateAMatrix(self,factor):
	    with open('MatrixValuesCMNNew.csv') as csvfile:
	        spamreader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
	        self.A = {}
	        n = 0
	        m = 11    
	        for row in spamreader:
	            if n == 0:
	                self.A[m]=np.zeros((3,3))
	                self.A[m][0,0]=row[0]*self.gx*self.gx*factor
	                self.A[m][0,1]=row[1]*self.gx*self.gy*factor
	                self.A[m][0,2]=row[2]*self.gx*self.gz*factor
	                n+=1
	            elif n == 1:
	                self.A[m][1,0]=row[0]*self.gy*self.gx*factor
	                self.A[m][1,1]=row[1]*self.gy*self.gy*factor
	                self.A[m][1,2]=row[2]*self.gy*self.gz*factor
	                n+=1
	            else:
	                self.A[m][2,0]=row[0]*self.gz*self.gx*factor
	                self.A[m][2,1]=row[1]*self.gz*self.gy*factor
	                self.A[m][2,2]=row[2]*self.gz*self.gz*factor
	                n=0
	                m+=1


	def set_L_matrices(self):
		A = self.A
		self.L = {}
		self.L[1] = A[11]+A[12]+A[13]+A[14]+A[15]+A[16]+A[17]+A[18]
		self.L[2] = A[11]-A[12]-A[13]+A[14]-A[15]+A[16]+A[17]-A[18]
		self.L[3] = A[11]+A[12]-A[13]-A[14]+A[15]+A[16]-A[17]-A[18]
		self.L[4] = A[11]+A[12]+A[13]+A[14]-A[15]-A[16]-A[17]-A[18]
		self.L[5] = A[11]-A[12]+A[13]-A[14]+A[15]-A[16]+A[17]-A[18]
		self.L[6] = A[11]+A[12]-A[13]-A[14]-A[15]-A[16]+A[17]+A[18]
		self.L[7] = A[11]-A[12]-A[13]+A[14]+A[15]-A[16]-A[17]+A[18]
		self.L[8] = A[11]-A[12]+A[13]-A[14]-A[15]+A[16]-A[17]+A[18]

	def CreateMatrices(self,factor):
		DyCl3.CreateAMatrix(factor)

		DyCl3.set_L_matrices()


	def EigenVals(self):
		eigvals = {}
		for key in self.L:
			eigvals[key] = np.linalg.eigvals(self.L[key])
		with open('EigValsCMNNew.csv', 'w') as csvfile:
			writer = csv.writer(csvfile, lineterminator = '\n')  
			for key in eigvals:                   
				writer.writerow(eigvals[key])

			


DyCl3 = Lattice(0,0,0,0)
DyCl3.Ion1Position(0,0,0)
DyCl3.gTensor(1.84,1.84,0)
DyCl3.S_terms(80)
#SI UNITS
mB=9.274*10**(-24) 
k=1.380*10**(-23)
NA=6.022*10**23 
mu0=4*pi*10**(-7) 
factor = (mu0*mB**2)/(k*8*4*pi)/((10**(-10))**3)
DyCl3.CreateMatrices(1)
DyCl3.EigenVals()
