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

class MonoclinicLattice:
	def __init__(self):
		self.Position = {}

	def Axes(self,a_len,b_len,c_len,beta):
		""" Define the crystalographic properties of the lattice"""
		self.a = np.array([0,0,a_len])
		self.b = np.array([0,b_len,0])
		self.c = Ry(-beta) @ np.array([0,0,c_len])

	def gTensor(self,gx,gy,gz,zeta_a):
		""" Define the g-tensor of the crystal.
		Assumes a axial tensor in the AC plane
		"""
		self.gx = gx	
		self.gy = gy
		self.gz = gz
		# rotate the crystal coordinates so that I'm now in the coordinate system 
		# given by the zeeman tensor's principal axes
		self.a = Ry(zeta_a) @ self.a
		self.b = Ry(zeta_a) @ self.b
		self.c = Ry(zeta_a) @ self.c

	def Ion1Position(self,x,y,z):
		#Note Ive swapped a and c to match the niemiejer paper
		self.ion1 = x*self.a + y*self.b + z*self.c
		self.Position['1A'] = np.array([0,0,0]) + self.ion1
		self.Position['2A'] = self.c + self.ion1
		self.Position['3A'] = self.b + self.ion1
		self.Position['4A'] = self.c + self.b + self.ion1
		self.Position['5A'] = self.a + self.ion1
		self.Position['6A'] = self.a + self.c + self.ion1
		self.Position['7A'] = self.b + self.a + self.ion1
		self.Position['8A'] = self.a + self.b + self.c + self.ion1

	def Ion2Position(self,x,y,z):
		self.ion2 = x*self.a + y*self.b + z*self.c
		self.Position['1B'] = np.array([0,0,0]) + self.ion2
		self.Position['2B'] = self.a + self.ion2
		self.Position['3B'] = self.b + self.ion2
		self.Position['4B'] = self.a + self.b + self.ion2
		self.Position['5B'] = self.c + self.ion2
		self.Position['6B'] = self.a + self.c + self.ion2
		self.Position['7B'] = self.b + self.c + self.ion2
		self.Position['8B'] = self.a + self.b + self.c + self.ion2

	
	def LatticePoints(self,R):
		"""Generates a lattice with spacing twice the unit cell vectors, centred on (0,0,0)
		returns a list of lattice sites across a grid of size 2R x 2R x 2R, which will encompass
		the relevant sphere
		"""

		#Calculate the number of lattice points needed in each direction to cover a length of R
		#I use the ceiling function so that when I shift the origin by a one unit cell vector,
		#I still cover all lattive points within a distance of R
		Na = int(np.ceil(R/np.linalg.norm(2*self.a)))
		Nb = int(np.ceil(R/np.linalg.norm(2*self.b)))
		Nc = int(np.ceil(R/np.linalg.norm(2*self.c)))

		#calculate the number of vertices in a grid that covers the sphere
		#A sphere of radius R fits within a grid of size 2R x 2R x 2R
		#Adding one to account for origin
		number_vertices = (2*Na+1)*(2*Nb+1)*(2*Nc+1)
		vertices = np.empty((number_vertices,3))
		
		# populate the vertices list with the positions of a lattice with double spacing
		n = 0
		for i in np.arange(-Na,Na+1):   	
			for j in np.arange(-Nb,Nb+1):   
				for k in np.arange(-Nc,Nc+1):                  
					vertices[n]=2*np.dot([[i,j,k]],[[self.a[0],self.a[1],self.a[2]],[self.b[0],self.b[1],self.b[2]],[self.c[0],self.c[1],self.c[2]]])
					n += 1
		
		return vertices

	def ShiftAndTrim(self,vertices,R,iNumber,iLetter,jNumber,jLetter):
		"""
		Given a lattice, it shifts of origin to be centred on the jth ion
		Then return all latice points within R of the ith ion
		"""

		#Shift vertices to be the lattice generated the from the jth position
		vertices = vertices + self.Position[str(jNumber) + jLetter]
		#Calculate distances from the ith atom to each other atom
		distance = np.sqrt(np.sum(np.power(vertices - self.Position[str(iNumber) + iLetter],2),axis=1))
		#only keep the locations of which are within a distance R from ion i
		#I take the intersection with non-zero distances to avoid counting origin when ith and jth ions are equal
		vertices = vertices[(distance < R) & (distance != 0.0)]
		
		return vertices


	
		jLetter = 'A'
		while True:
			with open(jLetter+'MatrixValuesDyCl3New.csv', 'w') as csvfile:
				writer = csv.writer(csvfile, lineterminator = '\n')
				
				verticesbase = self.LatticePoints(R)
				
				for jNumber in np.arange(1,9): 
					
					iVector = self.Position['1A']

					vertices = self.ShiftAndTrim(verticesbase,R,1,'A',jNumber,jLetter)				
					
					x = vertices[:,0] - iVector[0]				
					y = vertices[:,1] - iVector[1]
					z = vertices[:,2] - iVector[2]
					r = np.sqrt(np.sum(np.power(vertices - iVector,2),axis=1))
					 
					xx = sum(((r**2-3*x**2)/r**5))
					yy = sum(((r**2-3*y**2)/r**5))
					zz = sum(((r**2-3*z**2)/r**5))
					xy = sum(((-3*x*y)/r**5))
					xz = sum(((-3*x*z)/r**5))
					yz = sum(((-3*y*z)/r**5)) 
			                       
					writer.writerow([xx,xy,xz])
					writer.writerow([xy,yy,yz])
					writer.writerow([xz,yz,zz])
			if jLetter == 'A':
				jLetter = 'B'
			else:
				break

	def WriteLatticeSums(self,R):
		jLetter = 'A'
		while True:
			with open(jLetter+'MatrixValuesDyCl3New.csv', 'w') as csvfile:
				writer = csv.writer(csvfile, lineterminator = '\n')
				
				verticesbase = self.LatticePoints(R)
				
				for jNumber in np.arange(1,9): 
					
					iVector = self.Position['1A']

					vertices = self.ShiftAndTrim(verticesbase,R,1,'A',jNumber,jLetter)				
					
					x = vertices[:,0] - iVector[0]				
					y = vertices[:,1] - iVector[1]
					z = vertices[:,2] - iVector[2]
					r = np.sqrt(np.sum(np.power(vertices - iVector,2),axis=1))
					 
					xx = sum(((r**2-3*x**2)/r**5))
					yy = sum(((r**2-3*y**2)/r**5))
					zz = sum(((r**2-3*z**2)/r**5))
					xy = sum(((-3*x*y)/r**5))
					xz = sum(((-3*x*z)/r**5))
					yz = sum(((-3*y*z)/r**5)) 
			                       
					writer.writerow([xx,xy,xz])
					writer.writerow([xy,yy,yz])
					writer.writerow([xz,yz,zz])
			if jLetter == 'A':
				jLetter = 'B'
			else:
				break


	def CreateAMatrix(self,factor,demagnetisation):
	    with open('AMatrixValuesDyCl3New.csv') as csvfile:
	        spamreader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
	        self.A = {}
	        n = 0
	        m = 11    
	        for row in spamreader:
	            if n == 0:
	                self.A[m] = np.zeros((3,3))
	                self.A[m][0,0] = ((row[0] + demagnetisation[0,0])*self.gx*self.gx)*factor
	                self.A[m][0,1] = ((row[1] + demagnetisation[0,1])*self.gx*self.gy)*factor
	                self.A[m][0,2] = ((row[2] + demagnetisation[0,2])*self.gx*self.gz)*factor
	                n += 1
	            elif n == 1:
	                self.A[m][1,0] = ((row[0] + demagnetisation[1,0])*self.gy*self.gx)*factor
	                self.A[m][1,1] = ((row[1] + demagnetisation[1,1])*self.gy*self.gy)*factor
	                self.A[m][1,2] = ((row[2] + demagnetisation[1,2])*self.gy*self.gz)*factor
	                n += 1
	            else:
	                self.A[m][2,0] = ((row[0] + demagnetisation[2,0])*self.gz*self.gx)*factor
	                self.A[m][2,1] = ((row[1] + demagnetisation[2,1])*self.gz*self.gy)*factor
	                self.A[m][2,2] = ((row[2] + demagnetisation[2,2])*self.gz*self.gz)*factor
	                n = 0
	                m += 1


	def CreateBMatrix(self,factor,demagnetisation):
	    with open('BMatrixValuesDyCl3New.csv') as csvfile:
	        spamreader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
	        self.B = {}
	        n = 0
	        m = 11    
	        for row in spamreader:
	            if n == 0:
	                self.B[m] = np.zeros((3,3))
	                self.B[m][0,0] = ((row[0] + demagnetisation[0,0])*self.gx*self.gx)*factor
	                self.B[m][0,1] = ((row[1] + demagnetisation[0,1])*self.gx*self.gy)*factor
	                self.B[m][0,2] = ((row[2] + demagnetisation[0,2])*self.gx*self.gz)*factor
	                n += 1
	            elif n == 1:
	                self.B[m][1,0] = ((row[0] + demagnetisation[1,0])*self.gy*self.gx)*factor
	                self.B[m][1,1] = ((row[1] + demagnetisation[1,1])*self.gy*self.gy)*factor
	                self.B[m][1,2] = ((row[2] + demagnetisation[1,2])*self.gy*self.gz)*factor
	                n += 1
	            else:
	                self.B[m][2,0] = ((row[0] + demagnetisation[2,0])*self.gz*self.gx)*factor
	                self.B[m][2,1] = ((row[1] + demagnetisation[2,1])*self.gz*self.gy)*factor
	                self.B[m][2,2] = ((row[2] + demagnetisation[2,2])*self.gz*self.gz)*factor
	                n = 0
	                m += 1

	def set_L_matrices(self):
		A = self.A
		B = self.B
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
		self.L = {}
		for key in LA:
			self.L[key] = LA[key] + LB[key]
		for key in LA:
			self.L[key+8] = LA[key] - LB[key]
	


	def CreateMatrices(self,factor,demagnetisation):
		DyCl3.CreateAMatrix(factor,demagnetisation)
		DyCl3.CreateBMatrix(factor,demagnetisation)
		DyCl3.set_L_matrices()


	def EigenVals(self):
		eigvals = {}
		for key in self.L:
			eigvals[key] = np.linalg.eigvals(self.L[key])
		with open('EigValsNew.csv', 'w') as csvfile:
			writer = csv.writer(csvfile, lineterminator = '\n')  
			for key in eigvals:                   
				writer.writerow(eigvals[key])
		# print(eigvals[1])

			


DyCl3 = MonoclinicLattice()
DyCl3.Axes(9.61, 6.49, 7.87, 93.65*pi/180)
DyCl3.gTensor(1.76, 1.76, 16.52, 157*pi/180)
DyCl3.Ion1Position(0.25, 0.1521, 0.25)
DyCl3.Ion2Position(0.75, 0.8479, 0.75)
DyCl3.WriteLatticeSums(100)
#SI UNITS
mB=9.274*10**(-24) 
k=1.380*10**(-23)
NA=6.022*10**23 
mu0=4*pi*10**(-7) 
factor = (mu0*mB**2)/(k*8*4*pi)/((10**(-10))**3)
demagnetisation = np.array([[-0.00106,0,0],[0,-0.00106,0],[0,0,-0.00106]])
demagnetisation =  np.array([[0,0,0],[0,0,0],[0,0,0]])

DyCl3.CreateMatrices(factor,demagnetisation)
DyCl3.EigenVals()
