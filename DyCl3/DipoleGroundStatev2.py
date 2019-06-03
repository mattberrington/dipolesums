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
		self.position = {}

	def axes(self,a_len,b_len,c_len,beta):
		""" Define the crystalographic properties of the lattice"""
		self.a = np.array([0,0,a_len])
		self.b = np.array([0,b_len,0])
		self.c = Ry(-beta) @ np.array([0,0,c_len])

	def g_tensor(self,gx,gy,gz,zeta_a):
		""" Define the g-tensor of the crystal.
		Assumes a axial tensor in the AC plane
		"""
		self.gx = gx	
		self.gy = gy
		self.gz = gz
		self.g_grid = np.array([[gx*gx, gx*gy, gx*gz],[gy*gx, gy*gy, gy*gz],[gz*gx, gz*gy, gz*gz]])
		# rotate the crystal coordinates so that I'm now in the coordinate system 
		# given by the zeeman tensor's principal axes
		self.a = Ry(zeta_a) @ self.a
		self.b = Ry(zeta_a) @ self.b
		self.c = Ry(zeta_a) @ self.c

	def ion1_position(self,x,y,z):
		#Note Ive swapped a and c to match the niemiejer paper
		self.ion1 = x*self.a + y*self.b + z*self.c
		self.position['1A'] = np.array([0,0,0]) + self.ion1
		self.position['2A'] = self.c + self.ion1
		self.position['3A'] = self.b + self.ion1
		self.position['4A'] = self.c + self.b + self.ion1
		self.position['5A'] = self.a + self.ion1
		self.position['6A'] = self.a + self.c + self.ion1
		self.position['7A'] = self.b + self.a + self.ion1
		self.position['8A'] = self.a + self.b + self.c + self.ion1

	def ion2_position(self,x,y,z):
		self.ion2 = x*self.a + y*self.b + z*self.c
		self.position['1B'] = np.array([0,0,0]) + self.ion2
		self.position['2B'] = self.a + self.ion2
		self.position['3B'] = self.b + self.ion2
		self.position['4B'] = self.a + self.b + self.ion2
		self.position['5B'] = self.c + self.ion2
		self.position['6B'] = self.a + self.c + self.ion2
		self.position['7B'] = self.b + self.c + self.ion2
		self.position['8B'] = self.a + self.b + self.c + self.ion2

	
	def lattice_points(self,R):
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

	def lattice_sphere(self,R,iNumber,iLetter,jNumber,jLetter):
		"""
		Returns the lattice points of a lattice centred on the jth ion, for all ions within R of the ith ion
		"""
		vertices = self.lattice_points(R)
		#Shift vertices to be the lattice generated from the jth position
		vertices = vertices + self.position[str(jNumber) + jLetter]
		#Calculate distances from the ith atom to each other atom
		distance = np.sqrt(np.sum(np.power(vertices - self.position[str(iNumber) + iLetter],2),axis=1))
		#only keep the locations of which are within a distance R from ion i
		#I take the intersection with non-zero distances to avoid counting origin when ith and jth ions are equal
		vertices = vertices[(distance < R) & (distance != 0.0)]
		
		return vertices

	def lattice_sums(self,R):
		#SI UNITS
		mB=9.274*10**(-24) 
		k=1.380*10**(-23)
		NA=6.022*10**23 
		mu0=4*pi*10**(-7) 			
		factor = (mu0*mB**2)/(32*k*pi)/((10**(-10))**3)

		A = {}
		B = {}

		for jNumber in np.arange(1,9):

			#obtain a sphere of lattice points of radius R centred on ion position 1A				
			iVector = self.position['1A']				
			vertices = self.lattice_sphere(R,1,'A',jNumber,'A')				
			
			#calculate the relative position of every lattice point
			x = vertices[:,0] - iVector[0]				
			y = vertices[:,1] - iVector[1]
			z = vertices[:,2] - iVector[2]
			r = np.sqrt(np.sum(np.power(vertices - iVector,2),axis=1))
			
			#calculate lattice sums
			xx = sum(((r**2-3*x**2)/r**5))
			yy = sum(((r**2-3*y**2)/r**5))
			zz = sum(((r**2-3*z**2)/r**5))
			xy = sum(((-3*x*y)/r**5))
			xz = sum(((-3*x*z)/r**5))
			yz = sum(((-3*y*z)/r**5)) 
	                       
			A[jNumber] = factor*np.array([[xx, xy, xz],[xy,yy,yz],[xz,yz,zz]])*self.g_grid

		for jNumber in np.arange(1,9):

			#obtain a sphere of lattice points of radius R centred on ion position 1B			
			iVector = self.position['1A']				
			vertices = self.lattice_sphere(R,1,'A',jNumber,'B')				
			
			#calculate the relative position of every lattice point
			x = vertices[:,0] - iVector[0]				
			y = vertices[:,1] - iVector[1]
			z = vertices[:,2] - iVector[2]
			r = np.sqrt(np.sum(np.power(vertices - iVector,2),axis=1))
			
			#calculate lattice sums
			xx = sum(((r**2-3*x**2)/r**5))
			yy = sum(((r**2-3*y**2)/r**5))
			zz = sum(((r**2-3*z**2)/r**5))
			xy = sum(((-3*x*y)/r**5))
			xz = sum(((-3*x*z)/r**5))
			yz = sum(((-3*y*z)/r**5)) 
	                       
			B[jNumber] = factor*np.array([[xx, xy, xz],[xy,yy,yz],[xz,yz,zz]])*self.g_grid

		return A, B
		# return A + demag, B + demag

	def L_matrices(self, A, B):
		LA = {}
		LA[1] = A[1]+A[2]+A[3]+A[4]+A[5]+A[6]+A[7]+A[8]
		LA[2] = A[1]-A[2]-A[3]+A[4]-A[5]+A[6]+A[7]-A[8]
		LA[3] = A[1]+A[2]-A[3]-A[4]+A[5]+A[6]-A[7]-A[8]
		LA[4] = A[1]+A[2]+A[3]+A[4]-A[5]-A[6]-A[7]-A[8]
		LA[5] = A[1]-A[2]+A[3]-A[4]+A[5]-A[6]+A[7]-A[8]
		LA[6] = A[1]+A[2]-A[3]-A[4]-A[5]-A[6]+A[7]+A[8]
		LA[7] = A[1]-A[2]-A[3]+A[4]+A[5]-A[6]-A[7]+A[8]
		LA[8] = A[1]-A[2]+A[3]-A[4]-A[5]+A[6]-A[7]+A[8]
		LB = {}
		LB[1] = B[1]+B[2]+B[3]+B[4]+B[5]+B[6]+B[7]+B[8]
		LB[2] = B[1]-B[2]-B[3]+B[4]-B[5]+B[6]+B[7]-B[8]
		LB[3] = B[1]+B[2]-B[3]-B[4]+B[5]+B[6]-B[7]-B[8]
		LB[4] = B[1]+B[2]+B[3]+B[4]-B[5]-B[6]-B[7]-B[8]
		LB[5] = B[1]-B[2]+B[3]-B[4]+B[5]-B[6]+B[7]-B[8]
		LB[6] = B[1]+B[2]-B[3]-B[4]-B[5]-B[6]+B[7]+B[8]
		LB[7] = B[1]-B[2]-B[3]+B[4]+B[5]-B[6]-B[7]+B[8]
		LB[8] = B[1]-B[2]+B[3]-B[4]-B[5]+B[6]-B[7]+B[8]
		L = {}
		for key in LA:
			L[key] = LA[key] + LB[key]
		for key in LA:
			L[key+8] = LA[key] - LB[key]
		return L

	def configuration_energies(self, R):
		A,B = self.lattice_sums(R)
		L = self.L_matrices(A,B)
		eigvals = {}
		for key in L:
			eigvals[key] = np.linalg.eigvals(L[key])
		return eigvals



DyCl3 = MonoclinicLattice()
DyCl3.axes(9.61, 6.49, 7.87, 93.65*pi/180)
DyCl3.g_tensor(1.76, 1.76, 16.52, 157*pi/180)
DyCl3.ion1_position(0.25, 0.1521, 0.25)
DyCl3.ion2_position(0.75, 0.8479, 0.75)

eigvals = DyCl3.configuration_energies(100)
print(eigvals)
