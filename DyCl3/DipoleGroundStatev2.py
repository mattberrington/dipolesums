import time
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

		self.epsilonq = {}
		self.epsilonq[1] = np.array([1, 1, 1, 1, 1, 1, 1, 1])
		self.epsilonq[2] = np.array([1,-1,-1, 1,-1, 1, 1,-1])
		self.epsilonq[3] = np.array([1, 1,-1,-1, 1, 1,-1,-1])
		self.epsilonq[4] = np.array([1, 1, 1, 1,-1,-1,-1,-1])
		self.epsilonq[5] = np.array([1,-1, 1,-1, 1,-1, 1,-1])
		self.epsilonq[6] = np.array([1, 1,-1,-1,-1,-1, 1, 1])
		self.epsilonq[7] = np.array([1,-1,-1, 1, 1,-1,-1, 1])
		self.epsilonq[8] = np.array([1,-1, 1,-1,-1, 1,-1, 1])

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
		#Based of positions given in Figure 1 of Niemeijer paper
		self.ion1 = x*self.a + y*self.b + z*self.c
		self.position['1A'] = np.array([0,0,0]) + self.ion1

		self.position['2A'] = self.a + self.ion1
		self.position['3A'] = self.c + self.ion1
		self.position['4A'] = self.a + self.c + self.ion1
		self.position['5A'] = self.b + self.ion1
		self.position['6A'] = self.a + self.b + self.ion1
		self.position['7A'] = self.b + self.c + self.ion1
		self.position['8A'] = self.a + self.b + self.c + self.ion1

	def ion2_position(self,x,y,z):
		self.ion2 = x*self.a + y*self.b + z*self.c
		self.position['1B'] = np.array([0,0,0]) + self.ion2
		self.position['2B'] = self.a + self.ion2
		self.position['3B'] = self.c + self.ion2
		self.position['4B'] = self.a + self.c + self.ion2
		self.position['5B'] = self.b + self.ion2
		self.position['6B'] = self.a + self.b + self.ion2 
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
		#Equation 23
		LA = {}
		LA[1] = np.sum([A[i+1]*self.epsilonq[1][i] for i in range(8)],axis=0)
		LA[2] = np.sum([A[i+1]*self.epsilonq[2][i] for i in range(8)],axis=0)
		LA[3] = np.sum([A[i+1]*self.epsilonq[3][i] for i in range(8)],axis=0)
		LA[4] = np.sum([A[i+1]*self.epsilonq[4][i] for i in range(8)],axis=0)
		LA[5] = np.sum([A[i+1]*self.epsilonq[5][i] for i in range(8)],axis=0)
		LA[6] = np.sum([A[i+1]*self.epsilonq[6][i] for i in range(8)],axis=0)
		LA[7] = np.sum([A[i+1]*self.epsilonq[7][i] for i in range(8)],axis=0)
		LA[8] = np.sum([A[i+1]*self.epsilonq[8][i] for i in range(8)],axis=0)

		#Equation 23a
		LB = {}
		LB[1] = np.sum([B[i+1]*self.epsilonq[1][i] for i in range(8)],axis=0)
		LB[2] = np.sum([B[i+1]*self.epsilonq[2][i] for i in range(8)],axis=0)
		LB[3] = np.sum([B[i+1]*self.epsilonq[3][i] for i in range(8)],axis=0)
		LB[4] = np.sum([B[i+1]*self.epsilonq[4][i] for i in range(8)],axis=0)
		LB[5] = np.sum([B[i+1]*self.epsilonq[5][i] for i in range(8)],axis=0)
		LB[6] = np.sum([B[i+1]*self.epsilonq[6][i] for i in range(8)],axis=0)
		LB[7] = np.sum([B[i+1]*self.epsilonq[7][i] for i in range(8)],axis=0)
		LB[8] = np.sum([B[i+1]*self.epsilonq[8][i] for i in range(8)],axis=0)

		#Equation 27
		L = {}
		for key in LA:
			L[key] = LA[key] + LB[key]
		for key in LA:
			L[key+8] = LA[key] - LB[key]
		
		return L


	def configuration_energies(self, R):
		A,B = self.lattice_sums(R)
		L = self.L_matrices(A,B)
		w, v = np.linalg.eig(L[12])
		energies = {}

		for key in L:
			w, v = np.linalg.eig(L[key])
			energies[key,1] = w[0]
			energies[key,2] = w[1]
			energies[key,3] = w[2]
		return energies

	def configuration_direction(self, R):
		A,B = self.lattice_sums(R)
		L = self.L_matrices(A,B)
		w, v = np.linalg.eig(L[12])
		directions = {}

		for key in L:
			w, v = np.linalg.eig(L[key])
			directions[key,1] = v[:,0]
			directions[key,2] = v[:,1]
			directions[key,3] = v[:,2]
		return directions

	def site_field(self, R, config_number, config_idx, ion_number, ion_letter):
		vertices = self.lattice_sphere(R,ion_number,ion_letter,ion_number,ion_letter)	
		#SI UNITS
		mB=9.274*10**(-24) 
		k=1.380*10**(-23)
		NA=6.022*10**23 
		mu0=4*pi*10**(-7) 		
		#TODO: Check this factor 
		factor = (mu0)/(2)/((10**(-10))**3)

		rx = vertices[:,0]
		ry = vertices[:,1]
		rz = vertices[:,2]
		rtot = np.sqrt(np.sum(np.power(vertices,2),axis=1))

		mux, muy, muz = 0.5*mB*np.array([self.gx,self.gy,self.gz])*self.configuration_direction(R)[config_number,config_idx]

		Hx = factor*np.sum(-(mux/rtot**3) + (3*(mux*rx + muy*ry + muz*rz)*rx)/(rtot**5))
		Hy = factor*np.sum(-(muy/rtot**3) + (3*(mux*rx + muy*ry + muz*rz)*ry)/(rtot**5))
		Hz = factor*np.sum(-(muz/rtot**3) + (3*(mux*rx + muy*ry + muz*rz)*rz)/(rtot**5))

		return Hx, Hy, Hz

	def view_configuration(self, config_number, config_idx):
		x = np.array([1,0,0])
		y = np.array([0,1,0])
		z = np.array([0,0,1])

		if config_number < 9:
			spins = np.append(self.epsilonq[config_number],self.epsilonq[config_number])
		else:
			spins = np.append(self.epsilonq[config_number-8],-self.epsilonq[config_number-8])
		
		X, Y, Z = zip(*list(self.position.values()))

		direction = self.configuration_direction(100)[config_number,config_idx]

		U, V, W = zip(*[direction*spins[i] for i in range(16)])
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.quiver(X,Y,Z,U,V,W,colors='r',arrow_length_ratio=0.5)

		ax.plot3D(*zip(self.position['1A'],self.position['2A']), color="k")
		ax.plot3D(*zip(self.position['2A'],self.position['4A']), color="k")
		ax.plot3D(*zip(self.position['4A'],self.position['3A']), color="k")
		ax.plot3D(*zip(self.position['3A'],self.position['1A']), color="k")
		ax.plot3D(*zip(self.position['1A'],self.position['5A']), color="k")
		ax.plot3D(*zip(self.position['2A'],self.position['6A']), color="k")
		ax.plot3D(*zip(self.position['3A'],self.position['7A']), color="k")
		ax.plot3D(*zip(self.position['4A'],self.position['8A']), color="k")
		ax.plot3D(*zip(self.position['5A'],self.position['6A']), color="k")
		ax.plot3D(*zip(self.position['6A'],self.position['8A']), color="k")
		ax.plot3D(*zip(self.position['8A'],self.position['7A']), color="k")
		ax.plot3D(*zip(self.position['7A'],self.position['5A']), color="k")

		ax.plot3D(*zip(self.position['1B'],self.position['2B']), color="k")
		ax.plot3D(*zip(self.position['2B'],self.position['4B']), color="k")
		ax.plot3D(*zip(self.position['4B'],self.position['3B']), color="k")
		ax.plot3D(*zip(self.position['3B'],self.position['1B']), color="k")
		ax.plot3D(*zip(self.position['1B'],self.position['5B']), color="k")
		ax.plot3D(*zip(self.position['2B'],self.position['6B']), color="k")
		ax.plot3D(*zip(self.position['3B'],self.position['7B']), color="k")
		ax.plot3D(*zip(self.position['4B'],self.position['8B']), color="k")
		ax.plot3D(*zip(self.position['5B'],self.position['6B']), color="k")
		ax.plot3D(*zip(self.position['6B'],self.position['8B']), color="k")
		ax.plot3D(*zip(self.position['8B'],self.position['7B']), color="k")
		ax.plot3D(*zip(self.position['7B'],self.position['5B']), color="k")


		ax.text(*self.position["1A"], "1A", color='red')
		ax.text(*self.position["2A"], "2A", color='red')
		ax.text(*self.position["3A"], "3A", color='red')
		ax.text(*self.position["4A"], "4A", color='red')
		ax.text(*self.position["5A"], "5A", color='red')
		ax.text(*self.position["6A"], "6A", color='red')
		ax.text(*self.position["7A"], "7A", color='red')
		ax.text(*self.position["8A"], "8A", color='red')
		ax.text(*self.position["1B"], "1B", color='b')
		ax.text(*self.position["2B"], "2B", color='b')
		ax.text(*self.position["3B"], "3B", color='b')
		ax.text(*self.position["4B"], "4B", color='b')
		ax.text(*self.position["5B"], "5B", color='b')
		ax.text(*self.position["6B"], "6B", color='b')
		ax.text(*self.position["7B"], "7B", color='b')
		ax.text(*self.position["8B"], "8B", color='b')
		ax.view_init(103, -90)

		ax.set_xlim(2,17)
		ax.set_ylim(0,15)
		ax.set_zlim(-10,5)
		plt.show()

DyCl3 = MonoclinicLattice()
DyCl3.axes(9.61, 6.49, 7.87, 93.65*pi/180)
# DyCl3.axes(9.61, 6.49, 7.87, 110*pi/180)
DyCl3.g_tensor(1.76, 1.76, 16.52, 157*pi/180)
DyCl3.ion1_position(0.25, 0.1521, 0.25)
DyCl3.ion2_position(0.75, 0.8479, 0.75)

energies = DyCl3.configuration_energies(100)
directions = DyCl3.configuration_direction(100)
# DyCl3.view_configuration(12,1)
print(DyCl3.site_field(100, 12,1,7,'A'))

# print(DyCl3.position['1B'])
# print(DyCl3.position['2B'])
# print(DyCl3.position['3B'])
# print(DyCl3.position['4B'])
# print(DyCl3.position['5B'])
# print(DyCl3.position['6B'])
# print(DyCl3.position['7B'])
# print(DyCl3.position['8B'])


# print(DyCl3.a,DyCl3.b,DyCl3.c)
