import time
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Define SI units
pi = np.pi
mB=9.274*10**(-24)
k=1.380*10**(-23)
NA=6.022*10**23
mu0=4*pi*10**(-7)


#define rotation matrices
def Rx(theta):
    return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
def Ry(theta):
    return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
def Rz(theta):
    return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])

class MonoclinicLattice:
    def __init__(self):
        #Based of positions given in Figure 1 of Niemeijer paper
        self.position_map = {}
        self.position_map[1] = [0,0,0]
        self.position_map[2] = [1,0,0]
        self.position_map[3] = [0,0,1]
        self.position_map[4] = [1,0,1]
        self.position_map[5] = [0,1,0]
        self.position_map[6] = [1,1,0]
        self.position_map[7] = [0,1,1]
        self.position_map[8] = [1,1,1]

        self.position_map_inverse = {}
        self.position_map_inverse[0,0,0] = 1
        self.position_map_inverse[1,0,0] = 2
        self.position_map_inverse[0,0,1] = 3
        self.position_map_inverse[1,0,1] = 4
        self.position_map_inverse[0,1,0] = 5
        self.position_map_inverse[1,1,0] = 6
        self.position_map_inverse[0,1,1] = 7
        self.position_map_inverse[1,1,1] = 8

        self.epsilonq = {}
        self.epsilonq[1] = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        self.epsilonq[2] = np.array([1,-1,-1, 1,-1, 1, 1,-1])
        self.epsilonq[3] = np.array([1, 1,-1,-1, 1, 1,-1,-1])
        self.epsilonq[4] = np.array([1, 1, 1, 1,-1,-1,-1,-1])
        self.epsilonq[5] = np.array([1,-1, 1,-1, 1,-1, 1,-1])
        self.epsilonq[6] = np.array([1, 1,-1,-1,-1,-1, 1, 1])
        self.epsilonq[7] = np.array([1,-1,-1, 1, 1,-1,-1, 1])
        self.epsilonq[8] = np.array([1,-1, 1,-1,-1, 1,-1, 1])

        self.position = {}

    def axes(self,a_len,b_len,c_len,beta):
        """ Define the crystalographic properties of the lattice

        Args:
        a_len (float): length of the 'a' crystal vector
        b_len (float): length of the 'b' crystal vector
        c_len (float): length of the 'c' crystal vector
        beta (float): the angle on the monoclinic crystal in radians. pi/2 = rectangular

        Returns:
            None
        """
        self.a = np.array([0,0,a_len])
        self.b = np.array([0,b_len,0])
        self.c = Ry(-beta) @ np.array([0,0,c_len])

    def g_tensor(self,gpara,gperp,zeta_a):
        """ Define the g-tensor of the crystal. 
        Assumes an axial tensor in the AC plane
        
        Args:
        gpara (float): axial component of g tensor
        gperp (float): perdendicular component of g tensor
        zeta_a (float): the angle between the g tensor's axial direction and the crystal a vector

        Returns:
            None
        """
        gx = gperp
        gy = gperp
        gz = gpara

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
        """ Define the position of the first ion within the unit cell
        Populates the position dictionary with the locations of this ion in a double unit cell
        
        Args:
        x (float): x coordinate expressed as a fraction of crystal a vector (between 0 and 1)
        y (float): y coordinate expressed as a fraction of crystal b vector (between 0 and 1)
        z (float): z coordinate expressed as a fraction of crystal c vector (between 0 and 1)

        Returns:
            None
        """
        axes_vector = np.array([self.a,self.b,self.c])
        self.ion1 = x*self.a + y*self.b + z*self.c
        self.position['1A'] = np.dot(self.position_map[1],axes_vector) + self.ion1
        self.position['2A'] = np.dot(self.position_map[2],axes_vector) + self.ion1
        self.position['3A'] = np.dot(self.position_map[3],axes_vector) + self.ion1
        self.position['4A'] = np.dot(self.position_map[4],axes_vector) + self.ion1
        self.position['5A'] = np.dot(self.position_map[5],axes_vector) + self.ion1
        self.position['6A'] = np.dot(self.position_map[6],axes_vector) + self.ion1
        self.position['7A'] = np.dot(self.position_map[7],axes_vector) + self.ion1
        self.position['8A'] = np.dot(self.position_map[8],axes_vector) + self.ion1

    def ion2_position(self,x,y,z):
        """ Define the position of the second ion within the unit cell
        Populates the position dictionary with the locations of this ion in a double unit cell
        
        Args:
        x (float): x coordinate expressed as a fraction of crystal a vector (between 0 and 1)
        y (float): y coordinate expressed as a fraction of crystal b vector (between 0 and 1)
        z (float): z coordinate expressed as a fraction of crystal c vector (between 0 and 1)

        Returns:
            None
        """
        axes_vector = np.array([self.a,self.b,self.c])
        self.ion2 = x*self.a + y*self.b + z*self.c
        self.position['1B'] = np.dot(self.position_map[1],axes_vector) + self.ion2
        self.position['2B'] = np.dot(self.position_map[2],axes_vector) + self.ion2
        self.position['3B'] = np.dot(self.position_map[3],axes_vector) + self.ion2
        self.position['4B'] = np.dot(self.position_map[4],axes_vector) + self.ion2
        self.position['5B'] = np.dot(self.position_map[5],axes_vector) + self.ion2
        self.position['6B'] = np.dot(self.position_map[6],axes_vector) + self.ion2
        self.position['7B'] = np.dot(self.position_map[7],axes_vector) + self.ion2
        self.position['8B'] = np.dot(self.position_map[8],axes_vector) + self.ion2
    
    def square_bravais_lattice(self,R,lattice_multiplier=1):
        """Generates a lattice constructed from the unit cell vectors, centred on (0,0,0)
        returns a list of lattice sites across a grid of size 2R x 2R x 2R, which will contain
        a sphere of radius R

        Args:
        R (float): The sphere radius (in angstroms) that the square lattice will just cover
        lattice_multiplier (int): a value of n will generate a lattice using crystal vectors n times bigger
   
        Returns:
        vertices, vertex_labels (np.array, np.array): an array of coordinates of lattice points, and an array of the vertex names (1 through 8 according to how the points are defined)
        """
        a = lattice_multiplier*self.a
        b = lattice_multiplier*self.b
        c = lattice_multiplier*self.c

        #Calculate the number of lattice points needed in each direction to cover a length of R
        #I use the ceiling function so that when I shift the origin by a one unit cell vector,
        #I still cover all lattive points within a distance of R
        Na = int(np.ceil(R/np.linalg.norm(a)))
        Nb = int(np.ceil(R/np.linalg.norm(b)))
        Nc = int(np.ceil(R/np.linalg.norm(c)))

        #calculate the number of vertices in a grid that covers the sphere
        #A sphere of radius R fits within a grid of size 2R x 2R x 2R
        #Adding one to account for origin
        number_vertices = (2*Na+1)*(2*Nb+1)*(2*Nc+1)
        vertices = np.empty((number_vertices,3))
        vertex_labels = np.empty(number_vertices ,dtype=int)
        
        # populate the vertices list with the positions of a lattice with single spacing
        n = 0
        for i in np.arange(-Na,Na+1):
            for j in np.arange(-Nb,Nb+1):
                for k in np.arange(-Nc,Nc+1):
                    vertices[n]=np.dot([[i,j,k]],[[a[0],a[1],a[2]],[b[0],b[1],b[2]],[c[0],c[1],c[2]]])
                    vertex_labels[n] = self.position_map_inverse[(i*lattice_multiplier)%2,(j*lattice_multiplier)%2,(k*lattice_multiplier)%2]
                    n += 1
        return vertices, vertex_labels

    def spherical_bravais_lattice(self,R,iNumber,iLetter,jNumber,jLetter,lattice_multiplier=1):
        """
        Returns the bravais lattice generated from the jth ion, for a given mutliplicity
        Return all ions within R of the ith ion
        When lattice_multiplier=1, the jth ion has no effect on what's returned

        Args:
        R (float): The radius of the sphere (in angstroms) to include lattice points
        iNumber (int): specifies the ith ion's position label
        iLetter (string): specific the ith ion's sublattice (must be 'A' or 'B')
        jNumber (int): specifies the ith ion's position label
        jLetter (string): specific the jth ion's sublattice (must be 'A' or 'B')
        lattice_multiplier (int): a value of n will generate a lattice using crystal vectors n times bigger
   
        Returns:
        vertices, vertex_labels (np.array, np.array): an array of coordinates of lattice points, and an array of the vertex names (1 through 8 according to how the points are defined)
        """

        vertices, vertex_labels = self.square_bravais_lattice(R,lattice_multiplier)
        #Shift vertices to be the lattice generated from the jth position
        vertices = vertices + self.position[str(jNumber) + jLetter]
        #Calculate distances from the ith atom to each other atom
        distance = np.sqrt(np.sum(np.power(vertices - self.position[str(iNumber) + iLetter],2),axis=1))
        #only keep the locations of which are within a distance R from ion i
        #I take the intersection with non-zero distances to avoid counting origin when ith and jth ions are equal
        vertices = vertices[(distance < R) & (distance != 0.0)]
        vertex_labels = vertex_labels[(distance < R) & (distance != 0.0)]
        #If this is a lattice of the B ions, then change the vertex labels accordingly
        if jLetter == 'B':
            vertex_labels += 8
        
        return vertices, vertex_labels

    def lattice_sums(self,R):
        """
        Calculates the lattice sums as described in NIEMEIJER paper
        
        Args:
        R (float): The radius of the sphere (in angstroms) to include lattice points
   
        Returns:
        A, B (dict, dict): dictionaries where each entry is an A or B matrix for certain values of i/j
        """
        
        factor = (mu0*mB**2)/(32*k*pi)/((10**(-10))**3)

        A = {}
        B = {}

        for jNumber in np.arange(1,9):

            #obtain a sphere of lattice points of radius R centred on ion position 1A
            iVector = self.position['1A']               
            vertices,_ = self.spherical_bravais_lattice(R,1,'A',jNumber,'A',lattice_multiplier=2)           
            
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
            vertices,_ = self.spherical_bravais_lattice(R,1,'A',jNumber,'B',lattice_multiplier=2)       
            
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

        factor = (mu0*mB**2)/(32*k*pi)/((10**(-10))**3)
        rho = 2/(np.dot(np.cross(self.a,self.b),self.c)) # I've double checked this using molecular weight
        Nx = 2*pi/3
        Ny = 2*pi/3
        Nz = -4*pi/3
        demagnetisation = np.array([[Nx*rho,0,0],[0,Ny*rho,0],[0,0,Nz*rho]])
        L[1] = L[1]+factor*demagnetisation*self.g_grid

        
        return L

    def configuration_energies(self, R):
        A,B = self.lattice_sums(R)
        L = self.L_matrices(A,B)
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
        vertices, vertex_labels = self.square_bravais_lattice(R)
        #the ion location where I'm evaluating the magnetic field
        ion_vector = self.position[str(ion_number)+ion_letter]

        A_vertices, A_vertex_labels = self.spherical_bravais_lattice(R,ion_number,ion_letter,1,'A')
        B_vertices, B_vertex_labels = self.spherical_bravais_lattice(R,ion_number,ion_letter,1,'B')

        if config_number < 9:
            spin_orientation = np.append(self.epsilonq[config_number],self.epsilonq[config_number])
        else:
            spin_orientation = np.append(self.epsilonq[config_number-8],-self.epsilonq[config_number-8])

        factor = (mu0)/(4*pi)/((10**(-10))**3)

        A_rx = A_vertices[:,0] - ion_vector[0]
        A_ry = A_vertices[:,1] - ion_vector[1]
        A_rz = A_vertices[:,2] - ion_vector[2]
        A_rtot = np.sqrt(np.sum(np.power(A_vertices - ion_vector,2),axis=1))

        B_rx = B_vertices[:,0] - ion_vector[0]
        B_ry = B_vertices[:,1] - ion_vector[1]
        B_rz = B_vertices[:,2] - ion_vector[2]
        B_rtot = np.sqrt(np.sum(np.power(B_vertices - ion_vector,2),axis=1))

        direction = self.configuration_direction(R)[config_number,config_idx]

        A_mux = 0.5*mB*self.gx*direction[0]*spin_orientation[A_vertex_labels-1]
        A_muy = 0.5*mB*self.gy*direction[1]*spin_orientation[A_vertex_labels-1]
        A_muz = 0.5*mB*self.gz*direction[2]*spin_orientation[A_vertex_labels-1]
        B_mux = 0.5*mB*self.gx*direction[0]*spin_orientation[B_vertex_labels-1]
        B_muy = 0.5*mB*self.gy*direction[1]*spin_orientation[B_vertex_labels-1]
        B_muz = 0.5*mB*self.gz*direction[2]*spin_orientation[B_vertex_labels-1]

        Bx_A = -(A_mux/A_rtot**3) + (3*(A_mux*A_rx + A_muy*A_ry + A_muz*A_rz)*A_rx)/(A_rtot**5)
        By_A = -(A_muy/A_rtot**3) + (3*(A_mux*A_rx + A_muy*A_ry + A_muz*A_rz)*A_ry)/(A_rtot**5)
        Bz_A = -(A_muz/A_rtot**3) + (3*(A_mux*A_rx + A_muy*A_ry + A_muz*A_rz)*A_rz)/(A_rtot**5)

        Bx_B = -(B_mux/B_rtot**3) + (3*(B_mux*B_rx + B_muy*B_ry + B_muz*B_rz)*B_rx)/(B_rtot**5)
        By_B = -(B_muy/B_rtot**3) + (3*(B_mux*B_rx + B_muy*B_ry + B_muz*B_rz)*B_ry)/(B_rtot**5)
        Bz_B = -(B_muz/B_rtot**3) + (3*(B_mux*B_rx + B_muy*B_ry + B_muz*B_rz)*B_rz)/(B_rtot**5)

        Bx = factor*(np.sum(Bx_A)+np.sum(Bx_B))
        By = factor*(np.sum(By_A)+np.sum(By_B))
        Bz = factor*(np.sum(Bz_A)+np.sum(Bz_B))

        if(config_number == 1):
            rho = 2/(np.dot(np.cross(self.a,self.b),self.c))/((10**(-10))**3)
            mux = 0.5*mB*self.gx*direction[0]
            muy = 0.5*mB*self.gy*direction[1]
            muz = 0.5*mB*self.gz*direction[2]

            M = rho*np.array([mux,muy,muz])
            Bcavity = mu0*M/3
            Bx, By, Bz = np.array([Bx, By,Bz]) + Bcavity

        return np.array([Bx, By, Bz])

    def view_configuration(self, config_number, config_idx, R=100):
        x = np.array([1,0,0])
        y = np.array([0,1,0])
        z = np.array([0,0,1])

        if config_number < 9:
            spins = np.append(self.epsilonq[config_number],self.epsilonq[config_number])
        else:
            spins = np.append(self.epsilonq[config_number-8],-self.epsilonq[config_number-8])
        
        X, Y, Z = zip(*list(self.position.values()))

        direction = self.configuration_direction(R)[config_number,config_idx]

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

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        upper_lim = np.max(np.array(list(self.position.values())),axis=0)
        lower_lim = np.min(np.array(list(self.position.values())),axis=0)
        max_range = np.max(upper_lim-lower_lim)/2
        midpoint = (upper_lim+lower_lim)/2

        ax.set_xlim(midpoint[0]-max_range,midpoint[0]+max_range)
        ax.set_ylim(midpoint[1]-max_range,midpoint[1]+max_range)
        ax.set_zlim(midpoint[2]-max_range,midpoint[2]+max_range)
        plt.show()

class Tetragonal:
    def __init__(self):
        #Based of positions given in Figure 1 of Niemeijer paper
        self.position = {}

    def axes(self,a_len,c_len):
        """ Define the crystalographic properties of the lattice
        Args:
        a_len (float): length of the 'a' crystal vector
        c_len (float): length of the 'c' crystal vector
        Returns:
            None
        """
        self.a = np.array([a_len,0,0])
        self.b = np.array([0,b_len,0])
        self.c = np.array([0,0,c_len])

    def g_tensor(self,gpara,gperp):
        """ Define the g-tensor of the crystal.         
        Args:
        gpara (float): axial component of g tensor
        gperp (float): perdendicular component of g tensor

        Returns:
            None
        """
        gx = gperp
        gy = gperp
        gz = gpara

        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.g_grid = np.array([[gx*gx, gx*gy, gx*gz],[gy*gx, gy*gy, gy*gz],[gz*gx, gz*gy, gz*gz]])
        # rotate the crystal coordinates so that I'm now in the coordinate system 
        # given by the zeeman tensor's principal axes

    def ion1_position(self,x,y,z):
        """ Define the position of the first ion within the unit cell
        Populates the position dictionary with the locations of this ion in a double unit cell
        
        Args:
        x (float): x coordinate expressed as a fraction of crystal a vector (between 0 and 1)
        y (float): y coordinate expressed as a fraction of crystal b vector (between 0 and 1)
        z (float): z coordinate expressed as a fraction of crystal c vector (between 0 and 1)

        Returns:
            None
        """
        axes_vector = np.array([self.a,self.b,self.c])
        self.ion1 = x*self.a + y*self.b + z*self.c
        self.position['1A'] = np.dot(self.position_map[1],axes_vector) + self.ion1
        self.position['2A'] = np.dot(self.position_map[2],axes_vector) + self.ion1
        self.position['3A'] = np.dot(self.position_map[3],axes_vector) + self.ion1
        self.position['4A'] = np.dot(self.position_map[4],axes_vector) + self.ion1
        self.position['5A'] = np.dot(self.position_map[5],axes_vector) + self.ion1
        self.position['6A'] = np.dot(self.position_map[6],axes_vector) + self.ion1
        self.position['7A'] = np.dot(self.position_map[7],axes_vector) + self.ion1
        self.position['8A'] = np.dot(self.position_map[8],axes_vector) + self.ion1