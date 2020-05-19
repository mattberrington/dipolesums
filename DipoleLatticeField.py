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
        self.position = {}
        self.orientation = {}

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

    def add_position(self,x,y,z,orientation):
        """ Define the position of the first ion within the unit cell
        Populates the position dictionary with the locations of this ion in a double unit cell
        
        Args:
        x (float): x coordinate expressed as a fraction of crystal a vector (between 0 and 1)
        y (float): y coordinate expressed as a fraction of crystal b vector (between 0 and 1)
        z (float): z coordinate expressed as a fraction of crystal c vector (between 0 and 1)

        Returns:
            None
        """
        ion_position = x*self.a + y*self.b + z*self.c
        ion_number = len(self.position)
        self.position[ion_number] = ion_position
        self.orientation[ion_number] = orientation
    
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
                    n += 1
        return vertices

    def spherical_bravais_lattice(self,R,central_ion,lattice_multiplier=1):
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
        lattices = {}
        
        for key in self.position:
            #Shift vertices to be the lattice generated from the jth position
            vertices = self.square_bravais_lattice(R,lattice_multiplier) + self.position[key]
            #Calculate distances from the central_ion atom to each other atom
            distance = np.sqrt(np.sum(np.power(vertices - self.position[central_ion],2),axis=1))
            #only keep the locations of which are within a distance R from ion i
            #I take the intersection with non-zero distances to avoid counting origin when ith and jth ions are equal
            vertices = vertices[(distance < R) & (distance != 0.0)]
            lattices[key] = vertices
        return lattices

    def D_terms(self, R, sublattice1, sublattice2):
        """
        Returns the sums of the geometric Dipole Tensor terms. 
        It will sum the dipole tensors between sublattice1 and sublattice2
        Defined by Eqn 3 of Kraemer et al, "Dipolar Antiferromagnetism and Quantum Criticality in LiErF4", 10.1126/science.1221878
        Returns Eqn 3 times the crystal unit cell, so the result is unitless
        Args:
        R (float): The radius of the sphere (in angstroms) to sum over
        sublattice1 (int): specifies which ion position of sublattice 1
        sublattice2 (int): specifies which ion position of sublattice 2
   
        Returns:
        D_terms (array): an array of nine terms, ordered xx, xy, xz, yx, yy, yz, zx, zy, zz
        """

        vertices = self.spherical_bravais_lattice(R,sublattice2)
        #the ion location where I'm evaluating the lattice_sum
        central_ion_vector = self.position[sublattice1]

        lattices = self.spherical_bravais_lattice(R,sublattice1)

        vertices = lattices[sublattice2]

        rx = vertices[:,0] - central_ion_vector[0]
        ry = vertices[:,1] - central_ion_vector[1]
        rz = vertices[:,2] - central_ion_vector[2]
        rtot = np.sqrt(np.sum(np.power(vertices - central_ion_vector,2),axis=1))
        V = self.a[0]*self.b[1]*self.c[2] #unit cell volume

        Dxx = V*np.sum(-(1/rtot**3) + (3*rx*rx)/(rtot**5))
        Dxy = V*np.sum((3*rx*ry)/(rtot**5))
        Dxz = V*np.sum((3*rx*rz)/(rtot**5))
        Dyx = V*np.sum((3*ry*rx)/(rtot**5))
        Dyy = V*np.sum(-(1/rtot**3) + (3*ry*ry)/(rtot**5))
        Dyz = V*np.sum((3*ry*rz)/(rtot**5))
        Dzx = V*np.sum((3*rz*rx)/(rtot**5))
        Dzy = V*np.sum((3*rz*ry)/(rtot**5))
        Dzz = V*np.sum(-(1/rtot**3) + (3*rz*rz)/(rtot**5))

        return Dxx, Dxy, Dxz, Dyx, Dyy, Dyz, Dzx, Dzy, Dzz

    def J_terms(self, R, sublattice1, sublattice2):
        """
        Returns the sums of the Dipole Tensor terms, incorporation the anisotropic g factor
        It will sum the dipole tensors between sublattice1 and sublattice2
        Defined by Eqn 3 of Kraemer et al, "Dipolar Antiferromagnetism and Quantum Criticality in LiErF4", 10.1126/science.1221878
        Returns Eqn 3 times the relevant g tensors and the crystal unit cell, so the result is unitless
        Args:
        R (float): The radius of the sphere (in angstroms) to sum over
        sublattice1 (int): specifies which ion position of sublattice 1
        sublattice2 (int): specifies which ion position of sublattice 2
   
        Returns:
        J_terms (array): an array of nine terms, ordered xx, xy, xz, yx, yy, yz, zx, zy, zz
        """

        Dxx, Dxy, Dxz, Dyx, Dyy, Dyz, Dzx, Dzy, Dzz = self.D_terms(R, sublattice1, sublattice2)
        Jxx = Dxx*self.gx*self.gx
        Jxy = Dxx*self.gx*self.gy
        Jxz = Dxx*self.gx*self.gz
        Jyx = Dyy*self.gy*self.gx
        Jyy = Dyy*self.gy*self.gy
        Jyz = Dyy*self.gy*self.gz
        Jzx = Dzz*self.gz*self.gx
        Jzy = Dzz*self.gz*self.gy
        Jzz = Dzz*self.gz*self.gz

        return Jxx, Jxy, Jxz, Jyx, Jyy, Jyz, Jzx, Jzy, Jzz
  
    def site_field(self, R, central_ion):
        vertices = self.spherical_bravais_lattice(R,central_ion)
        #the ion location where I'm evaluating the magnetic field
        central_ion_vector = self.position[central_ion]

        lattices = self.spherical_bravais_lattice(R,central_ion)

        factor = (mu0)/(4*pi)/((10**(-10))**3)

        Bx = 0
        By = 0
        Bz = 0

        for key in self.position:
            vertices = lattices[key]

            rx = vertices[:,0] - central_ion_vector[0]
            ry = vertices[:,1] - central_ion_vector[1]
            rz = vertices[:,2] - central_ion_vector[2]
            rtot = np.sqrt(np.sum(np.power(vertices - central_ion_vector,2),axis=1))

            direction = np.array(self.orientation[key])


            mux = 0.5*mB*self.gx*direction[0]
            muy = 0.5*mB*self.gy*direction[1]
            muz = 0.5*mB*self.gz*direction[2]


            Bx += factor*np.sum(-(mux/rtot**3) + (3*(mux*rx + muy*ry + muz*rz)*rx)/(rtot**5))
            By += factor*np.sum(-(muy/rtot**3) + (3*(mux*rx + muy*ry + muz*rz)*ry)/(rtot**5))
            Bz += factor*np.sum(-(muz/rtot**3) + (3*(mux*rx + muy*ry + muz*rz)*rz)/(rtot**5))

        return np.array([Bx, By, Bz])
class Tetragonal:
    def __init__(self):
        self.position = {}
        self.orientation = {}

    def axes(self,a_len,c_len):
        """ Define the crystalographic properties of the lattice
        Args:
        a_len (float): length of the 'a' crystal vector
        c_len (float): length of the 'c' crystal vector
        Returns:
            None
        """
        self.a = np.array([a_len,0,0])
        self.b = np.array([0,a_len,0])
        self.c = np.array([0,0,c_len])

    def g_tensor(self,gperp,gpara):
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
        self.g_grid = np.array([[gx*gx, gx*gy, gx*gz],[gy*gx, gy*gy, gy*gz],[gz*gx, gz*gy, gz*gz]])
        self.gx = gx
        self.gy = gy
        self.gz = gz
        # rotate the crystal coordinates so that I'm now in the coordinate system 
        # given by the zeeman tensor's principal axes

    def add_position(self,x,y,z,orientation):
        """ Define the position of the first ion within the unit cell
        Populates the position dictionary with the locations of this ion in a double unit cell
        
        Args:
        x (float): x coordinate expressed as a fraction of crystal a vector (between 0 and 1)
        y (float): y coordinate expressed as a fraction of crystal b vector (between 0 and 1)
        z (float): z coordinate expressed as a fraction of crystal c vector (between 0 and 1)

        Returns:
            None
        """
        ion_position = x*self.a + y*self.b + z*self.c
        ion_number = len(self.position)
        self.position[ion_number] = ion_position
        self.orientation[ion_number] = orientation

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
                    n += 1
        return vertices

    def spherical_bravais_lattice(self,R,central_ion,lattice_multiplier=1):
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
        lattices = {}
        
        for key in self.position:
            #Shift vertices to be the lattice generated from the jth position
            vertices = self.square_bravais_lattice(R,lattice_multiplier) + self.position[key]
            #Calculate distances from the central_ion atom to each other atom
            distance = np.sqrt(np.sum(np.power(vertices - self.position[central_ion],2),axis=1))
            #only keep the locations of which are within a distance R from ion i
            #I take the intersection with non-zero distances to avoid counting origin when ith and jth ions are equal
            vertices = vertices[(distance < R) & (distance != 0.0)]
            lattices[key] = vertices
        self.lattices = lattices        
        return lattices

    def D_terms(self, R, sublattice1, sublattice2):
        """
        Returns the sums of the geometric Dipole Tensor terms. 
        It will sum the dipole tensors between sublattice1 and sublattice2
        Defined by Eqn 3 of Kraemer et al, "Dipolar Antiferromagnetism and Quantum Criticality in LiErF4", 10.1126/science.1221878
        Returns Eqn 3 times the crystal unit cell, so the result is unitless
        Args:
        R (float): The radius of the sphere (in angstroms) to sum over
        sublattice1 (int): specifies which ion position of sublattice 1
        sublattice2 (int): specifies which ion position of sublattice 2
   
        Returns:
        D_terms (array): an array of nine terms, ordered xx, xy, xz, yx, yy, yz, zx, zy, zz
        """

        vertices = self.spherical_bravais_lattice(R,sublattice2)
        #the ion location where I'm evaluating the lattice_sum
        central_ion_vector = self.position[sublattice1]

        lattices = self.spherical_bravais_lattice(R,sublattice1)

        vertices = lattices[sublattice2]

        rx = vertices[:,0] - central_ion_vector[0]
        ry = vertices[:,1] - central_ion_vector[1]
        rz = vertices[:,2] - central_ion_vector[2]
        rtot = np.sqrt(np.sum(np.power(vertices - central_ion_vector,2),axis=1))
        V = self.a[0]*self.b[1]*self.c[2] #unit cell volume

        Dxx = V*np.sum(-(1/rtot**3) + (3*rx*rx)/(rtot**5))
        Dxy = V*np.sum((3*rx*ry)/(rtot**5))
        Dxz = V*np.sum((3*rx*rz)/(rtot**5))
        Dyx = V*np.sum((3*ry*rx)/(rtot**5))
        Dyy = V*np.sum(-(1/rtot**3) + (3*ry*ry)/(rtot**5))
        Dyz = V*np.sum((3*ry*rz)/(rtot**5))
        Dzx = V*np.sum((3*rz*rx)/(rtot**5))
        Dzy = V*np.sum((3*rz*ry)/(rtot**5))
        Dzz = V*np.sum(-(1/rtot**3) + (3*rz*rz)/(rtot**5))

        return Dxx, Dxy, Dxz, Dyx, Dyy, Dyz, Dzx, Dzy, Dzz

    def J_terms(self, R, sublattice1, sublattice2):
        """
        Returns the sums of the Dipole Tensor terms, incorporation the anisotropic g factor
        It will sum the dipole tensors between sublattice1 and sublattice2
        Defined by Eqn 3 of Kraemer et al, "Dipolar Antiferromagnetism and Quantum Criticality in LiErF4", 10.1126/science.1221878
        Returns Eqn 3 times the relevant g tensors and the crystal unit cell, so the result is unitless
        Args:
        R (float): The radius of the sphere (in angstroms) to sum over
        sublattice1 (int): specifies which ion position of sublattice 1
        sublattice2 (int): specifies which ion position of sublattice 2
   
        Returns:
        J_terms (array): an array of nine terms, ordered xx, xy, xz, yx, yy, yz, zx, zy, zz
        """

        Dxx, Dxy, Dxz, Dyx, Dyy, Dyz, Dzx, Dzy, Dzz = self.D_terms(R, sublattice1, sublattice2)
        Jxx = Dxx*self.gx*self.gx
        Jxy = Dxx*self.gx*self.gy
        Jxz = Dxx*self.gx*self.gz
        Jyx = Dyy*self.gy*self.gx
        Jyy = Dyy*self.gy*self.gy
        Jyz = Dyy*self.gy*self.gz
        Jzx = Dzz*self.gz*self.gx
        Jzy = Dzz*self.gz*self.gy
        Jzz = Dzz*self.gz*self.gz

        return Jxx, Jxy, Jxz, Jyx, Jyy, Jyz, Jzx, Jzy, Jzz

    def site_field(self, R, central_ion):
        vertices = self.spherical_bravais_lattice(R,central_ion)
        #the ion location where I'm evaluating the magnetic field
        central_ion_vector = self.position[central_ion]

        lattices = self.spherical_bravais_lattice(R,central_ion)

        factor = (mu0)/(8*pi)/((10**(-10))**3)

        Bx = 0
        By = 0
        Bz = 0

        for key in self.position:
            vertices = lattices[key]

            rx = vertices[:,0] - central_ion_vector[0]
            ry = vertices[:,1] - central_ion_vector[1]
            rz = vertices[:,2] - central_ion_vector[2]
            rtot = np.sqrt(np.sum(np.power(vertices - central_ion_vector,2),axis=1))
            direction = np.array(self.orientation[key])


            mux = 0.5*mB*self.gx*direction[0]
            muy = 0.5*mB*self.gy*direction[1]
            muz = 0.5*mB*self.gz*direction[2]
            # print('Sublattice ' + str(key))
            # print(np.sum(-(1/rtot**3) + (3*(1*rx)*rx)/(rtot**5)))

            Bx += factor*np.sum(-(mux/rtot**3) + (3*(mux*rx + muy*ry + muz*rz)*rx)/(rtot**5))
            By += factor*np.sum(-(muy/rtot**3) + (3*(mux*rx + muy*ry + muz*rz)*ry)/(rtot**5))
            Bz += factor*np.sum(-(muz/rtot**3) + (3*(mux*rx + muy*ry + muz*rz)*rz)/(rtot**5))


        return np.array([Bx, By, Bz])