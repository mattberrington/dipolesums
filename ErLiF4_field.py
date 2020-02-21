import numpy as np
import DipoleLatticeField

ErLiF4 = DipoleLatticeField.Tetragonal()
ErLiF4.axes(5.162, 10.70)
ErLiF4.g_tensor(8.105, 3.147)
#Define the ions to be in the bilayered antiferromagnetic ground state, aligned along the x-axis
ErLiF4.add_position(0.5, 0.0, 0.00, (+1,0,0))
ErLiF4.add_position(0.5, 0.5, 0.25, (-1,0,0))
ErLiF4.add_position(0.0, 0.5, 0.50, (-1,0,0))
ErLiF4.add_position(0.0, 0.0, 0.75, (+1,0,0))

#The magnetic field at the four ion positions are
print("[Bx, By, Bz] field at site 1 is " + str(np.round(ErLiF4.site_field(250,0)*1000)) + " mT")
print("[Bx, By, Bz] field at site 2 is " + str(np.round(ErLiF4.site_field(250,1)*1000)) + " mT")
print("[Bx, By, Bz] field at site 3 is " + str(np.round(ErLiF4.site_field(250,2)*1000)) + " mT")
print("[Bx, By, Bz] field at site 4 is " + str(np.round(ErLiF4.site_field(250,3)*1000)) + " mT")