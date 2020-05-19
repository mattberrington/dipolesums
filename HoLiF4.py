import numpy as np
import DipoleLatticeField

HoLiF4 = DipoleLatticeField.Tetragonal()
HoLiF4.axes(5.175, 10.75)
HoLiF4.g_tensor(0, 13.78)
#Define the ions to be in the bilayered antiferromagnetic ground state, aligned along the x-axis
HoLiF4.add_position(0.5, 0.0, 0.00, (0,0,1))
HoLiF4.add_position(0.5, 0.5, 0.25, (0,0,1))
HoLiF4.add_position(0.0, 0.5, 0.50, (0,0,1))
HoLiF4.add_position(0.0, 0.0, 0.75, (0,0,1))

#The magnetic field at the four ion positions are
# print("[Bx, By, Bz] field at site 1 is " + str(np.round(HoLiF4.site_field(50,0)*1000)) + " mT")
# print("[Bx, By, Bz] field at site 2 is " + str(np.round(HoLiF4.site_field(50,1)*1000)) + " mT")
# print("[Bx, By, Bz] field at site 3 is " + str(np.round(HoLiF4.site_field(50,2)*1000)) + " mT")
# print("[Bx, By, Bz] field at site 4 is " + str(np.round(HoLiF4.site_field(50,3)*1000)) + " mT")

#1.389e22 cm^-3 = 1.389e-2 A^-3
J00 = np.array(HoLiF4.D_terms(50,0,0))
J01 = np.array(HoLiF4.D_terms(50,0,1))
J02 = np.array(HoLiF4.D_terms(50,0,2))
J03 = np.array(HoLiF4.D_terms(50,0,3))
# print(J00)
# print(J01)
# print(J02)
# print(J03)
print((J00+J01+J02+J03)/4)
