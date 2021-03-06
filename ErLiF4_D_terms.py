import numpy as np
import DipoleLatticeField

ErLiF4 = DipoleLatticeField.Tetragonal()
ErLiF4.axes(5.162, 10.70)
ErLiF4.g_tensor(8.105, 3.147)
#Define the ions to be in the bilayered antiferromagnetic ground state, aligned along the x-axis
#Ion orientation doesn't matter when calculating the dipole tensors
ErLiF4.add_position(0.5, 0.0, 0.00, (0,0,1))
ErLiF4.add_position(0.5, 0.5, 0.25, (0,0,1))
ErLiF4.add_position(0.0, 0.5, 0.50, (0,0,1))
ErLiF4.add_position(0.0, 0.0, 0.75, (0,0,1))

#Calculate the D terms between each sublattice, summing over all ions within 50 angstroms
D00 = ErLiF4.D_terms(50,0,0)
D01 = ErLiF4.D_terms(50,0,1)
D02 = ErLiF4.D_terms(50,0,2)
D03 = ErLiF4.D_terms(50,0,3)
D10 = ErLiF4.D_terms(50,1,0)
D11 = ErLiF4.D_terms(50,1,1)
D12 = ErLiF4.D_terms(50,1,2)
D13 = ErLiF4.D_terms(50,1,3)
D20 = ErLiF4.D_terms(50,2,0)
D21 = ErLiF4.D_terms(50,2,1)
D22 = ErLiF4.D_terms(50,2,2)
D23 = ErLiF4.D_terms(50,2,3)
D30 = ErLiF4.D_terms(50,3,0)
D31 = ErLiF4.D_terms(50,3,1)
D32 = ErLiF4.D_terms(50,3,2)
D33 = ErLiF4.D_terms(50,3,3)

print('D00 = ' + str(np.round(D00,3)))
print('D01 = ' + str(np.round(D01,3)))
print('D02 = ' + str(np.round(D02,3)))
print('D03 = ' + str(np.round(D03,3)))
print('D10 = ' + str(np.round(D10,3)))
print('D11 = ' + str(np.round(D11,3)))
print('D12 = ' + str(np.round(D12,3)))
print('D13 = ' + str(np.round(D13,3)))
print('D20 = ' + str(np.round(D20,3)))
print('D21 = ' + str(np.round(D21,3)))
print('D22 = ' + str(np.round(D22,3)))
print('D23 = ' + str(np.round(D23,3)))
print('D30 = ' + str(np.round(D30,3)))
print('D31 = ' + str(np.round(D31,3)))
print('D32 = ' + str(np.round(D32,3)))
print('D33 = ' + str(np.round(D33,3)))

# J00 = ErLiF4.J_terms(200,0,0)
# J01 = ErLiF4.J_terms(200,0,1)
# J02 = ErLiF4.J_terms(200,0,2)
# J03 = ErLiF4.J_terms(200,0,3)
# J10 = ErLiF4.J_terms(200,1,0)
# J11 = ErLiF4.J_terms(200,1,1)
# J12 = ErLiF4.J_terms(200,1,2)
# J13 = ErLiF4.J_terms(200,1,3)
# J20 = ErLiF4.J_terms(200,2,0)
# J21 = ErLiF4.J_terms(200,2,1)
# J22 = ErLiF4.J_terms(200,2,2)
# J23 = ErLiF4.J_terms(200,2,3)
# J30 = ErLiF4.J_terms(200,3,0)
# J31 = ErLiF4.J_terms(200,3,1)
# J32 = ErLiF4.J_terms(200,3,2)
# J33 = ErLiF4.J_terms(200,3,3)

# print('J00 = ' + str(np.round(J00,2)))
# print('J01 = ' + str(np.round(J01,2)))
# print('J02 = ' + str(np.round(J02,2)))
# print('J03 = ' + str(np.round(J03,2)))
# print('J10 = ' + str(np.round(J10,2)))
# print('J11 = ' + str(np.round(J11,2)))
# print('J12 = ' + str(np.round(J12,2)))
# print('J13 = ' + str(np.round(J13,2)))
# print('J20 = ' + str(np.round(J20,2)))
# print('J21 = ' + str(np.round(J21,2)))
# print('J22 = ' + str(np.round(J22,2)))
# print('J23 = ' + str(np.round(J23,2)))
# print('J30 = ' + str(np.round(J30,2)))
# print('J31 = ' + str(np.round(J31,2)))
# print('J32 = ' + str(np.round(J32,2)))
# print('J33 = ' + str(np.round(J33,2)))