from DipoleGroundState import MonoclinicLattice
import numpy as np

pi = np.pi
DyCl3 = MonoclinicLattice()
DyCl3.axes(9.61, 6.49, 7.87, 93.65*pi/180)
DyCl3.g_tensor(16.52, 1.76, 157*pi/180)
DyCl3.ion1_position(0.25, 0.1521, 0.25)
DyCl3.ion2_position(0.75, 0.8479, 0.75)

print(DyCl3.configuration_energies(100))
print(DyCl3.site_field(100, 1, 1, 1, 'A'))
print(DyCl3.site_field(100, 12, 1, 1, 'A'))
