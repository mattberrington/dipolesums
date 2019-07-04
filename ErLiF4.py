from DipoleGroundState import MonoclinicLattice
import numpy as np

pi = np.pi

ErLiF4 = MonoclinicLattice()
ErLiF4.axes(5.162, 5.162, 10.70, 90*pi/180)
ErCl3.g_tensor(3.147, 8.105, 90*pi/180)
ErCl3.ion1_position(0.5, 0.5, 0.0)
ErCl3.ion2_position(0.5, 0.0, 0.25)
ErCl3.ion3_position(0.0, 0.5, 0.75)
ErCl3.ion4_position(0.0, 0.0, 0.5)


print(np.dot(np.cross(ErCl3.a,ErCl3.b),ErCl3.c))


# print(ErCl3.configuration_energies(100)[3,1])
# print(ErCl3.configuration_energies(100)[11,1])
# print(ErCl3.configuration_energies(100)[5,1])
# print(ErCl3.configuration_energies(100)[13,1])
# print(ErCl3.configuration_energies(100)[6,1])
# print(ErCl3.configuration_energies(100)[14,1])
# print(ErCl3.configuration_energies(100)[8,1])
# print(ErCl3.configuration_energies(100)[16,1])
 
# print(ErCl3.site_field(100,1,1,1,'A'))

# print(np.linalg.norm(ErCl3.site_field(100,1,1,1,'A')))
# print(ErCl3.site_field(100,2,1,1,'A'))
# print(ErCl3.site_field(100,4,1,1,'A'))
# print(ErCl3.site_field(100,7,1,1,'A'))
# print(ErCl3.site_field(100,9,1,1,'A'))
# print(ErCl3.site_field(100,10,1,1,'A'))
# print(ErCl3.site_field(100,12,1,1,'A'))
# print(ErCl3.site_field(100,15,1,1,'A'))
# print(ErCl3.site_field(100,1,2,1,'A'))
# print(ErCl3.site_field(100,2,2,1,'A'))
# print(ErCl3.site_field(100,4,2,1,'A'))
# print(ErCl3.site_field(100,7,2,1,'A'))
# print(ErCl3.site_field(100,9,2,1,'A'))
# print(ErCl3.site_field(100,10,2,1,'A'))
# print(ErCl3.site_field(100,12,2,1,'A'))
# print(ErCl3.site_field(100,15,2,1,'A'))
# print(ErCl3.site_field(100,1,3,1,'A'))
# print(ErCl3.site_field(100,2,3,1,'A'))
# print(ErCl3.site_field(100,4,3,1,'A'))
# print(ErCl3.site_field(100,7,3,1,'A'))
# print(ErCl3.site_field(100,9,3,1,'A'))
# print(ErCl3.site_field(100,10,3,1,'A'))
# print(ErCl3.site_field(100,12,3,1,'A'))
# print(ErCl3.site_field(100,15,3,1,'A'))
# print(ErCl3.site_field(100,16,3,1,'A'))
