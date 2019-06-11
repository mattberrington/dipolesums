import numpy as np
import matplotlib.pyplot as plt

def Rx(theta):
    return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])

def Ry(theta):
    return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])

def Rz(theta):
    return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])

def R(alpha, beta, gamma):
    return Rz(gamma) @ Ry(beta) @ Rz(alpha) 

def M(alpha, beta, gamma, gx, gy, gz):    
    uB = 9.274*10**-24
    h = 6.626*10**-34
    m = uB/(2*h)*R(alpha, beta, gamma) @ np.diag([gx,gy,gz]) @ R(-alpha, -beta, -gamma)
    return m

def H(m, Bx, By, Bz):
    b = np.array([Bx, By, Bz])
    Ix = np.array([[0,1],[1,0]])
    Iy = np.array([[0,-1j],[1j,0]])
    Iz = np.array([[1,0],[0,-1]])
    I = np.array([Ix,Iy,Iz])
    ham = np.einsum('k,kij->ij', np.dot(b,m), I)
    return ham

def eigenstates(H):
    w,v = np.linalg.eig(H)
    w = np.abs(w)*10**-9
    return w, v

def transitions(M_gnd,M_exc,Bx,By,Bz):
    gnd1, gnd2 = np.real(np.linalg.eigvals(H(M_gnd,Bx,By,Bz)))*10**-9
    exc1, exc2 = np.real(np.linalg.eigvals(H(M_exc,Bx,By,Bz)))*10**-9
    return (exc1-gnd1, exc2-gnd1, exc1-gnd2, exc2-gnd2)

def groundsplitting_single(M_gnd,Bx,By,Bz):
    w_gnd = np.linalg.eigvals(H(M_gnd,Bx,By,Bz))
    splitting = np.abs((w_gnd[0]-w_gnd[1]))*10**-9
    return splitting

def groundsplitting(M_gnd,distribution):
    splitting = np.zeros((len(distribution),1))
    for i in range(len(distribution)):
        splitting[i]= groundsplitting_single(M_gnd,*distribution[i])
    return splitting.flatten()


def transitions_and_intensity_single(M_gnd,M_exc,Bx,By,Bz):
    w_gnd,v_gnd = np.linalg.eig(H(M_gnd,Bx,By,Bz))
    w_exc,v_exc = np.linalg.eig(H(M_exc,Bx,By,Bz))

    #Sort sort that the lower energy is the 0th element, and the upper energy the 1st element
    idx = w_gnd.argsort()[::1]   
    w_gnd = w_gnd[idx]
    v_gnd = v_gnd[:,idx]

    idx = w_exc.argsort()[::1]   
    w_exc = w_exc[idx]
    v_exc = v_exc[:,idx]

    tran1_freq = np.real(w_exc[0]-w_gnd[0])*10**-9
    tran2_freq = np.real(w_exc[1]-w_gnd[0])*10**-9
    tran3_freq = np.real(w_exc[0]-w_gnd[1])*10**-9
    tran4_freq = np.real(w_exc[1]-w_gnd[1])*10**-9
    freqs = (tran1_freq,tran2_freq,tran3_freq,tran4_freq)

    trans1_intensity = (np.abs(np.sum(v_exc[:,0]*np.conj(v_gnd[:,0]))))**2
    trans2_intensity = (np.abs(np.sum(v_exc[:,1]*np.conj(v_gnd[:,0]))))**2
    trans3_intensity = (np.abs(np.sum(v_exc[:,0]*np.conj(v_gnd[:,1]))))**2
    trans4_intensity = (np.abs(np.sum(v_exc[:,1]*np.conj(v_gnd[:,1]))))**2
    intensities = (trans1_intensity, trans2_intensity, trans3_intensity, trans4_intensity)
    return freqs, intensities

def transitions_and_intensities(M_gnd,M_exc,distribution):
    trans = np.zeros((len(distribution),4))
    intensities = np.zeros((len(distribution),4))
    for i in range(len(distribution)):
        trans[i], intensities[i] = transitions_and_intensity_single(M_gnd,M_exc,*distribution[i])
    return trans, intensities

def synthetic_spectrum(M_gnd,M_exc,distribution,T):
    trans, intensities = transitions_and_intensities(M_gnd,M_exc,distribution)
    splittings = groundsplitting(M_gnd,distribution)
    plt.figure()
    k = 20.8368 #GHz/Kelvin
    boltzmann_factor = np.exp(-splittings/(k*T))
    P_gnd = 1/(1+boltzmann_factor)
    P_exc = boltzmann_factor/(1+boltzmann_factor)
    P = np.copy(trans)

    P[:,0] = P_gnd
    P[:,1] = P_gnd
    P[:,2] = P_exc
    P[:,3] = P_exc
    
    weight = P*intensities

    y0,binEdges0 = np.histogram(trans[:,0],weights=weight[:,0],bins=1200,range=(-60,20))
    y1,binEdges1 = np.histogram(trans[:,3],weights=weight[:,3],bins=1200,range=(-60,20))
    y2,binEdges2 = np.histogram(trans[:,1],weights=weight[:,1],bins=1200,range=(-60,20))
    y3,binEdges3 = np.histogram(trans[:,2],weights=weight[:,2],bins=1200,range=(-60,20))

    maxval = np.sum([max(y0),max(y1),max(y2),max(y3)])
    y0 = y0/maxval
    y1 = y1/maxval
    y2 = y2/maxval
    y3 = y3/maxval
    alpha = 1600000 #per m

    bincenters0 = 0.5*(binEdges0[1:]+binEdges0[:-1])
    bincenters1 = 0.5*(binEdges1[1:]+binEdges1[:-1])
    bincenters2 = 0.5*(binEdges2[1:]+binEdges2[:-1])
    bincenters3 = 0.5*(binEdges3[1:]+binEdges3[:-1])

    length = 2e-3
    absorption0 = np.exp(-y0*alpha*length)
    absorption1 = np.exp(-y1*alpha*length)
    absorption2 = np.exp(-y2*alpha*length)
    absorption3 = np.exp(-y3*alpha*length)

    plt.plot(bincenters3,(absorption0*absorption1*absorption2*absorption3))

    plt.ylim(0,1.1)
    plt.grid()
    plt.xlim(-20,20)
    plt.show()   