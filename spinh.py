import numpy as np

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

def groundsplitting(M_gnd,Bx,By,Bz):
    w_gnd = np.linalg.eigvals(H(M_gnd,Bx,By,Bz))
    splitting = np.abs((w_gnd[0]-w_gnd[1]))*10**-9
    return splitting

def transitions_and_intensity(M_gnd,M_exc,Bx,By,Bz):
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
