import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as ssl

def Pauli(idx):
    """
    Pauli matrices
    """
    if idx==0:
        pmat=np.identity(2, dtype=float)
    elif idx==1:
        pmat=np.array(([0,1],[1,0]),dtype=float)
    elif idx==2:
        pmat=np.array(([0,-1j],[1j,0]),dtype=complex)
    elif idx==3:
        pmat=np.array(([1,0],[0,-1]),dtype=float)

    return pmat

def BulkModel(kx,ky,kz,g):
    """
    Hamiltonian for the Bulk Weyl Hamiltonian
    """
    M_z = (2 - np.cos(kx) - np.cos(ky) - np.cos(kz) + g) * Pauli(3)
    M_x = np.sin(kx) * Pauli(1)
    M_y = np.sin(ky) * Pauli(2)

    MAT = M_x + M_y + M_z

    return MAT

def BulkSpectrumWidget(kx,ky,g):
    """
    Compute energies of Bulk 2-band Model
    Plots as a function of kz
    """
    res = 100
    kzs = np.linspace(-np.pi,np.pi,num=res)
    Es = np.zeros((int(2*res)),dtype=float)

    for i in range(res):
        kz = kzs[i]
        H = BulkModel(kx,ky,kz,g)
        E = np.linalg.eigvalsh(H)
        Es[2*i:2*(i+1)] = E

    # make k array for plotting
    kzs_plot = np.repeat(kzs,2)
    
    # plot it
    fig, ax = plt.subplots()
    
    ax.scatter(kzs_plot,Es,marker='.',c='k')
    ax.axhline(y=0,c='k',ls='--')
    ax.axvline(x=-np.arccos(g),c='r')
    ax.axvline(x=np.arccos(g),c='r')
    ax.set_ylim(-1.5,1.5)
    ax.set_xlim(-np.pi,np.pi)
    ax.set_ylabel(r"$k_z$")
    plt.show()
    
    return ax

def States(kz,res,occ=True,g=0):
    """
    Returns a grid of states
    4 dimensions: [kx,ky,kz, 2 band]
    occ is True (occupied band) or False (valence band)
    """
    # 0 if filled, 1 if valence
    if occ:
        index = 0
    else:
        index = 1

    states = np.zeros((res,res,2),dtype=complex)

    for i in range(res):
        kx = -np.pi + i * 2 * np.pi / res 
        for j in range(res):
            ky = -np.pi + j * 2 * np.pi / res 
            E, waves = np.linalg.eigh(BulkModel(kx,ky,kz,g=g))
            states[i,j,:] = waves[:,index]

    return states

def uij(u,v):
    """
    Computes overlap of wavefunctions u, v
    """
    return np.dot(np.conjugate(u),v)

def BerryFlux(n,m,states,res):
    """
    Computes product
    <u_{n,m}|u_{n+1,m}><u_{n+1,m}|u_{n+1,m+1}><u_{n+1,m+1}|u_{n,m+1}><u_{n,m+1}|u_{n,m}>
    Returns the Wilson loop for a given kz
    """
    # for a given kz
    # product over neighbouring sites
    # imposing pbc by virtue of remainder division %
    W = uij(states[n,m,:],states[(n+1)%res,m,:]) 
    W *= uij(states[(n+1)%res,m,:],states[(n+1)%res,(m+1)%res,:])
    W *= uij(states[(n+1)%res,(m+1)%res,:],states[n,(m+1)%res,:])
    W *= uij(states[n,(m+1)%res,:],states[n,m,:])

    return np.arctan2(W.imag,W.real) # might be a minus sign in front

def ChernNumberKz(states,res):
    """
    Discrete sum over all plaquettes (n,m)
    """
    # Chern numbers
    Q = 0

    # Sum over all plaquettes
    for n in range(res):
        for m in range(res):
            Fnm = BerryFlux(n,m,states,res)
            Q += Fnm
    
    Q /= 2 * np.pi

    return np.around(Q,2)

def BulkSpectrumChern(kx,ky,g,res=10):

    kzs = np.linspace(-np.pi,np.pi,num=res)
    Es = np.zeros((int(2*res)),dtype=float)
    Cs = np.zeros(int(2*res),dtype=int)
    kzs_plot = np.repeat(kzs,2)

    for i in range(res):
        kz = kzs[i]
        states = States(kz,res,occ=True,g=g)
        states_ = States(kz,res,occ=False,g=g)
        H = BulkModel(kx,ky,kz,g)
        E = np.linalg.eigvalsh(H)
        Es[2*i:2*(i+1)] = E
        C = ChernNumberKz(states,res)
        C_ = ChernNumberKz(states_,res)
        Cs[2*i:2*(i+1)] = np.asarray([C,C_])

    return kzs_plot, Es, Cs


### open system in y
def WeylHamiltonian(size,kx,kz,g):
    """
    Hamiltonian for Bulk Weyl Semimetal
    Two-node minimal model
    Open in y, closed in x, z
    """
    # diagonals
    diags_x = np.asarray([np.sin(kx) for _ in range(size)])
    diags_z = np.asarray([(2 + g - np.cos(kx) - np.cos(kz)) for _ in range(size)])

    diag_x = np.kron(np.diag(diags_x),Pauli(1))
    diag_z = np.kron(np.diag(diags_z),Pauli(3))

    diags = diag_x + diag_z

    # hopping
    hop_low = 1j / 2 * np.kron(np.eye(size,k=-1),Pauli(2)) - 1/ 2 * np.kron(np.eye(size,k=-1),Pauli(3))
    hop = hop_low + hop_low.conj().T

    MAT = diags + hop

    return MAT

def Position(wave,size):
    """
    position of the state in y
    Equipped to handle array where W[:,i] is ith wave
    works VVV
    todo: adapt for doubly-open system (already have code)
    """
    # make wave into what it was Born to be: probability
    prob = np.abs(wave)**2
    prob_norm = prob / np.sum(prob, axis=0)

    fac = int(wave.shape[0] / size) 

    ys = np.repeat(np.arange(size),int(fac))

    ypos = ys@prob_norm

    return np.asarray(ypos.T)


def SpectrumFiniteY(size,res,kfix,g,k_dir=3):
    """
    Compute energies of Finite X Hamiltonian
    Plots as a function of k_dir
    """
    Hdim = int(2*size)
    ks = np.linspace(-np.pi,np.pi,num=res)
    Es = np.zeros(int(Hdim*res),dtype=float)
    Ypos = np.zeros(int(Hdim*res), dtype=float)

    for i in range(res):
        k = ks[i]
        if k_dir == 3:
            H = WeylHamiltonian(size,kx=kfix,kz=k,g=g)
        else: 
            H = WeylHamiltonian(size,kx=k,kz=kfix,g=g)
        E, wave = ssl.eigsh(H, k=Hdim, return_eigenvectors=True)
        ypos = Position(wave,size=size)
        Es[Hdim*i:Hdim*(i+1)] = E
        Ypos[Hdim*i:Hdim*(i+1)] = ypos

    # make k array for plotting
    ks_plot = np.repeat(ks,Hdim)

    return ks_plot, Es, Ypos



### tunnelling


def MetalHamiltonian(size,kx,kz,t,mu):
    """
    Hamiltonian for Bulk Metal 
    Open in y, closed in x, z
    """
    # diagonals
    diags_0 = np.asarray([(- 2 * t * (np.cos(kx) + np.cos(kz) - 3) - mu) for _ in range(size)])

    diags = np.kron(np.diag(diags_0),Pauli(0))

    # hopping
    hop_low = -t * np.kron(np.eye(size,k=-1),Pauli(0))
    hop = hop_low + hop_low.conj().T

    MAT = diags + hop

    return MAT

def TunnellingMatrix(size_n,size_m,r):
    """
    Tunneling matrix for WSM-Metal system
    Returns upper diagonal T^{\dagger}
    """
    Tun_lower = np.zeros((2*size_n,2*size_m),dtype=complex)
    Tun_lower[2*(size_n-1):2*size_n,0:2] = r * Pauli(0)

    # add to other sites to see if more states localize
    # Tun_lower[2*(size_n-2):2*(size_n-1),2*1:2*2] = r * Pauli(0)
    # Tun_lower[2*(size_n-3):2*(size_n-2),2*2:2*3] = r * Pauli(0)

    
    return Tun_lower

def FullHamiltonian(size,kx,kz,g,tm,mu,r):
    """
    Hamiltonian for Bulk WSM - Bulk Metal system
    """
    # size of each sample
    new_size = int(size/2)

    # diagonals
    HWSM = WeylHamiltonian(size=new_size,kx=kx,kz=kz,g=g)
    HMetal =  MetalHamiltonian(size=new_size,kx=kx,kz=kz,t=tm,mu=mu) 
    diags = np.kron((Pauli(0)+Pauli(3))/2,HWSM)+ np.kron((Pauli(0)-Pauli(3))/2,HMetal)

    # tunneling
    Tun_upper = TunnellingMatrix(new_size,new_size,r)
    off_diag = np.kron((Pauli(1)+1j*Pauli(2))/2,Tun_upper) + np.kron((Pauli(1)-1j*Pauli(2))/2,Tun_upper.conj().T) 

    MAT = diags + off_diag

    return MAT

def G_summ(G,spin,base,sgn,edge):
    # computes G_sum for given spin and side (base,sgn)
    # both spins
    if spin == 0:
        G_sum = G[base,base]+G[base+sgn*1,base+sgn*1]
        # add remaining edge states
        for i in range(1,edge):
            G_sum += G[base + sgn * (2*i),base + sgn * (2*i)] + G[base + sgn * (2*i+1),base + sgn * (2*i+1)]
    # spin up
    elif spin == +1:
        G_sum = G[base,base]
        # add remaining edge states
        for i in range(1,edge):
            G_sum += G[base + sgn * (2*i),base + sgn * (2*i)]
    # spin down
    elif spin == -1:
        G_sum = G[base+sgn*1,base+sgn*1]
        # add remaining edge states
        for i in range(1,edge):
            G_sum += G[base + sgn * (2*i+1),base + sgn * (2*i+1)]
    return G_sum

def FullSpectralFunction(w,size,kx,kz,g,tm,mu,r,spin=0,side=0):
    """
    Full spectral function calculation
    """
    G = np.linalg.inv(w * np.eye(2 * size) - FullHamiltonian(size,kx,kz,g,tm,mu,r))

    edge = int(size/size)

    # both sides
    if side == 0:
        # combine both cases
        # G_sum = G_summ(G,spin,0,+1,edge) + G_summ(G,spin,size-1,-1,edge)
        G_sum = np.trace(G)

    # left side
    elif side == -1:
        # we start from
        base = 0
        # and we add
        sgn = +1
        # G_sum
        G_sum = G_summ(G,spin,base,sgn,edge)

    # right side
    elif side == 1:
        # we start from
        base = size-1
        # and we subtract
        sgn = -1
        # G_sum
        G_sum = G_summ(G,spin,base,sgn,edge)

    A = -1 / np.pi * np.imag(G_sum)

    # A = - 1 / np.pi * np.imag(np.trace(G[2*(size-1):2*size,2*(size-1):2*size]))
    return A

def FullSpectralFunctionWeylWK(size,res,wrange,kx,kz,g=0,tm=1,mu=0,r=0.5,spin=0,side=0):
    """
    Return array for plot as a function of energy and momentum
    """
    # set up arrays
    ws = np.linspace(-wrange,wrange,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(ws)):
        w = ws[i] + 1j * 0.03
        A = FullSpectralFunction(w,size,kx,kz,g,tm,mu,r,spin,side)
        As[i] = A

    return As

def FullSpectralFunctionWeylKK(w,size,res,krange,kx,g=0,tm=1,mu=0,r=0.5,spin=0,side=0):
    """
    Return array for plot as a function of momentum and momnetum
    """
    # fix w
    w += 1j * 0.03
    # set up arrays
    kzs = np.linspace(-krange,krange,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(kzs)):
        kz = kzs[i]
        A = FullSpectralFunction(w,size,kx,kz,g,tm,mu,r,spin,side)
        As[i] = A

    return As 





