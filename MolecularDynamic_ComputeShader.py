import matplotlib.pyplot as plt # pour les graph
import numpy as np
import numexpr as ne # ne.evaluate("") si j ai compris favorise le multicoeur -> oui et pas que
import time # calcul du... temps de calcul
import moderngl
from PIL import Image
from imageio_ffmpeg import *
import gl_util
import instanced_rendering
import PDF
import scipy.signal as sc

# Ce programme utilise:
# le potentiel de Lennard-Jones
# la methode de Velocity-Verlet pour cacluler les positions
# la methode de velicity rescaling pour asservir la temperature

# --- paramètres ---------------------------------------------------------------------------
# parametres du potentiel de Lennard-Jones (unite SI)
# argon
sigma_a = 3.4e-10 #metre
epsilon_a = 1.65e-21 # joules
m_a = 6.69e-26 # kilogrammes
# position du minimum de potentiel
re_a = 2.0**(1.0/6.0)*sigma_a

x_a = 0.99

# neon
sigma_b = 2.7e-10 #metre
epsilon_b = 3.1e-21 # joules
m_b = 3.38e-26 # kilogrammes
# position du minimum de potentiel
re_b = 2.0**(1.0/6.0)*sigma_b


re = re_a


# règles de Kong
sigma_ab = (  (epsilon_a * sigma_a**12 * (1 + ((epsilon_b*sigma_b**12)/(epsilon_a*sigma_a**12))**(1/13))**13)  /
              (2**13 * np.sqrt(epsilon_b*sigma_b**6*epsilon_a*sigma_a**6)) )**(1/6)
epsilon_ab = np.sqrt(epsilon_b*sigma_b**6*epsilon_a*sigma_a**6)/sigma_ab**6


# rayon de coupure (limite standard = 2*re)
rcut = 2.0*re

# nombre d'atomes sur x
nbrex = 100
# nombre d'atomes sur y
nbrey = 100

# nombre de pas
npas = 200


# pour le film, afficher une image simule sur:
pfilm = 10
# enregistrer les images du film?
film = False

# afficher en direct ?
rendu_direct  = False

# temperature voulue, on peut programmer ce qu'on veut: ici un cosinus
Tini = 2 # temperature initiale visee kelvin
DeltaT = 200 # Kelvin amplitude
perioT = 1.0*npas # periode en pas de temps
gamma = 0.5 # parametre pour asservir la temperature ("potard")
betaC = True # True si la temperature est controlee, False sinon

# --- \paramètres ---------------------------------------------------------------------------



# l'affichage prenant trop de temps (et commenter tous les "print" aussi), on redéfinit print sur du rien pour les tests de vitesse
ppprint=print
print=lambda *a:0

# nombre d'atomes au total
npart = nbrex*nbrey

n_a = int(x_a*npart)

m = np.concatenate((m_a*np.ones(n_a), m_b*np.ones(npart-n_a)))


#max_buffer_size = gl_util.testMaxSizes()
max_buffer_size = 256 # à faire d'après les conseils d'Intel, après essais on gagne effectivement du temps
nombre_buffer = int(np.ceil(npart/max_buffer_size))
buffer_size = int(np.ceil(npart/nombre_buffer))
# on découpe pour le shader qui ne sait pas faire trop de choses à la fois


print("distance interatomique: %s Angstrom"%(re*1e10))
#
#
# periode d'oscillation pour pouvoir calibrer le pas de temps
puls0 = np.sqrt((57.1464*epsilon_a/(sigma_a**2.0))/m_a)
freq0 = puls0/(2.0*3.14159)
peri0 = 1/freq0
print("periode oscillation atomique: %s s"%(peri0))
print("periode oscillation atomique: %s ps"%(peri0*1e12))
#
# pas de temps: 
dt = peri0/75
print("pas de temps: %s ps"%(dt*1e12))
#
#
# limite droite de boite (position)
XlimD = (nbrex-1)*re+0.5*re
# limite gauche boite (position)
XlimG = -0.5*re
# limite haute de boite (position)
YlimH = (nbrey-1)*re+0.5*re
# limite basse boite (position)
YlimB = -0.5*re
# dimension de la boite de calcul
LengthX = XlimD-XlimG
LengthY = YlimH-YlimB
Volume = LengthX*LengthY*re # volume avec une epaisseur re
#
#
# calcul de la masse simulee
masseSim = m_a*n_a + m_b*(npart-n_a)
print("masse simulee: %s kg"%(masseSim))
# masse volumique
densi = masseSim/Volume
print("masse volumique: %s kg/m3"%(densi))
# messages
print("nombre atomes: %s"%(npart))
print("nombre pas de temps: %s"%(npas))
print("temps simule: %s ps"%(dt*1e12*npas))
print("dimension max du systeme: %s Angstrom"%((max(nbrex,nbrey)-1)*re*1e10))
print (' ')
#
#
# creation des positions initiales: maille carre de cote re
# positions selon x: on cree la grille reguliere
posx = np.concatenate([np.arange(0,nbrex,1.0) for i in range(nbrey)])*re
posx0 = posx # pour le calcul MSD on garde la position initiale en memoire
# positions selon y: on cree la grille reguliere
posy = np.concatenate([i*np.ones(nbrex) for i in range(nbrey)])*re
posy0 = posy # pour le calcul MSD

pos = np.transpose([posx, posy])
np.random.shuffle(pos)
pos0 = pos.copy()

#
# longueur x et y pour la periodicite
shiftX = LengthX-rcut # un atome a droite d'un autre, plus loin que ca, sera considere comme a gauche et vice-versa
shiftY = LengthY-rcut # meme principe verticalement
#
### calcul pour initier le premier pas (vitesses aleatoire)
# il n'est pas indispensable d avoir une vitesse non nulle, mais ca ajoute de la dynamique
# vitesse distribuee en loi normale (moyenne 0.0, ecart type 0.001)
vx =np.random.normal(0.0, 0.001, npart)
vy =np.random.normal(0.0, 0.001, npart)
v = np.transpose([vx, vy])
meanmV2 = np.mean(ne.evaluate("m*(vx*vx+vy*vy)"))
# constante de Boltzmann pour les calculs de temperatures
kB=1.38064852e-23 #unite SI
vfact = np.sqrt(2*kB*Tini/(meanmV2)) # pourconvertir la vitesse (l'energie cinetique) en temperature voulue
vx = vx*vfact
vy = vy*vfact
v *= vfact
# estimation de la temperature correspondante pour initier la rampe
EC = ne.evaluate("0.5*m*(vx*vx+vy*vy)")
EC = ne.evaluate("sum(EC)")
Tvoulue = ne.evaluate("2.0*EC/(kB*2.0*npart)")
print("temperature initiale: %s K"%(Tvoulue))
#
# trace des positions et vecteurs deplacement initiaux (deduit de la vitesse initiale)
plt.figure(1)
plt.quiver([np.mean(posx)],[np.mean(posy)],[np.mean(vx*dt)],[np.mean(vy*dt)],color='g',angles='xy', scale_units='xy', scale=1) # le deplacement global de toutes les particules
plt.quiver(posx,posy,vx*dt*200,vy*dt*200,(vx*vx+vy*vy),angles='xy', scale_units='xy', scale=1) # le deplacement de chaque particule
plt.plot(posx, posy,'ro',markersize=2)
plt.ylim(YlimB,YlimH)
plt.xlim(XlimG,XlimD)
plt.show(block=True) # true empeche l'excecution de la suite du programme avant fermeture de la fenetre
plt.close()
#
# les forces au premier pas sont nulles et de la dimension de posx (posy)
Fx = ne.evaluate("0.0*posx")
Fy = ne.evaluate("0.0*posy")
F = np.zeros(np.shape(pos))
#
# enregistrement des positions dans des fichiers,
# c pas propre mais ca fonctionne: a modifier
#fichx = open('storex.npy', 'w+b')
#np.save(fichx, posx)
#fichy = open('storey.npy', 'w+b')
#np.save(fichy, posy)
#
# initialisation listes fonctions du pas
#

m = np.transpose([m,m])

omegaT=2*np.pi/perioT
knparts=kB*npart
inv2nparts=1/(2.0*npart)
inv2npartsre=inv2nparts/re
dt2m=dt/(2*m)
limInf = np.array([XlimG, YlimB])
limSup = np.array([XlimD, YlimH])
length = np.array([LengthX, LengthY])


# Initialisation du contexte pour le GLSL
consts = {
    "X": buffer_size,
    "N_A": n_a,
    "NPART":npart,
    "RCUT":rcut,

    "EPSILONA":epsilon_a,
    "EPSILONB":epsilon_b,
    "EPSILONAB":epsilon_ab,

    "SIGMAA":sigma_a,
    "SIGMAB":sigma_b,
    "SIGMAAB":sigma_ab,

    "SHIFTX":shiftX,
    "SHIFTY":shiftY,
    "LENGTHX":LengthX,
    "LENGTHY":LengthY,
}


if rendu_direct:
    window = instanced_rendering.prepareRender()
    context = window.example.ctx
else:
    context = moderngl.create_standalone_context(require=430)
compute_shader = context.compute_shader(gl_util.source('./moldyn.glsl', consts))

BUFFER_P = context.buffer(reserve=8*npart)
BUFFER_P.bind_to_storage_buffer(0)

BUFFER_F = context.buffer(reserve=8*npart)
BUFFER_F.bind_to_storage_buffer(2)

BUFFER_E = context.buffer(reserve=4*npart)
BUFFER_E.bind_to_storage_buffer(3)

BUFFER_M = context.buffer(reserve=4*npart)
BUFFER_M.bind_to_storage_buffer(4)

BUFFER_PARAMS = context.buffer(reserve=4*5)
BUFFER_PARAMS.bind_to_storage_buffer(5)

if rendu_direct:
    window.example.vao = window.example.ctx.simple_vertex_array(window.example.prog, BUFFER_P, 'in_vert')
    window.example.npart = npart
    window.example.lengthx = LengthX
    window.example.lengthy = LengthY
    window.example.vao = window.example.ctx.simple_vertex_array(window.example.prog, BUFFER_P, 'in_vert')
    window.example.scale.value = (2/window.example.lengthx, 2/window.example.lengthy)
    window.example.n_a.value = n_a

pask = np.array(range(npas)) # le pas lui-meme
pasCPU = np.zeros(npas) # le temps de calcul par pas
pasEC = np.zeros(npas) # energie cinetique
pasEP = np.zeros(npas) # energie potentielle
pasET = np.zeros(npas) # energie totale
pasT = np.zeros(npas) # temperature
pasTC = 0.5*DeltaT*(1-np.cos(omegaT*pask))+1 # temperature de consigne
pastemps = pask*dt*1e12 # le temps
pasLiai = np.zeros(npas) # nbre liaisons par atome
pasMSD = np.zeros(npas) # MSD
tempstot = 0 #initialisation du chrono

mask = np.zeros(npart)

start_time = time.time()
current_time = start_time

if film:
    width, height = window.size
    figure_size = (width,height)
    gen = write_frames("debug.mp4", figure_size,fps=24,quality=9)
    gen.send(None)
    box=(0,0,figure_size[0],figure_size[1])


v2 = np.zeros(v.shape)

for k in range(npas):
    debutk = time.perf_counter() # calcul le temps de calcul
    print('%%%%%%%%%%%%%%')
    print(k)
    # methode de velocity-verlet
    # vitesse au demi pas de temps
    #v2 = ne.evaluate("v + F*dt2m")
    ne.evaluate("v + F*dt2m", out=v2)
    # nouvelles positions deduites
    #pos = ne.evaluate("pos + v2*dt")
    ne.evaluate("pos + v2*dt", out=pos)

    # Mean Square Displacement
    pasMSD[k] = ne.evaluate("sum((pos-pos0)**2)")*inv2npartsre
    
    # correction de systeme periodic: on replace les atomes trop a droite a gauche, etc...
    #pos = ne.evaluate("pos + (pos<limInf)*length - (pos>limSup)*length")
    ne.evaluate("pos + (pos<limInf)*length - (pos>limSup)*length", out=pos)

    # stockage des positions dans les fichiers deja cree
    #np.save(fichx, pos[:,0])
    #np.save(fichy, pos[:,1])

    BUFFER_P.write(pos.astype('f4').tobytes())

    compute_shader.run(group_x=nombre_buffer)


    # calcul energie cinetique
    EC = 0.5*ne.evaluate("sum(m*v*v)")
    print("Energie cinetique : %s J"%(EC)) # affichage


    Fgl = np.frombuffer(BUFFER_F.read(), dtype=np.float32)

    EPgl = np.frombuffer(BUFFER_E.read(), dtype=np.float32)

    F[:,0] = Fgl[::2]
    F[:,1] = Fgl[1::2]

    mask = np.frombuffer(BUFFER_M.read(), dtype=np.float32)

# caclul energie potentielle
    EP = 0.5*ne.evaluate("sum(EPgl)")

    mask_sum = ne.evaluate("sum(mask)")

    print("Energie potentielle : %s J"%(EP)) # affichage

    # calcul energie totale
    ET = EC+EP
    
    # calcul liaison par atome
    pasLiai[k] = mask_sum * inv2nparts
    
    # stockage des energies
    pasEC[k] = EC
    pasEP[k] = EP
    pasET[k] = ET
    print("Energie totale : %s J"%(EP)) # affichage
    
    # calcul et asservissement temperature
    T = EC/knparts
    print("temperature: %s K"%(T))
    # stockage de la temperature
    pasT[k] += T
    
    # calcul du nouveau vx ou vy (methode de Verlet-vitesse) pour le pas suivant
    if betaC:
        beta = np.sqrt(1+gamma*(pasTC[k]/T-1))
        #v = ne.evaluate("(v2 + (F*dt2m))*beta")
        ne.evaluate("(v2 + (F*dt2m))*beta", out=v)
    else:
        #v = ne.evaluate("v2 + (F*dt2m)")
        ne.evaluate("v2 + (F*dt2m)", out=v)

    # fin des calculs utiles a Verlet
    fink = time.perf_counter()# marqueur fin temps de calcul
    print("Temps par pas : %s s"%(fink-debutk))
    pasCPU[k] = (fink-debutk)*1000 # stockage du temps de calcul
    tempstot += fink-debutk # temps de calcul total
    print("temps total: %s s"%(tempstot))

    if k % pfilm == 0:
        ppprint(k)

        if rendu_direct:

            current_time, prev_time = time.time(), current_time
            frame_time = max(current_time - prev_time, 1 / 1000)
            window.render(current_time - start_time, frame_time)
            window.swap_buffers()

            if film:
                gen.send(Image.frombytes('RGB', context.fbo.size, context.fbo.read()).crop(box).transpose(Image.FLIP_TOP_BOTTOM).tobytes())

    
# fin de la grosse boucle
###############################################

if film:
    gen.close()



# fermeture des fichiers de positions
#fichx.close()
#fichy.close()


if rendu_direct:
    window.destroy()

###############################################
### tous les dessins
# dessin final (dernieres positions)
plt.figure(2)
plt.xlim(XlimG,XlimD)
plt.ylim(YlimB,YlimH)
plt.plot(pos[:n_a,0],pos[:n_a,1],'ro', markersize=2)
plt.plot(pos[n_a:,0],pos[n_a:,1],'bo', markersize=2)
plt.show()

# dessin du temps CPU
plt.figure(3)
#plt.ylim(0,1.05*max(pasCPU))
plt.plot(pask,pasCPU)
plt.xlabel('pas')
plt.ylabel('temps de calcul (ms)')
plt.show()
"""
# dessin de temperature
plt.figure(4)
line_T, = plt.plot(pastemps,pasT)
line_TC, = plt.plot(pastemps,pasTC)
plt.xlabel('temps (ps)')
plt.ylabel('Temperature (K)')
plt.legend([line_T, line_TC], ['T', 'T consigne'])
plt.show()

# dessin de MSD (mean square displacement)
plt.figure(5)
plt.plot(pasT,pasMSD)
plt.xlabel('Temperature (K)')
plt.ylabel('MSD')
plt.show()

# dessin de nbre liaison
plt.figure(6)
plt.plot(pasT,pasLiai)
plt.xlabel('Temperature K')
plt.ylabel('liaison par atome')
plt.show()

plt.figure(7)
plt.plot(pasEP,pasLiai)
plt.xlabel('Energie potentielle (J)')
plt.ylabel('liaison par atome')
plt.show()

# dessin des energies
plt.figure(8)
plt.subplot(211)
line_EC, = plt.plot(pask,pasEC)
plt.xlabel('pas')
plt.ylabel('Energies (J)')
plt.legend([line_EC], ['EC'])
plt.subplot(212)
line_EP, = plt.plot(pask,pasEP)
line_ET, = plt.plot(pask,pasET)
plt.xlabel('pas')
plt.ylabel('Energies (J)')
plt.legend([line_EP,line_ET], ['EP','ET'])
plt.show()
"""

bins = nbrex*2
H,xedges,yedges = np.histogram2d(pos[:,0], pos[:,1], bins=bins)
"""masque = np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1],
])"""
def masque_cercle(N):
    masque = np.ones((N,N))
    centre = (N-1)//2
    for i in range(N):
        for j in range(N):
            if i!=centre or j!=centre:
                masque[i,j] /= np.sqrt((i-centre)**2+(j-centre)**2)
    return masque
masque = masque_cercle(11)
joli_truc = sc.convolve2d(H,masque,mode='same',boundary='wrap')

plt.figure(9)
pc = plt.pcolormesh(xedges, yedges, joli_truc.T)
plt.contourf(xedges[:-1], yedges[:-1], joli_truc.T)
plt.xlim(xedges[0], xedges[-1])
plt.ylim(yedges[0], yedges[-1])
plt.plot(pos[:n_a,0],pos[:n_a,1],'ro', markersize=1)
plt.plot(pos[n_a:,0],pos[n_a:,1],'bo', markersize=1)
plt.show()

"""
plt.figure(10)
plt.plot(*PDF.PDF(pos,2000,1.0*rcut,50))
plt.show()
"""
#
# film de la simu: image tout les X pas de temps
""""
if film :
# ouverture pour la lecture
    imgs = []
    fix=open('storex.npy','rb')
    fiy=open('storey.npy','rb')
    # liste de k ou tracer le graph
    klist = range(0,npas,pfilm)
    # boucle pour creer le film
    figure_size = (1920, 1088)
    gen = write_frames("debug.mp4", figure_size,fps=24,quality=9)
    gen.send(None)
    for k in range(npas):
        posx = np.load(fix) # on charge a chaque pas de temps
        posy = np.load(fiy) # on charge a chaque pas de temps
        # dessin a chaque pas (ne s'ouvre pas: est sauvegarde de maniere incrementale)
        if k in klist:
            plt.figure(0, figsize=(figure_size[0] / (72*2), figure_size[1] / (72*2)))
            # definition du domaine de dessin
            plt.ioff() # pour ne pas afficher les graphs)
            plt.ylim(YlimB,YlimH)
            plt.xlim(XlimG,XlimD)
            plt.xlabel(k)
            plt.plot(posx,posy,'ro', markersize=0.5)
            temp = io.BytesIO()
            plt.savefig(temp, format='raw',dpi=72*2) # sauvegarde incrementale
            plt.clf()
            temp.seek(0)
            #imgs.append(Image.frombytes('RGBA', figure_size, temp.read()).convert('RGB'))
            ppprint(k)
            gen.send(Image.frombytes('RGBA', figure_size, temp.read()).convert('RGB').tobytes())

    #imageio.mimwrite(f"./debug.gif", imgs, "GIF", duration=0.05,subrectangles=True)
    gen.close()
# fin du film
    fix.close()
    fiy.close()
print ('%%%%%%%%%%%%%%%%%%%%%%')
"""