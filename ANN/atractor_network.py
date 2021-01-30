import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

#Construir la red neuronal
#el grafo se representará con una matriz 
#cada fila guardará los vecinos del nodo actual
#la conexión será aleatoria
def create_network(num_nodes = 100, p_conection=0.5):
    network = []
    weights_matrix = []
    for _ in range(num_nodes):
        neighbors = bernoulli.rvs(p_conection, size=num_nodes)    
        neighbors = np.where(neighbors==1)[0]
        #conexiones de la neurona
        network.append(neighbors)
        #pesos de las conexiones
        weights_matrix.append(np.zeros(len(neighbors)))
                
    return network,weights_matrix

#aprender un patron dado una red y el patrón
def learn(network,weights_matrix,pattern):
    #voy a recorrer los vecinos de cada nodo
    for i,neighbors in enumerate(network):
        #hebbian learning
        weights_matrix[i] += pattern[neighbors] * pattern[i]



def vect2matrix(idx,width):
    y = idx // width
    x = idx - y*width 
    return x,y


##############
#EJECUCION PRINCIPAL
##############


#########################
#ABRIR LA IMGAEN
###########################

#abrir una imagen en blanco y negro
img = cv2.imread("logo.png",0)

img = cv2.resize(img,(100,100))
#poner un límite para binarizar
threshold = img.mean()

#print(threshold)
img[np.where(img<threshold)]= 0
img[np.where(img>=threshold)]= 1
img = np.array(img, dtype='int8')
img[np.where(img==0)]=-1 

#aplanar imagen para la red neuronal
flat_img = np.resize(img,img.size)


#Esto serviría para pasar de la imagen plana a una matriz
#img = np.resize(flat_img,img_size)


############################
#CREACION DE LA RED
############################

#creo la red neuronal
network,weights_matrix = create_network(num_nodes=len(flat_img),p_conection=0.3)
#aprender el patrón
learn(network,weights_matrix,flat_img)


#############################
# RECONSTRUCCION DE LA IMAGEN
#############################

#reproduce la reconstrucción

random_img = flat_img.copy()
noise = bernoulli.rvs(0.8, size=len(random_img))    
random_img[np.where(noise==1)]=-1
noise = bernoulli.rvs(0.5, size=len(random_img))    
random_img[np.where(noise==1)]= 1
noise_img = np.resize(random_img,img.shape)

noise_img[np.where(noise_img==1)]=120
noise_img[np.where(noise_img==-1)]=0
width = len(noise_img)
#plt.imshow(noise_img)
#plt.show()

#noise_img= cv2.resize(noise_img,(500,500))
 
#las neuronas utilizarán los pesos para reconstruir la imagen
#voy a recorrer los vecinos de cada nodo
for i,neighbors in enumerate(network):
    #índicees de los vecinos
    #cambio de estado
    new_state = sum(random_img[neighbors]*weights_matrix[i])
    if(new_state<0):
        new_state = 0
    else:
        new_state = 120

    x,y = vect2matrix(i,width)

    #matriz de imagen
    noise_img[y,x] = new_state
    cv2.imshow('Frame', noise_img)
    keyboard = cv2.waitKey(30)

    if keyboard == 'q' or keyboard == 27:
        break  



