import cv2 as cv
import numpy as np
import math
from dijkstar import Graph, find_path
import time as t
# Pressione e segure o mouse para selecionar pontos
# Pressione o botão esquerdo do mouse para o primeiro ponto
# Solte o clique do mouse para o segundo ponto
#code começa a partir daqui
def click(event, x, y, flags, param):
    global retPt
    # se o botão esquerdo do mouse foi clicado, registre o início
	# (x, y) coordenadas
    if event == cv.EVENT_LBUTTONDOWN:
        retPt = [(x, y)]
    elif event == cv.EVENT_LBUTTONUP:
        retPt.append((x, y))
	# gravar as coordenadas finais (x, y)
    
    
start_time = t.time()
img = cv.imread("image.jpge, 1)
final = img.copy()
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
vertex = img.copy()
h,w = img.shape[1::-1]

graph = Graph(undirected=True)

# Iterando uma Imagem e Evitando Limites
for i in range (1, w-1):
    for j in range(1, h-1):
        G_x = float(vertex[i,j]) - float(vertex[i,j+1])    # Center - right
        G_y = float(vertex[i,j]) - float(vertex[i+1, j])   # Center - bottom
        G = np.sqrt((G_x)**2 + (G_y)**2)
        if (G_x > 0 or G_x < 0):
            theeta = math.atan(G_y/G_x)
        # Theeta é girado no sentido horário (90 graus) para alinhar com a borda
        theeta_a = theeta + math.pi/2
        G_x_a = abs(G * math.cos(theeta_a)) + 0.00001
        G_y_a = abs(G * math.sin(theeta_a)) + 0.00001
        
        # O Edge mais forte terá os pesos mais baixos
        W_x = 1/G_x_a
        W_y = 1/G_y_a
        
        # Atribuindo pesos
        graph.add_edge((i,j), (i,j+1), W_x) # W_x é dado à direita do vértice atual
        graph.add_edge((i,j), (i+1,j), W_y) # W_y é dado na parte inferior do vértice atual
        
        
print (graph.node_count)
print (graph.edge_count)
print ("Tempo necessário para transformar a imagem em gráfico: {}".format(t.time()-start_time))
# Abre a imagem, selecione os pontos usando o mouse e pressione c para concluir
cv.namedWindow("image")
while True:
    while True:
        cv.setMouseCallback("image", click)
        cv.imshow("image", final)
        key = cv.waitKey(2) & 0xFF
        if key == ord("c"):
            #cv.destroyWindow("image")
            break
    # Obtém o ponto inicial e final nos formatos de imagem
    print(retPt[0][1], retPt[0][0])
    print(retPt[1][1], retPt[1][0])       
    startPt = (retPt[0][1], retPt[0][0])
    endPt = (retPt[1][1], retPt[1][0])
    
    # Find_path[0] nós de retorno que viajou para o caminho mais curto
    path = find_path(graph,startPt,endPt)[0]
    if path is None:
        break
    # Transforme esses nós visitados em branco
    for i in range(0,len(path)):
        final[path[i][0], path[i][1]] = 255
        
    cv.imshow('ImageWindow', final)
    cv.waitKey(0)
    cv.destroyWindow('ImageWindow')
