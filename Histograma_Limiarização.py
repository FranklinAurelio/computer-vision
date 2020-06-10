import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('lion.jpg',1)
# o parametro 1 depois do arquivo da imagem quer dizer que a imagem e colorida
# ele poderia ser trocado para 0 para preto e branco e para -1 com transparencia
print(img.shape)
# printa a imagem 
cv.imshow('image:', img)
# o comando imshow abre a imagem
# o parametro passado e o titulo da janela que sera aberta
cv.waitKey(0)
# como o opencv apos abrir uma imagem sempre destroi a janela, usamos o comando waitkey para so fechar/destruir a janela apos pressionar algum botao
cv.destroyAllWindows()
# outra maneira de abrir a imagem e usando o matplotlib
RGBimg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# como o matplotlib usa canais de cores diferente, opencv(BGR), matplotlib(RGB), usamos o cvtColor para converter para RGB
plt.imshow(RGBimg)

######PROCESAMENTO#######

#processamento em preto e branco 
grayimg = cv.imread('lion.jpg', 0)
print(grayimg.shape)
plt.imshow(grayimg, cmap='gray')
# o paramentro adicional na matplot e para reconhecer que a imagem e cinza
cv.imshow('Gray Image:', grayimg)
cv.waitKey(0)
cv.destroyAllWindows()

###Gerar o histograma###

#tons de cinza(preto e branco)
grayhist = cv.calcHist([grayimg], [0], None, [256], [0, 255])
#os parametros da funcao em ordem sao 
#todos os paremetros devem ser entre colchetes, exceto os vazios como no caso do terceiro 
#1_ imagem onde se calcula o histograma
#2_ canal de cor(no caso 0 por ser cinza)
#3_ mascara
#4_ quantidade de barras do histograma
#5_ faixa de valores
plt.plot(grayhist)

#colorida
color = ('b', 'g', 'r')
#lista das cores BGR 
for i,clr in enumerate(color):
#for para gerar de todas as cores    
    hist = cv.calcHist([img], [i], None, [256], [0, 255])
    #gerando o histograma
    plt.plot(hist, color = clr)
    #printando o histograma, o segundo parametro e para  cada cor 
    plt.xlim([0,255])
    #corta os valores do eixo x antes e depois dos paramentros
plt.show()

#equalizacao
eqgrayimg = cv.equalizeHist(grayimg)
plt.imshow(eqgrayimg, cmap='gray')

### picos e vales ###

peak_bin = 0
#variavel que recebera a posicao do pico
peak_value = grayhist[0]
#variavel que recebera o valor do pico
valley_bin = 0
#variavel que recebera a posicao do vale
valley_value = grayhist[0]
#variavel que recebera o alor do vale
for bin in range(256):
#for que percorrera a imagem delimitado a um range de 256
    if peak_value < grayhist[bin]:
    #se o valor do pico na posicao x for menor que o valor do pico na posicao x+1
        peak_value = grayhist[bin]
        #o valr do pico recebera o novo valor
        peak_bin = bin
        #sera armazenada a nova posicao do pico
    if valley_value > grayhist[bin] and grayhist[bin] != 0:
    #se o valor do vale na posixao x for maior que x+1 e (x+1 != 0)
        valley_value = grayhist[bin]
        #o valor do vale recebera o novo vale
        valley_bin = bin
        #sera armazenada a nova posicao do vale
print("Histogram peak at ", peak_bin, " with value of ", peak_value)
print("Histogram valley at ", valley_bin, " with value of ", valley_value)

### Funcao para calcular pico ###

def peak_valley_value(x)
#onde x é o histograma da imagem
peak_bin = 0
#variavel que recebera a posicao do pico
peak_value = x[0]
#variavel que recebera o valor do pico
valley_bin = 0
#variavel que recebera a posicao do vale
valley_value = x[0]
#variavel que recebera o alor do vale
for bin in range(256):
#for que percorrera a imagem delimitado a um range de 256
    if peak_value < x[bin]:
    #se o valor do pico na posicao x for menor que o valor do pico na posicao x+1
        peak_value = x[bin]
        #o valr do pico recebera o novo valor
        peak_bin = bin
        #sera armazenada a nova posicao do pico
    if valley_value > x[bin] and grayhist[bin] != 0:
    #se o valor do vale na posixao x for maior que x+1 e (x+1 != 0)
        valley_value = x[bin]
        #o valor do vale recebera o novo vale
        valley_bin = bin
        #sera armazenada a nova posicao do vale
print("Histogram peak at ", peak_bin, " with value of ", peak_value)
print("Histogram valley at ", valley_bin, " with value of ", valley_value)

#chamando a funcao
peak_valley_value(grayhist)

### Limiarizacao ###

ret, thres1 = cv.threshold(grayimg, peak_bin, 255, cv.THRESH_BINARY)
ret, thres2 = cv.threshold(grayimg, peak_bin, 255, cv.THRESH_BINARY_INV)
fig = plt.figure(figsize=(10, 10))
#o comando figure permite criar uma matriz para exibicao simultanea da imagem
fig.add_subplot(2, 1, 1)
# aqui indicamos onde sera adicionada cada figura levando em consideracao o numero de linhas, colunas e posicao global
plt.imshow(thres1, cmap='gray')
#aqui plotamos a imagem
fig.add_subplot(2, 1, 2)
# aqui indicamos onde sera adicionada cada figura levando em consideracao o numero de linhas, colunas e posicao global
plt.imshow(thres2, cmap='gray')
#aqui plotamos a imagem

### limiar de Otsu ###

ret, thres3 = cv.threshold(grayimg, 0, 255, cv.THRESH_OTSU)
#o limiar de otsu e aplicado da mesma forma so muda a flag _OTSU
plt.imshow(thres3, cmap='gray')
