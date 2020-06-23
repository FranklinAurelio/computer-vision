import numpy as np
from collections import deque

class Watershed(object):
   MASK = -2
   WSHD = 0
   INIT = -1
   INQE = -3

   def __init__(self, levels = 256):
      self.levels = levels

   # Pixels vizinhos (coordenadas de) pixels, incluindo o pixel fornecido.
   #localizacao (coordenadas dos pixels do mapeamento)
   def _get_neighbors(self, height, width, pixel):
      return np.mgrid[
         max(0, pixel[0] - 1):min(height, pixel[0] + 2),
         max(0, pixel[1] - 1):min(width, pixel[1] + 2)
      ].reshape(2, -1).T

   def apply(self, image):
      current_label = 0
      flag = False
      fifo = deque()

      height, width = image.shape
      total = height * width
      labels = np.full((height, width), self.INIT, np.int32)

      reshaped_image = image.reshape(total)
      # [y, x] pares de coordenadas de pixel da imagem achatada.
      pixels = np.mgrid[0:height, 0:width].reshape(2, -1).T
      # Coordenadas de pixels vizinhos para cada pixel.
      neighbours = np.array([self._get_neighbors(height, width, p) for p in pixels])
      if len(neighbours.shape) == 3:
         # Caso em que todos os pixels têm o mesmo número de vizinhos.
         neighbours = neighbours.reshape(height, width, -1, 2)
      else:
         # Caso em que pixels podem ter um número diferente de pixels vizinhos.
         neighbours = neighbours.reshape(height, width)

      indices = np.argsort(reshaped_image)
      sorted_image = reshaped_image[indices]
      sorted_pixels = pixels[indices]

      # self.levels etapas uniformemente espaçadas do mínimo ao máximo.
      levels = np.linspace(sorted_image[0], sorted_image[-1], self.levels)
      level_indices = []
      current_level = 0

      # Obtenha os índices que delimitam pixels com valores diferentes.
      for i in xrange(total):
         if sorted_image[i] > levels[current_level]:
            # Pule os níveis até que o próximo nível mais alto seja alcançado.
            while sorted_image[i] > levels[current_level]: current_level += 1
            level_indices.append(i)
      level_indices.append(total)

      start_index = 0
      for stop_index in level_indices:
         # Mascarar todos os pixels no nível atual.
         for p in sorted_pixels[start_index:stop_index]:
            labels[p[0], p[1]] = self.MASK
            # Inicialize a fila com vizinhos de bacias existentes no nível atual.
            for q in neighbours[p[0], p[1]]:
               # p == q é ignorado aqui porque rótulos [p] < WSHD
               if labels[q[0], q[1]] >= self.WSHD:
                  labels[p[0], p[1]] = self.INQE
                  fifo.append(p)
                  break

         # Estender bacias.
         while fifo:
            p = fifo.popleft()
            # Rotule p inspecionando vizinhos.
            for q in neighbours[p[0], p[1]]:
               # Não defina lab_p no loop externo, pois isso pode mudar.
               lab_p = labels[p[0], p[1]]
               lab_q = labels[q[0], q[1]]
               if lab_q > 0:
                  if lab_p == self.INQE or (lab_p == self.WSHD and flag):
                     labels[p[0], p[1]] = lab_q
                  elif lab_p > 0 and lab_p != lab_q:
                     labels[p[0], p[1]] = self.WSHD
                     flag = False
               elif lab_q == self.WSHD:
                  if lab_p == self.INQE:
                     labels[p[0], p[1]] = self.WSHD
                     flag = True
               elif lab_q == self.MASK:
                  labels[q[0], q[1]] = self.INQE
                  fifo.append(q)

         # Detectar e processar novos mínimos no nível atual.
         for p in sorted_pixels[start_index:stop_index]:
            # p está dentro de um novo mínimo. Crie um novo rótulo.
            if labels[p[0], p[1]] == self.MASK:
               current_label += 1
               fifo.append(p)
               labels[p[0], p[1]] = current_label
               while fifo:
                  q = fifo.popleft()
                  for r in neighbours[q[0], q[1]]:
                     if labels[r[0], r[1]] == self.MASK:
                        fifo.append(r)
                        labels[r[0], r[1]] = current_label

         start_index = stop_index

      return labels

if __name__ == "__main__":
   import sys
   from PIL import Image
   import matplotlib.pyplot as plt
   from scipy.misc import imsave

   w = Watershed()
   image = np.array(Image.open(sys.argv[1]))
   labels = w.apply(image)
   imsave('image.png', labels)
   plt.imshow(labels, cmap='Paired', interpolation='nearest')
   plt.show()
