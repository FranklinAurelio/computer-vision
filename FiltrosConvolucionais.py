import cv2 as cv
import math
import numpy as np
from . import helper as hp


def kernel_filter(path, mask_path):
	img = cv.imread(path, 3)
	row, col, ch = img.shape
	mask = hp.rebate_mask(np.loadtxt(mask_path))
	row_mask, col_mask = mask.shape

	inc = row_mask // 2
	r_ext, g_ext, b_ext = hp.create_extended_matrixes(img, inc)
	r_matrix, g_matrix, b_matrix = hp.create_rgb_matrixes(row, col)

	for i in range(row):
		for j in range(col):
			red_pixel = 0
			green_pixel = 0
			blue_pixel = 0
			for x in range(row_mask):
				for y in range(col_mask):
					red_pixel += r_ext[i+x, j+y] * mask[x, y]
					green_pixel += g_ext[i+x, j+y] * mask[x, y]
					blue_pixel += b_ext[i+x, j+y] * mask[x, y]
			r_matrix[i, j] = red_pixel
			g_matrix[i, j] = green_pixel
			b_matrix[i, j] = blue_pixel
	kernel_image = cv.merge([r_matrix, g_matrix, b_matrix])
	hp.show_image('Kernel Filter', kernel_image)

	np.savetxt("./assets/files/red_extended_matrix.txt", r_ext, fmt="%s")
	np.savetxt("./assets/files/green_extended_matrix.txt", g_ext, fmt="%s")
	np.savetxt("./assets/files/blue_extended_matrix.txt", b_ext, fmt="%s")
	np.savetxt("./assets/files/red_final_matrix.txt", r_matrix, fmt="%s")
	np.savetxt("./assets/files/green_final_matrix.txt", g_matrix, fmt="%s")
	np.savetxt("./assets/files/blue_final_matrix.txt", b_matrix, fmt="%s")


def median_filter(path, mask):
	img = cv.imread(path, 3)
	row, col, ch = img.shape
	mask = hp.rebate_mask(np.loadtxt(mask))
	row_mask, col_mask = mask.shape

	inc = row_mask // 2
	r_ext, g_ext, b_ext = hp.create_extended_matrixes(img, inc)
	r_matrix, g_matrix, b_matrix = hp.create_rgb_matrixes(row, col)

	for i in range(row):
		for j in range(col):
			red_aux = np.sort(np.array(r_ext[i:i+row_mask, j:j+col_mask]).flatten())
			green_aux = np.sort(np.array(g_ext[i:i+row_mask, j:j+col_mask]).flatten())
			blue_aux = np.sort(np.array(g_ext[i:i+row_mask, j:j+col_mask]).flatten())

			index = red_aux.shape[0] // 2
			r_matrix[i, j] = red_aux[index]
			g_matrix[i, j] = green_aux[index]
			b_matrix[i, j] = blue_aux[index]

	median_image = cv.merge([r_matrix, g_matrix, b_matrix])
	hp.show_image('Median Filter', median_image)


def sobel_filter(path, mask):
	img = cv.imread(path, 3)
	row, col, ch = img.shape
	mask = np.loadtxt(mask)
	row_mask, col_mask = mask.shape
	sobel_ver = hp.rebate_mask(mask)
	sobel_hor = hp.rebate_mask(np.rot90(mask))

	inc = row_mask // 2
	r_ext, g_ext, b_ext = hp.create_extended_matrixes(img, inc)
	r_matrix, g_matrix, b_matrix = hp.create_rgb_matrixes(row, col)

	for i in range(row):
		for j in range(col):
			red_pixel_hor = 0
			red_pixel_ver = 0
			green_pixel_hor = 0
			green_pixel_ver = 0
			blue_pixel_hor = 0
			blue_pixel_ver = 0
			for x in range(row_mask):
				for y in range(col_mask):
					red_pixel_hor += r_ext[i+x, j+y] * sobel_hor[x, y]
					red_pixel_ver += r_ext[i+x, j+y] * sobel_ver[x, y]
					green_pixel_hor += g_ext[i+x, j+y] * sobel_hor[x, y]
					green_pixel_ver += g_ext[i+x, j+y] * sobel_ver[x, y]
					blue_pixel_hor += b_ext[i+x, j+y] * sobel_hor[x, y]
					blue_pixel_ver += b_ext[i+x, j+y] * sobel_ver[x, y]
			
			r_matrix[i, j] = abs(red_pixel_hor) + abs(red_pixel_ver)
			g_matrix[i, j] = abs(green_pixel_hor) + abs(green_pixel_ver)
			b_matrix[i, j] = abs(blue_pixel_hor) + abs(blue_pixel_ver)
			
	sobel_image = cv.merge([r_matrix, g_matrix, b_matrix])
	hp.show_image('Sobel Filter', sobel_image)