import cv2

def main(args):
    limiar_inferior = 30
    limiar_superior = 90
    
    tamanho_kernel = 5
    
    #load image
    
    img = cv2.imread('imgage.png')
    cv2.imshow('algoritmoDeCanny',img)
    img1 = cv2.Canny(img, limiar_inferior, limiar_superior, tamanho_kernel)
    cv2.imshow('algoritmoDeCanny1',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))

