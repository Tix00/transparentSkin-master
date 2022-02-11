#import dei moduli esterni
import numpy as np
import cv2

#import dei moduli del progetto
from matplotlib import pyplot as plt


def fix_skin_mask(frame, final_mask):
    """
    cv2.imshow("frame", img_original)
    cv2.waitKey()

    cv2.imshow("maschera", img_mask)
    cv2.waitKey()

    cv2.imshow("", )
    cv2.waitKey()
    """

    #effettuiamo una closing
    #elemento_strutturante = np.ones((10, 10), np.uint8)        #usato con 002317_image.png
    #elemento_strutturante = np.ones((12, 12), np.uint8)
    #elemento_strutturante = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    #final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, elemento_strutturante)

    #cv2.imshow('Closing', final_mask)
    #cv2.waitKey()

    #effettuiamo una piccola dilation
    #elemento_strutturante = np.ones((3, 3), np.uint8)
    #final_mask = cv2.dilate(final_mask, elemento_strutturante, iterations=1)

    #cv2.imshow('Dilation', final_mask)
    #cv2.waitKey()

    #rendiamo la maschera VERAMENTE binaria
    #ret_threshold, final_mask = cv2.threshold(final_mask, 1, 255, cv2.THRESH_BINARY)
    #il valore del pixel rosso ora sarà [0, 0, 255]
    #print("valore di ret_threshold: ", ret_threshold)
    #cv2.imshow("fixed", final_mask)
    #cv2.waitKey()

    """
    #print dei valori
    for i in range(0, final_mask.shape[0]):
        for j in range(0, final_mask.shape[1]):
            print(final_mask[i,j])
    """
    """
    #LAVORAZIONE CON LA MASCHERA SEGMENTATA DI OCCHI E BOCCA
    #applyNewMask

    #isoliamo il canale red della maschera
    #final_mask = final_mask.getchannel(0)                # 0 = canale RED
    final_mask = final_mask[:, :, 2]
    #printElementsFromImage(final_mask)

    #sostituiamo il  alla immagine
    #result_image = cv2.bitwise_or(frame, final_mask, mask=None)
    frame = replace(frame, , final_mask)
    #cv2.imshow("immagine finale", frame)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    """
    return final_mask


def resizeImage_Mask(img_noMask, img_mask, maskEL, maskSS):
    #output di test
    #print('Original Dimensions img_original : ', img_original.shape)
    #print('Original Dimensions img_mask: ', img_mask.shape)

    #facciamo un resize delle 2 immagini, cosicche siano grandi uguali
    height_noMask = img_noMask.shape[0]
    width_noMask = img_noMask.shape[1]

    height_mask = img_mask.shape[0]
    width_mask = img_mask.shape[1]

    #facciamo il resize della immagine con piu pixel
    if ((height_noMask * width_noMask) > (height_mask * width_mask)):
        #noMask è piu grande
        img_noMask = cv2.resize(img_noMask, (width_mask, height_mask), interpolation = cv2.INTER_AREA)
        maskEL = cv2.resize(maskEL, (width_mask, height_mask), interpolation = cv2.INTER_AREA)
        maskSS = cv2.resize(maskSS, (width_mask, height_mask), interpolation = cv2.INTER_AREA)
    else:
        #mask è piu grande
        img_mask = cv2.resize(img_mask, (width_noMask, height_noMask), interpolation = cv2.INTER_AREA)
        maskEL = cv2.resize(maskEL, (width_noMask, height_noMask), interpolation = cv2.INTER_AREA)
        maskSS = cv2.resize(maskSS, (width_noMask, height_noMask), interpolation = cv2.INTER_AREA)

    #output di test
    #print('Resized Dimensions img_original : ', img_original.shape)
    #print('Resized Dimensions img_mask: ', img_mask.shape)
    return img_noMask, img_mask, maskEL, maskSS


def printElementsFromImage(image):
    for i, j in np.ndindex(image.shape):
        print(image[i, j])


#LAVORAZIONE CON LA MASCHERA SEGMENTATA DI OCCHI E BOCCA
def applyNewMask(maskIntera, maskMouthEyes, maskSkin):
    """
    input:
        maskIntera maschera totale
            valori validi: [0, 0, 255]
        maskMouthEyes maschera solo con bocca e occhi
            valori validi: [0, 0, 255]
        maskSkin maschera solo pelle
            valori validi: [???]

    output
        maskFinal maschera definitiva ultima e mo basta
            valori validi: [0, 0, 255]

    funzionamento
        CICLO per ogni valore delle maschere (avranno tutte le stesse dimensioni)
            SE maskIntera[i, j] == [0, 0, 255]
                SE maskSkin[i, j] != [???]
                    SE maskMouthEyes[i, j] == [0, 0, 255]
                        PIXEL VA TENUTO NELLA MASCHERA
                        continue
            PIXEL VA TOLTO DALLA MASCHERA
    """

    """
    for i in range(0, maskMouthEyes.shape[0]):
        for j in range(0, maskMouthEyes.shape[1]):
            #if maskMouthEyes[i, j] != [255, 255, 255]:
            if not np.array_equal(maskMouthEyes[i, j], [255, 255, 255]):
                #print("maskIntera i,j: ", i, j)
                print("maskMouthEyes[i, j]: ", maskMouthEyes[i, j])
    """

    """
    for i in range(0, maskSkin.shape[0]):
        for j in range(0, maskSkin.shape[1]):
            #if maskMouthEyes[i, j] != [255, 255, 255]:
            if not np.array_equal(maskSkin[i, j], [0, 0, 0]):
                #print("maskIntera i,j: ", i, j)
                print("maskSkin[i, j]: ", maskSkin[i, j])
    """

    """
    for i in range(0, maskIntera.shape[0]):
        for j in range(0, maskIntera.shape[1]):
            #if maskMouthEyes[i, j] != [255, 255, 255]:
            if not np.array_equal(maskIntera[i, j], [0, 0, 0]):
                #print("maskIntera i,j: ", i, j)
                print("maskIntera[i, j]: ", maskIntera[i, j])
    """

    # printElementsFromImage(maskIntera)
    # plt.imshow(maskIntera, interpolation='nearest')
    # plt.show()

    # 1 = pelle/occhi; 0 = non pelle
    #newMask = np.ones((maskIntera.shape[0], maskIntera.shape[1], 1))
    #newMask = np.zeros((maskIntera.shape[0], maskIntera.shape[1], 1))
    for i in range(0, maskIntera.shape[0]):
        for j in range(0, maskIntera.shape[1]):
            #if maskIntera[i, j] == [0, 0, 255]:
            #if np.array_equal(maskIntera[i, j], [0, 0, 255]):
            #if np.array_equal(maskIntera[i, j], 1):
            if maskIntera[i, j] == 1:
                #maskIntera[i, j] = 1
                #if not np.array_equal(maskSkin[i, j], [0, 0, 255]) and not np.array_equal(maskMouthEyes[i, j], [0, 0, 255]):
                if not np.array_equal(maskSkin[i, j], [0, 0, 255]) and not np.array_equal(maskMouthEyes[i, j], [0, 0, 255]):
                    #if maskMouthEyes[i, j].all == [0, 0, 255]:
                    #if np.array_equal(maskMouthEyes[i, j], [0, 0, 255]):
                    #print("newMask[i, j] = 255")
                    #newMask[i, j] = 1
                    #continue
                    maskIntera[i, j] = 0
            #newMask[i, j] = 255
            #maskIntera[i, j] = [0, 0, 0]

    #plt.imshow(maskIntera, interpolation='nearest')
    #plt.show()
    """
    cv2.imshow("maskIntera", maskIntera)
    cv2.imshow("maskSkin", maskSkin)
    cv2.imshow("maskMouthEyes", maskMouthEyes)
    cv2.imshow("newMask", newMask)
    cv2.waitKey()
    cv2.destroyAllWindows()
    """

    #return maskIntera
    return maskIntera


#if __name__ == '__main__':
def getFinalMask(img_noMask, img_mask, maskEL, maskSS):
    img_noMask, img_mask, maskEL, maskSS = resizeImage_Mask(img_noMask, img_mask, maskEL, maskSS)

    print("Resized Dimensions img_original: ", img_noMask.shape)
    print("Resized Dimensions img_mask: ", img_mask.shape)
    print("Resized Dimensions maskEL: ", maskEL.shape)
    print("Resized Dimensions maskEL: ", maskSS.shape)

    img_mask = fix_skin_mask(img_noMask, img_mask)
    """
    debug_mask = applyNewMask(img_mask, maskEL, maskSS)
    height, width = debug_mask.shape
    print("debug_mask.height: ", height)
    print("debug_mask.width: ", width)
    return debug_mask
    """
    return applyNewMask(img_mask, maskEL, maskSS), img_noMask
    #return applyNewMask(img_mask, maskEL, maskSS)
