import cv2
import os
import numpy as np

import argparse

#import dei moduli del progetto
#from test import evaluate                  #non funziona
from source.maskEyesLips.test import evaluate


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--img-path', default='images/img1.jpg')
    return parse.parse_args()


def mask(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    # image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    # print (parsing)
    # print (part)

    """
    if(parsing.any() != part):
        changed[True] = image[True]
    else:
        changed[False] = image[False]
    """

    # print(parsing.shape)
    # print(image.shape)

    """
    print(parsing != part)
    tmp1 = parsing != part
    print("fine print")
    a = image[tmp1]
    changed[parsing != part] = a
    """

    changed[parsing != part] = image[parsing != part]

    return changed


# if __name__ == '__main__':
# 1  face
# 4  left eye
# 5  right eye
# 12 upper lip
# 13 lower lip
def eyesAndLips(image):
    # args = parse_args()

    table = {
        'left_eye': 4,
        'right_eye': 5,
        'upper_lip': 12,
        'lower_lip': 13
    }

    # image_path = args.img_path
    cp = "source/maskEyesLips/cp/79999_iter.pth"
    #controlliamo che ci sia il file
    print("Il path di 79999_iter.pth è corretto?", os.path.exists(cp))

    """
    try:
        # image = cv2.imread(image_path)
        ori = image.copy()
        im2 = image.copy()
        im2 = cv2.rectangle(im2, (0, 0), (1080, 1080), (255, 255, 255), thickness=1080)
    except AttributeError:
        print('Image not found. Please enter a valid path.')
        quit()
    """
    # image = cv2.imread(image_path)
    ori = image.copy()
    im2 = image.copy()
    im2 = cv2.rectangle(im2, (0, 0), (1080, 1080), (255, 255, 255), thickness=1080)


    # ho spostato sto codice perche se no non funziona

    # MIO CODICE, RESIZIAMO LA IMMAGINE
    # image = cv2.resize(image, (513, 513))
    original_shape = (image.shape[1], image.shape[0])
    # parsing = evaluate(image_path, cp)
    # parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)
    parsing = evaluate(image, cp)
    #parsing = cv2.resize(parsing, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    parsing = cv2.resize(parsing, original_shape, interpolation = cv2.INTER_NEAREST)

    parts = [table['left_eye'], table['right_eye'], table['upper_lip'], table['lower_lip']]
    # valore della maschera
    # color = [255, 0, 139]
    color = [0, 0, 255]

    for part in parts:
        image = mask(image, parsing, part, color)
        im2 = mask(im2, parsing, part, color)

    """
    cv2.imshow('image', cv2.resize(ori, (512, 512)))
    cv2.imshow('mask', cv2.resize(im2, (512, 512)))
    cv2.imshow('color', cv2.resize(image, (512, 512)))
    """

    #ori = cv2.resize(ori, original_shape)
    im2 = cv2.resize(im2, original_shape)
    #image = cv2.resize(image, original_shape)

    """
    cv2.imshow('image', ori)
    cv2.imshow('mask', im2)
    cv2.imshow('color', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    """
    for i, j in np.ndindex(image.shape):
        print(image[i, j])
    """

    """
    # printiamo la maschera
    for i in range(0, im2.shape[0]):
        for j in range(0, im2.shape[1]):
            # if im2[i, j] != [255, 255, 255]:
            print(im2[i, j])
    """

    #im2 è la maschera
    return im2


if __name__ == "__main__":
    args = parse_args()
    image_path = args.img_path
    image = cv2.imread(image_path)
    eyesAndLips(image)
