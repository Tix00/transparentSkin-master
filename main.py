# import dei moduli esterni
import cv2
from PIL import Image
# from matplotlib import pyplot as plt
import numpy as np
from imutils.video import FPS
import sys

# import dei moduli del progetto
import source.finalMask.getFinalMask as FM
import source.maskEyesLips.eyesAndLips as EL
import source.skinSegmentation.skinSegmentation as SS
import deeplabmodel as deeplabm
# import webcam_test as wt

"""
def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
"""

def replaceBackground(image, background, mask):
    #effettuiamo il resize del background
    background = cv2.resize(background, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
    #utilizziamo np al posto del for
    #image = np.where(mask == 1, background, image)
    #image = (image * mask)
    #mask = mask.astype(np.uint8)
    #image = np.multiply(image, mask, out=image, casting='unsafe')
    mask = mask[..., None]
    #mask = -(mask - 1)
    mask = np.where(mask == 0, 1, 0)
    #image = (image * mask)
    #image = np.multiply(image.all(), mask)
    image = np.where(mask == 0, background, image)
    """
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if mask[i, j] == 1:
                image[i, j] = background[i, j]
    """
    return image


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # otteniamo il path del video
    video_path = ""
    for i in range(1, len(sys.argv)):
        video_path = sys.argv[i]
        print("sys.argv[i]: ", sys.argv[i])

    img_background = cv2.imread("images/background.png")
    dlm = deeplabm.DeepLabModel("newmodeltrainval.pb")

    # iniziamo il frame per prendere tutti gli elementi
    # se non è stato passato alcun parametro, allora chiamiamo la webcam
    if video_path == "":
        print("Uso webcam")
        stream = cv2.VideoCapture(0)
    else:
        print("Uso video: ", video_path)
        stream = cv2.VideoCapture(video_path)
    fps = FPS().start()
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video file stream
        (grabbed, frame_cv2) = stream.read()
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
        # frame è in cv2, aka BGR
        # frame_RGB è quello in PIL
        frame_RGB = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
        frame_RGB = Image.fromarray(frame_RGB)
        # effettuiamo le operazioni per ottenere i nuovi frame
        frame_mask_RGB = dlm.run(frame_RGB)

        # 1. FRAME CON SOLO DEEPLABMODEL
        # la maschera prende le dimensioni del frame di partenza
        # finalMask = cv2.resize(frame_mask_RGB, (frame_cv2.shape[1], frame_cv2.shape[0]))
        frame_cv2 = cv2.resize(frame_cv2, (frame_mask_RGB.shape[1], frame_mask_RGB.shape[0]))
        finalMask = frame_mask_RGB
        print("Dimensioni frame_cv2: ", frame_cv2.shape)
        print("Dimensioni finalMask: ", finalMask.shape)

        """
        # 2. FRAME CON ANCHE maskEyesLips e SkinSegmentation
        maskEL = EL.eyesAndLips(frame_cv2)
        maskSS = SS.skinSegmentation(frame_cv2)
        finalMask, frame_cv2 = FM.getFinalMask(frame_cv2, frame_mask_RGB, maskEL, maskSS)
        """

        # rimpiazza il background e mostra a video
        finalFrame = replaceBackground(frame_cv2, img_background, finalMask)
        cv2.imshow("finalFrame", finalFrame)
        cv2.waitKey(1)
        fps.update()


    """
    # carica le immagini
    # img_original = cv2.imread("images/000002_image.png")
    # img_mask = cv2.imread("images/000002_prediction.png")

    img_original = cv2.imread("images/000008_image.png")
    img_original_PIL = Image.open("images/000008_image.png")
    img_mask = cv2.imread("images/000008_prediction.png")

    # img_original = cv2.imread("000011_image.png")
    # img_mask = cv2.imread("000011_prediction.png")
    # img_original = cv2.imread("002317_image.png")
    # img_mask = cv2.imread("002317_prediction.png")

    # chiamata a deeplabmodel
    dlm = deeplabm.DeepLabModel("newmodeltrainval.pb")
    img_mask = dlm.run(img_original_PIL)
    # print("tipo di img_mask: ", type(img_mask))
    # print("tipo di img_original: ", type(img_original))

    # riportiamo img_mask in cv2
    # img_mask = cv2.cvtColor(np.array(img_mask), cv2.COLOR_RGB2BGR)
    # img_mask = cv2.imread(img_mask)
    # img_mask = Image.fromarray(img_mask)
    # img_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGB2BGR)

    # cv2_img_mask = img_mask.astype(np.uint8) * 255                  #riporta la immagine in valori da a 1 a 255
    # np_img_mask = np.array(img_mask)
    # cv2_img_mask = cv2.cvtColor(np_img_mask, cv2.COLOR_RGB2BGR)
    # cv2_img_mask = np_img_mask[:, ::-1].copy()

    # cv2_img_mask = img_mask.astype(np.uint8) * 255                  #riporta la immagine in valori da a 1 a 255
    # cv2_img_mask = cv2.cvtColor(cv2_img_mask, cv2.COLOR_RGB2BGR)
    # img_mask = np.float32(img_mask)
    # img_mask = pil_to_cv2(img_mask)

    #cv2.imshow('cv2_img_mask', img_mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #for i, j in np.ndindex(cv2_img_mask.shape):
    #    print(cv2_img_mask[i, j])

    # plt.imshow(tmp1, interpolation='nearest')
    # plt.show()
    # plt.imshow(img_mask, interpolation='nearest')
    # plt.show()
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # iniziamo il processo di segmentazione
    # 1. otteniamo la maschera per occhi e labbra
    maskEL = EL.eyesAndLips(img_original)
    # cv2.imshow('maskEL', maskEL)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 2. otteniamo la maschera con sola pelle
    maskSS = SS.skinSegmentation(img_original)
    # cv2.imshow('maskSS', maskSS)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 3. creiamo la maschera finale
    finalmask = FM.getFinalMask(img_original, img_mask, maskEL, maskSS)
    #cv2.imshow('finalmask', finalmask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #plt.imshow(finalmask, interpolation='nearest')
    #plt.show()

    # 4. rimpiazziamo la immagine col background
    img_original = replaceBackground(img_original, img_background, finalmask)
    cv2.imshow('frame finale', img_original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    """