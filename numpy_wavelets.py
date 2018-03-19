import numpy as np
import pywt
from PIL import Image
import math
import matplotlib.pyplot as plt


# Haar matrix is now orthogonal!
def build_haar_matrix(row_size):
    size = (int)(row_size / 2)
    haarM1 = np.zeros((size, row_size))
    for i in range(size):
        haarM1[i][2 * i] = math.sqrt(2) / 2
        haarM1[i][2 * i + 1] = math.sqrt(2) / 2

    haarM2 = np.zeros((size, row_size))
    for i in range(size):
        haarM2[i][2 * i] = math.sqrt(2) / 2
        haarM2[i][2 * i + 1] = -math.sqrt(2) / 2

    m = np.concatenate((haarM1, haarM2), axis=0)
    return m


def dwt(image):
    if not (image.shape[0] % 4 == 0) or not (image.shape[1] % 4 == 0):
        print("The size of your image must bea multiple of 4!")
        return None

    haarMatrix = build_haar_matrix(image.shape[0])
    haarMatrix2 = haarMatrix.T
    result = np.zeros(image.shape, dtype=np.float)
    nchannels = 3
    if len(image.shape) == nchannels:
        for channel in range(nchannels):
           data0 = image[:, :, channel]
           result0 = np.dot(np.dot(haarMatrix, data0), haarMatrix.T)
           result[:, :, channel] = result0
        #result = np.dot(np.dot(haarMatrix, image), haarMatrix2)
    else:

        result = np.array(np.matmul(np.matmul(haarMatrix, image), haarMatrix.T), dtype=np.int32)

    return result


def idwt(fdwt):
    haarMatrix = build_haar_matrix(fdwt.shape[0])
    idw = haarMatrix.T
    result = np.zeros(fdwt.shape, dtype=np.float)
    nchannels = 3
    if len(fdwt.shape) == nchannels:
        for channel in range(nchannels):
            data0 = fdwt[:, :, channel]
            result0 = np.dot(np.dot(haarMatrix.T, data0), haarMatrix)
            result[:, :, channel] = result0
    else:

        result = np.array(np.matmul(np.matmul(haarMatrix.T, fdwt), haarMatrix), dtype=np.int32)

    return result


def library_wavelet(image):
    if not (image.shape[0] % 4 == 0) or not (image.shape[1] % 4 == 0):
        print("The size of your image must bea multiple of 4!")
        return None

    wav_result = np.zeros(image.shape)
    print
    len(image.shape)
    nchannels = 3
    if len(image.shape) == nchannels:
        for channel in range(nchannels):
            data0 = image[:, :, channel]

            coeffs0 = pywt.dwt2(data0, 'haar')
            cA, (cH, cV, cD) = coeffs0
            aux0 = np.concatenate((cA, cV), axis=1)
            aux1 = np.concatenate((cH, cD), axis=1)
            wav_result0 = np.concatenate((aux0, aux1), axis=0)

            wav_result[:, :, channel] = wav_result0

    else:
        coeffs = pywt.dwt2(image, 'haar')
        cA, (cH, cV, cD) = coeffs
        aux0 = np.concatenate((cA, cV), axis=1)
        aux1 = np.concatenate((cH, cD), axis=1)
        wav_result = np.concatenate((aux0, aux1), axis=0)

    return wav_result


def main():
    img = Image.open("mockup.jpg")
    img.load()
    data = np.asarray(img, dtype="int32")

    result = dwt(data)
    if result.any():
        # plt.imshow(result)
        # plt.show()

        wav_result = library_wavelet(data)
        inv = idwt(result)
        # ==============================================================================
        #         print(np.array_equal(wav_result, result))
        # ==============================================================================
        print
        np.sum(np.asarray(np.clip(wav_result, 0, 255) - np.clip(result, 0, 255)), dtype="uint8") == 0
        img = Image.fromarray(np.asarray(np.clip(result, 0, 255), dtype="uint8"))
        img.save("mockup4.jpg")
        img = Image.fromarray(np.asarray(np.clip(wav_result, 0, 255), dtype="uint8"))
        img.save("mockup3_wav.jpg")
        img = Image.fromarray(np.asarray(np.clip(np.subtract(result, wav_result), 0, 255), dtype="uint8"))
        img.save("mockup3_subtract.jpg")
        img = Image.fromarray(np.asarray(np.clip(inv, 0, 255), dtype="uint8"))
        img.save("mockup_inv.jpg")

main()