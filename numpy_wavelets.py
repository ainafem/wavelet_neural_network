import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt



def build_haar_matrix(row_size):
    size = (int)(row_size/ 2)
    haarM1 = np.zeros((size, row_size))
    for i in range(size):
        haarM1[i][2 * i] = 0.5
        haarM1[i][2 * i + 1] = 0.5

    haarM2 = np.zeros((size, row_size))
    for i in range(size):
        haarM2[i][2 * i] = -0.5
        haarM2[i][2 * i + 1] = 0.5

    m = np.concatenate((haarM1, haarM2), axis=0)
    print(m)
    return m

def dwt(image):
    if not (image.shape[0] % 4 == 0) or not (image.shape[1] % 4 == 0):
        print("The size of your image must bea multiple of 4!")
        return None

    haarMatrix = build_haar_matrix(image.shape[0])

    result = np.zeros(image.shape)
    if len(image.shape) == 3:
        data0 = image[:, :, 0]
        data1 = image[:, :, 1]
        data2 = image[:, :, 2]

        result0 = np.matmul(np.matmul(haarMatrix, data0), haarMatrix.T)
        result1 = np.matmul(np.matmul(haarMatrix, data1), haarMatrix.T)
        result2 = np.matmul(np.matmul(haarMatrix, data2), haarMatrix.T)

        result[:, :, 0] = result0
        result[:, :, 1] = result1
        result[:, :, 2] = result2
    else:

        result = np.matmul(np.matmul(haarMatrix, image), haarMatrix.T)

    return result


def library_wavelet(image):
    if not (image.shape[0] % 4 == 0) or not (image.shape[1] % 4 == 0):
        print("The size of your image must bea multiple of 4!")
        return None

    wav_result = np.zeros(image.shape)
    if len(image.shape) == 3:
        data0 = image[:, :, 0]
        data1 = image[:, :, 1]
        data2 = image[:, :, 2]

        coeffs0 = pywt.dwt2(data0, 'haar')
        cA, (cH, cV, cD) = coeffs0
        aux0 = np.concatenate((cA, cH), axis=1);
        aux1 = np.concatenate((cV, cD), axis=1);
        wav_result0 = np.concatenate((aux0, aux1), axis=0);

        coeffs1 = pywt.dwt2(data1, 'haar')
        cA, (cH, cV, cD) = coeffs1
        aux0 = np.concatenate((cA, cH), axis=1);
        aux1 = np.concatenate((cV, cD), axis=1);
        wav_result1 = np.concatenate((aux0, aux1), axis=0);

        coeffs2 = pywt.dwt2(data2, 'haar')
        cA, (cH, cV, cD) = coeffs2
        aux0 = np.concatenate((cA, cH), axis=1);
        aux1 = np.concatenate((cV, cD), axis=1);
        wav_result2 = np.concatenate((aux0, aux1), axis=0);

        wav_result[:, :, 0] = wav_result0
        wav_result[:, :, 1] = wav_result1
        wav_result[:, :, 2] = wav_result2

    else:
        coeffs = pywt.dwt2(image, 'haar')
        cA, (cH, cV, cD) = coeffs
        aux0 = np.concatenate((cA, cH), axis=1);
        aux1 = np.concatenate((cV, cD), axis=1);
        wav_result = np.concatenate((aux0, aux1), axis=0);

    return wav_result


def main():
    img = Image.open("mockup.jpg")
    img.load()
    data = np.asarray(img, dtype="int32")


    result = dwt(data)
    if result.any():
        #plt.imshow(result)
        #plt.show()

        wav_result = library_wavelet(data)
        print(np.array_equal(wav_result, result))
        img =Image.fromarray( np.asarray( np.clip(result,0,255), dtype="uint8"), "RGB" )
        img.save("mockup2.jpg")
        img = Image.fromarray(np.asarray(np.clip(wav_result, 0, 255), dtype="uint8"), "RGB")
        img.save("mockup2_wav.jpg")

main()