import numpy as np
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

    return np.concatenate((haarM1, haarM2), axis=0)

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




def main():
    img = Image.open("mockup.jpg")
    img.load()
    data = np.asarray(img, dtype="int32")

    result = dwt(data)
    if result.any():
        #plt.imshow(result)
        #plt.show()

        img =Image.fromarray( np.asarray( np.clip(result,0,255), dtype="uint8"), "RGB" )
        img.save("mockup2.jpg")


main()