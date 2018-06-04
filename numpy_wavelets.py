import numpy as np
import pywt
from PIL import Image
import math
from numpy import linalg as LA
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


def dwt(image, matrix):
    result = np.zeros(image.shape, dtype=np.float)
    nchannels = 3
    if len(image.shape) == nchannels:
        for channel in range(nchannels):
           data0 = image[:, :, channel]
           result0 = np.dot(np.dot(matrix, data0), matrix.T)
           result[:, :, channel] = result0
        #result = np.dot(np.dot(haarMatrix, image), haarMatrix2)
    else:

        result = np.array(np.matmul(np.matmul(matrix, image), matrix.T), dtype=np.int32)

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
    print(pywt.wavelist())
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

            coeffs0 = pywt.dwt2(data0, 'db4')
            cA, (cH, cV, cD) = coeffs0
            aux0 = np.concatenate((cA, cV), axis=1)
            aux1 = np.concatenate((cH, cD), axis=1)
            wav_result0 = np.concatenate((aux0, aux1), axis=0)

            wav_result[:, :, channel] = wav_result0

    else:
        coeffs = pywt.dwt2(image, 'db2')
        cA, (cH, cV, cD) = coeffs
        aux0 = np.concatenate((cA, cV), axis=1)
        aux1 = np.concatenate((cH, cD), axis=1)
        wav_result = np.concatenate((aux0, aux1), axis=0)

    return wav_result


def build_daubechie_matrix(row_size):
  size = (int)(row_size / 2)
  h0 = (1 + math.sqrt(3))/(4*math.sqrt(2))
  h1 = (3 +math.sqrt(3))/(4*math.sqrt(2))
  h2 = (3 - math.sqrt(3))/(4*math.sqrt(2))
  h3 = (1 - math.sqrt(3))/(4*math.sqrt(2))

  g0 = h3
  g1 = -h2
  g2 = h1
  g3 = -h0

  zeros = np.zeros((size*2, size*2))
  for i in range(size):
      zeros[i][(2*i)%row_size] = h0
      zeros[i][(2*i + 1)%row_size] = h1
      zeros[i][(2*i + 2)%row_size] = h2
      zeros[i][(2*i + 3)%row_size] = h3

  for i in range(size):
      zeros[size + i][(2 * i)%row_size] = g0
      zeros[size + i][(2 * i + 1)%row_size] = g1
      zeros[size + i][(2 * i + 2)%row_size] = g2
      zeros[size + i][(2 * i + 3)%row_size] = g3

  return zeros


def build_coiflet_matrix(row_size):
    size = (int)(row_size / 2)
    h0 = -0.0156557281
    h1 = -0.0727326195
    h2 = 0.3848648469
    h3 = 0.8525720202
    h4 = 0.3378976625
    h5 = -0.0727326195

    g0 = 0.0727326195
    g1 = 0.3378976625
    g2 = -0.8525720202
    g3 = 0.3848648469
    g4 = 0.0727326195
    g5 = -0.0156557281

    zeros = np.zeros((size * 2, size * 2))
    for i in range(size):
        zeros[i][(2 * i) % row_size] = h0
        zeros[i][(2 * i + 1) % row_size] = h1
        zeros[i][(2 * i + 2) % row_size] = h2
        zeros[i][(2 * i + 3) % row_size] = h3
        zeros[i][(2 * i + 4) % row_size] = h4
        zeros[i][(2 * i + 5) % row_size] = h5

    for i in range(size):
        zeros[size + i][(2 * i) % row_size] = g0
        zeros[size + i][(2 * i + 1) % row_size] = g1
        zeros[size + i][(2 * i + 2) % row_size] = g2
        zeros[size + i][(2 * i + 3) % row_size] = g3
        zeros[size + i][(2 * i + 4) % row_size] = g4
        zeros[size + i][(2 * i + 5) % row_size] = g5

    return zeros


def build_biorthogonal_matrix(row_size):
    size = (int)(row_size / 2)
    h0 = -0.0883883476
    h1 = 0.0883883476
    h2 = 0.7071067812
    h3 = 0.7071067812
    h4 = 0.0883883476
    h5 = -0.0883883476

    g0 = 0
    g1 = 0
    g2 = -0.7071067812
    g3 = 0.7071067812
    g4 = 0
    g5 = 0

    zeros = np.zeros((size * 2, size * 2))
    for i in range(size):
        zeros[i][(2 * i) % row_size] = h0
        zeros[i][(2 * i + 1) % row_size] = h1
        zeros[i][(2 * i + 2) % row_size] = h2
        zeros[i][(2 * i + 3) % row_size] = h3
        zeros[i][(2 * i + 4) % row_size] = h4
        zeros[i][(2 * i + 5) % row_size] = h5

    for i in range(size):
        zeros[size + i][(2 * i) % row_size] = g0
        zeros[size + i][(2 * i + 1) % row_size] = g1
        zeros[size + i][(2 * i + 2) % row_size] = g2
        zeros[size + i][(2 * i + 3) % row_size] = g3
        zeros[size + i][(2 * i + 4) % row_size] = g4
        zeros[size + i][(2 * i + 5) % row_size] = g5

    return zeros


#rotación siguiendo el paper
def rotate(image):
    result = np.zeros(image.shape, dtype=np.float)
    nchannels = 3
    if len(image.shape) == nchannels:
        for channel in range(nchannels):
            mean_x = image[:,:,channel].mean(1)
            mx = np.matmul(mean_x.T, mean_x)
            vx = np.matmul(image[:,:,channel].T, image[:,:,channel]) / 32
            cov_xx = vx - mx
            w, v = LA.eig(cov_xx)
            result[:,:,channel] = np.matmul(v, image[:,:,channel] - mean_x)
    return result

def main():
    img = Image.open("lena.jpg")
    img.load()
    data = np.asarray(img, dtype="int32")
    print(data.shape)

    result = dwt(data, build_haar_matrix(256))
    img = Image.fromarray(np.asarray(np.clip(result, 0, 255), dtype="uint8"))
    img.save("mockup_haar.jpg")
    result = dwt(data, build_daubechie_matrix(256))
    img = Image.fromarray(np.asarray(np.clip(result, 0, 255), dtype="uint8"))
    img.save("mockup_daubechie.jpg")
    result = dwt(data, build_coiflet_matrix(256))
    img = Image.fromarray(np.asarray(np.clip(result, 0, 255), dtype="uint8"))
    img.save("mockup_coiflet.jpg")
    result = dwt(data, build_biorthogonal_matrix(256))
    img = Image.fromarray(np.asarray(np.clip(result, 0, 255), dtype="uint8"))
    img.save("mockup_biorthogonal.jpg")


#test
def testRotate():
    img = Image.open("lena.jpg")
    img.load()
    data = np.asarray(img, dtype="int32")
    result = rotate(data)
    img = Image.fromarray(np.asarray(np.clip(result, 0, 255), dtype="uint8"))
    img.save("lena_rotated.jpg")


testRotate()