# This is a sample Python script.
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import h5py as h5
import numpy as np

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def show_model():
    # read H5 file
    pc = h5.File('UKBRVLV_All.h5', 'r')

    # note that H5PY matrices in python are transposed

    # generate the first principal mode
    # with 1.5 times the standard deviation
    S = np.transpose(pc['MU']) + (1.5 * np.sqrt(pc['LATENT'][0, 0]) * pc['COEFF'][0, :])

    # get ED & ES points, & convert to 3 columns matrix [x, y, z]
    N = S.shape[1] // 2
    ed = np.reshape(S[0, :N], (-1, 3))
    es = np.reshape(S[0, N:], (-1, 3))

    # plot ED points in blue
    # plot ES points in red
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ed[:, 0], ed[:, 1], ed[:, 2], color='dodgerblue', marker='.')
    # ax.scatter(es[:, 0], es[:, 1], es[:, 2], color='firebrick', marker='.')
    plt.pause(199)
    plt.close()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    show_model()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
