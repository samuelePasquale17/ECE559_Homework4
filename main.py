from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np


def read_MNIST():
    """
    Function that reads MNIST dataset
    :return: return Xraw and Yraw matrices
    """
    # read MNIST data set
    mnist = fetch_openml('mnist_784', as_frame=False, parser='liac-arff')
    Xraw, Yraw = mnist['data'], mnist['target']
    Yraw = Yraw.astype(int)
    return Xraw, Yraw


def print_image(row, number):
    """
    Function that shows the image of a given value
    :param row: row of Xraw, with 28x28 pixels
    :param number: Actual number shown in the image
    :return:
    """
    image = row.reshape(28, 28)  # MNIST images are 28x28 pixels
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {number}")
    plt.show()


def print_example_each_digit(Xraw, Yraw):
    """
    Function that shows one example of all the digits from 0 to 9
    :param Xraw: Xraw matrix
    :param Yraw: Yraw matrix
    :return:
    """
    # found flag
    found = 0

    # search up to digit 9
    while found < 10:
        # scan Yraw
        for i in range(len(Yraw)):
            # check if it is the target digit
            if Yraw[i] == found:
                # print the image
                print_image(Xraw[i], found)
                # update flag
                found += 1
                # exiting the for loop
                break


def generate_M(rows, cols, divisor, uniform_lower_bound=0, uniform_upper_bound=1):
    """
    Function that generates a rowsXcols matrix filled of random uniform values between the two
    given boundaries.
    :param rows: number of rows
    :param cols: numeber of columns
    :param divisor: divisor for M/M
    :param uniform_lower_bound: lower bound of uniform function
    :param uniform_upper_bound: upper bound of uniform function
    :return: return the generated matrix M
    """
    # generation of random matrix M
    M = np.random.uniform(uniform_lower_bound, uniform_upper_bound, (rows, cols))
    # return matrix M
    return M / divisor


def generate_X(M, Xraw):
    """
    Function that generates X matrix as M * Xraw^T
    :param M: M matrix
    :param Xraw: Xraw matrix
    :return: matrix with dimension d × 70,000
    """
    # return  M * Xraw^T
    return M @ Xraw.T


def generate_Y(rows, Yraw):
    """
    Function that generates Y matrix, where each i-column represents
    the one-hot encoding of each i-row of Yraw
    :param rows: number of rows
    :param Yraw: Yraw matrix
    :return: Y matrix
    """
    # Y matrix zeroed out
    Y = np.zeros((rows, Yraw.shape[0]))
    # one-hot encoding
    Y[Yraw, np.arange(Yraw.shape[0])] = 1
    # return matrix Y
    return Y


def MoorePensorePseudoInverse(X, Y):
    """
    Function that computes the moore pensore pseudo-inverse given X and Y matrices
    :param X: X matrix
    :param Y: Y matrix
    :return: W = Y X^T (X X^T)^(-1)
    """
    # pseudo inverse with linalg  library
    X_pseudo_inv = np.linalg.pinv(X)
    # weights computation
    W = Y @ X_pseudo_inv
    # return weights
    return W


def predictor(X, W):
    """
    Function that given X matrix and W computes the prediction
    :param X: X matrix of values
    :param W: weight matrix
    :return: prediction
    """
    # return the prediction
    return W @ X


def MSE_predictor(Y, Y_pred):
    """
    Function that given the expected values and the actual ones, return the MSE
    :param Y: Y matrix of expected values
    :param Y_pred: Y_pred matrix of actual values generated by the predictor
    :return: MSE of the predictor
    """
    # return the MSE
    return np.mean(np.linalg.norm(Y - Y_pred, axis=0) ** 2)


def ERROR_predictor(Y, Y_pred):
    """
    Function that given the expected values and the actual ones, return the number of errors
    :param Y: Expected values
    :param Y_pred: Actual values generated by the predictor
    :return: number of errors
    """
    # get the labels predicted
    labels_pred = np.argmax(Y_pred, axis=0)
    # get the correct labels
    actual_labels = np.argmax(Y, axis=0)
    # count the number of times in which there is a missmatch between labels
    error_count = np.sum(labels_pred != actual_labels)
    # return the number of errors
    return error_count


def MSE_plot(MSE_results_LMS, epochs):
    """
    Function that plots the MSE of the Widrow-Hoff LMS algorithm
    :param MSE_results_LMS: MSE of the Widrow-Hoff LMS algorithm
    :param epochs: number of epochs
    :return:
    """
    # plot the MSE vs. Number of epochs
    plt.plot(range(1, epochs + 1), MSE_results_LMS, marker='o')
    plt.xlabel('Number of Epochs')
    plt.ylabel('MSE')
    plt.title('MSE vs. Number of Epochs')
    plt.show()


def WidrowHoffLMS(Xraw, Y, d, eta, epochs):
    """
    Function that implements the Widrow-Hoff LMS algorithm
    :param Xraw: Xraw matrix
    :param Y: labels
    :param d: d parameter
    :param eta: eta parameter
    :param epochs: number of epochs
    :return: return the weight matrix, the data X, and the MSE per each epoch
    """
    # init weight vector at the origin (all 0's)
    W = np.zeros((10, d))

    # init M and X
    M = generate_M(d, 784, 255*d)
    X = generate_X(M, Xraw)

    # vector containing MSE
    MSE_results_LMS = []

    # run the algorithm epochs times
    for epoch in range(epochs):
        # online algorithm (execution over one point at a time)
        for i in range(X.shape[1]):
            # single sample (dx1)
            x_i = X[:, i].reshape(-1, 1)
            # true lable
            y_i = Y[:, i].reshape(-1, 1)
            # update weights
            W += eta * ((y_i - W @ x_i) @ x_i.T)

        # compute MSE at the end of each epoch
        Y_pred = W @ X
        MSE = MSE_predictor(Y, Y_pred)
        MSE_results_LMS.append(MSE)

    # return result (weights, data, and MSE per epoch)
    return W, X, MSE_results_LMS


def main():
    #####################################################################################
    ###################################### POINT a ######################################
    #####################################################################################
    # load MNIST data set
    Xraw, Yraw = read_MNIST()

    # print one example per each digit
    print_example_each_digit(Xraw, Yraw)

    #####################################################################################
    ###################################### POINT b ######################################
    #####################################################################################
    # set d to 10
    d = 10

    # generation of M, X, and Y
    M = generate_M(d, 784, 255*d)
    X = generate_X(M, Xraw)
    Y = generate_Y(10, Yraw)

    # Moore-Penrose pseudo inverse computation
    W = MoorePensorePseudoInverse(X, Y)

    #####################################################################################
    ###################################### POINT c ######################################
    #####################################################################################
    # setting different d's
    ds = [10, 50, 100, 200, 500]
    # MSE results
    MSE_results = []
    # Errors results
    ERROR_results = []

    # run for each d
    for d in ds:
        # generation of M, X, and Y
        M = generate_M(d, 784, 255 * d)
        X = generate_X(M, Xraw)
        Y = generate_Y(10, Yraw)

        # Moore-Penrose pseudo inverse computation
        W = MoorePensorePseudoInverse(X, Y)

        # predictor
        Y_pred = predictor(X, W)

        # Mean Squared Error computation
        MSE = MSE_predictor(Y, Y_pred)
        MSE_results.append(MSE)

        # ERRORs predictor
        ERROR = ERROR_predictor(Y, Y_pred)
        ERROR_results.append(ERROR)

        # Print obtained result
        print("### Point c ###")
        print(f"d = {d}: MSE = {MSE}, Number of errors = {ERROR}")

    #####################################################################################
    ###################################### POINT d ######################################
    #####################################################################################
    # setting d
    d = 100
    # setting eta
    eta = 0.001
    # setting number of epochs
    epochs = 10

    # run Widrow-Hoff LMS algorithm
    W, X, MSE_results_LMS = WidrowHoffLMS(Xraw, Y, d, eta, epochs)

    # plot MSE of Widrow-Hoff LMS algorithm vs. epochs
    MSE_plot(MSE_results_LMS, epochs)

    # predictor
    Y_pred = predictor(X, W)

    # ERRORs predictor computed and printed
    ERROR = ERROR_predictor(Y, Y_pred)
    print("### Point d ###")
    print(f"Number of errors: {ERROR}")


main()