
import numpy as np
import random

from utils.gradcheck import gradcheck_naive
from utils.utils import normalizeRows, softmax


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors (rows of matrix) for all words in vocab
                      (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    (dJ / dU)
    """

    y_pred=np.dot(outsideVectors,centerWordVec)
    y_pred=softmax(y_pred.copy())
    print(y_pred)
    
    loss=-np.log(y_pred[outsideWordIdx])


    y_pred=y_pred.reshape(y_pred.shape[0],1)
    gradCenterVec=np.multiply(y_pred,outsideVectors)
    gradCenterVec=np.sum(gradCenterVec,axis=0)
    gradCenterVec=-(outsideVectors[outsideWordIdx]-gradCenterVec)


    ref_y_pred=y_pred.copy()
    ref_y_pred[outsideWordIdx]=-1
    gradOutsideVecs=ref_y_pred.dot(np.array([centerWordVec]))

    


    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 


    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def test(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models
    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.
    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors (rows of matrix) for all words in vocab
                      (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.
    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    (dJ / dU)
    """

    ### YOUR CODE HERE

    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 
    value = np.dot(outsideVectors, centerWordVec) # N x 1
    y_hat  = softmax(value)
    loss = - np.log(y_hat[outsideWordIdx])

    d_value = y_hat
    d_value[outsideWordIdx] -= 1 # y_hat - y, matrix shape (N, 1)
    gradCenterVec   = outsideVectors.T.dot(d_value) # shape d x 1
    gradOutsideVecs = d_value[:, np.newaxis].dot( np.array([centerWordVec]) ) # (N, 1) dot (1, d) -> (N, d)

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs




centervec=np.loadtxt("/home/ravi/Downloads/a2/center.csv",delimiter=",")
outsidevecs=np.loadtxt("/home/ravi/Downloads/a2/outside.csv",delimiter=",")


x,y,z=naiveSoftmaxLossAndGradient(centervec,3,outsidevecs,"x")
print("loss : ",x)
print("vc gradient : ",y)
print("Uc gradient : ",z)


x,y,z=test(centervec,3,outsidevecs,"x")
print("loss : ",x)
print("vc gradient : ",y)
print("Uc gradient : ",z)