import numpy
import csv
from bs4 import BeautifulSoup
import scipy.linalg as la
label=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
Feature_number=4
Training_number=50
Iris_setosa= numpy.empty((50,4))
Iris_versicolor=numpy.empty((50,4))
Iris_virginica=numpy.empty((50,4))
Iris=numpy.empty((150,4))
def parser(data):
    x = list()
    for i in range(len(data)):
        temp = data[i].split(',')
        x.append([temp[0],temp[1],temp[2],temp[3]])
    return numpy.array(x).astype(numpy.float64)
def data_processing():
    with open('irisdata.xml') as f:
        irisdata = f.read()
    data = BeautifulSoup(irisdata, 'xml').text.split('\n')[1:-3]
    numpy.copyto(Iris,parser(data))
    numpy.copyto(Iris_setosa,Iris[:50])
    numpy.copyto(Iris_versicolor,Iris[50:100])
    numpy.copyto(Iris_virginica,Iris[100:150])

def get_covariance_matrix(A):
    A=numpy.array(A, dtype='f')
    mean_vector=numpy.mean(A,axis=0)
    cov_matrix = numpy.reshape(numpy.zeros(Feature_number*Feature_number), (Feature_number,Feature_number))

    for x in range(Feature_number):
        for y in range(len(A[:,x])):
            A[:,x][y]=float(A[:,x][y])-float(mean_vector[x])

    cov_matrix=(numpy.dot(numpy.transpose(A),A))/Training_number
    return cov_matrix

def eigvals_vectors(A):

    _matrix = get_covariance_matrix(A)
    eigvals, eigvectors = la.eig(_matrix)

    return eigvals, eigvectors

data_processing()
eigvals, eigvectors = eigvals_vectors(Iris_setosa)

print("Eigvals:\n",eigvals.real)
print("Eigvectors:\n",eigvectors)


