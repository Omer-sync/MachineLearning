import numpy
import csv
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
label=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
Feature_number=4
Training_number=50
Iris_setosa= numpy.empty((50,4))
Iris_versicolor=numpy.empty((50,4))
Iris_virginica=numpy.empty((50,4))
Iris=numpy.empty((150,4))
def get_covariance_matrix(A):
    A=numpy.array(A, dtype='f')
    mean_vector=numpy.mean(A,axis=0)
    cov_matrix = numpy.reshape(numpy.zeros(Feature_number*Feature_number), (Feature_number,Feature_number))
    print(A[:,1][1])
    print(mean_vector[1])
    for x in range(Feature_number):
        
        for y in range(len(A[:,x])):
            A[:,x][y]=float(A[:,x][y])-float(mean_vector[x])

    cov_matrix=(numpy.dot(numpy.transpose(A),A))/Training_number
    print(cov_matrix)

def parser(data):
    x = list()
    for i in range(len(data)):
        temp = data[i].split(',')
        x.append([temp[0],temp[1],temp[2],temp[3]])
    return numpy.array(x).astype(numpy.float64)
def data_processing():
    with open("irisdata.xml") as f:
        irisdata = f.read()
    data = BeautifulSoup(irisdata, 'xml').text.split('\n')[1:-3]
    numpy.copyto(Iris,parser(data))
    numpy.copyto(Iris_setosa,Iris[:50])
    numpy.copyto(Iris_versicolor,Iris[50:100])
    numpy.copyto(Iris_virginica,Iris[100:150])


def draw():
    for m in range(Feature_number):
        for n in range(Feature_number):
            if m < n:
                fn=open("irisdata.txt","r")
                for row in csv.DictReader(fn, label):
                    #plt.xlim(0,10)
                    #plt.ylim(0,10)
                    plt.xlabel(label[m])
                    plt.ylabel(label[n])
                    plt.title(label[m]+"  and  "+label[n])
                    x = row[label[m]]
                    y = row[label[n]]
                    if row["class"]=="Iris-setosa":
                        plt.plot(x,y,"ro")
                    elif row["class"]=="Iris-versicolor":
                        plt.plot(x,y,"bo")
                    else:
                        plt.plot(x,y,"go")
                plt.savefig(label[m]+"_and_"+label[n]+".png",dpi=300,format="png")
                plt.close()
                fn.close()



data_processing()
print("Iris_setosa\n")
get_covariance_matrix(Iris_setosa)
print("Iris_versicolor\n")
get_covariance_matrix(Iris_versicolor)
print("Iris_virginica\n")
print(get_covariance_matrix(Iris_virginica))

print("Iris_setosa mean vector\n")
print(numpy.mean(numpy.reshape(Iris_setosa,(Training_number,Feature_number)).astype(numpy.float64),axis = 0))
print("Iris_versicolor mean vector\n")
print(numpy.mean(numpy.reshape(Iris_versicolor,(Training_number,Feature_number)).astype(numpy.float64),axis = 0))
print("Iris_virginica mean vector\n")
print(numpy.mean(numpy.reshape(Iris_virginica,(Training_number,Feature_number)).astype(numpy.float64),axis = 0))
#draw()