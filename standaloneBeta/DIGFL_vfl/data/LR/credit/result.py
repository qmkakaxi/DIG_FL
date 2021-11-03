from math import sqrt
import math

def multipl(a,b):
    sumofab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sumofab+=temp
    return sumofab

def corrcoef(x,y):
    n=len(x)
    sum1=sum(x)
    sum2=sum(y)
    sumofxy=multipl(x,y)
    sumofx2 = sum([pow(i,2) for i in x])
    sumofy2 = sum([pow(j,2) for j in y])
    num=sumofxy-(float(sum1)*float(sum2)/n)
    den=sqrt((sumofx2-float(sum1**2)/n)*(sumofy2-float(sum2**2)/n))
    return num/den


def distance(p0,p1,digits=2):
    a=map(lambda x: (x[0]-x[1])**2, zip(p0, p1))
    return round(math.sqrt(sum(a)),digits)

if __name__ == "__main__":

    actual_shapley_value =  [0.005260346347778239, 0.5845217168071354, 0.34123682003035466, 0.023430197081996628, 0.03454012773670954, 0.006552498689618937, 0.0044582933064065385]
    estimated_shapley_value =  [0.004101133654824589, 0.46917982507130296, 0.4677789315767813, 0.0024271048721176687, 0.02101332278458965, 0.018921871849666098, 0.016577810188856484]

    print(corrcoef(actual_shapley_value,estimated_shapley_value))
