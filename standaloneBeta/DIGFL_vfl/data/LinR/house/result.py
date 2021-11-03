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

    actual_shapley_value =  [0.5717333846971726, 0.023416893998732372, 0.06925675120117605, 0.045790829146662625, 0.000689737083512624, 0.001371247779112669, 0.15419389552816112, 0.13354726056546984]
    estimated_shapley_value =  [0.8854474929755708, 0.023621278929593405, 0.030731479728048686, 0.006563067899481218, 0.0, 0.0019330844868736103, 0.05073189241126085, 0.0008809262803573005]

    print(corrcoef(actual_shapley_value,estimated_shapley_value))
