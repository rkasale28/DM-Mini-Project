from statistics import mean

def slr_train(x,y,rows):
    x_mean=mean(x)
    y_mean=mean(y)

    n=0
    d=0
    for i in range(rows):
        n+=((x[i]-x_mean)*(y[i]-y_mean))
        d+=((x[i]-x_mean)**2)
    w1=n/d
    w0=y_mean-(w1*x_mean)

    return w0,w1

def slr_test(x,w0,w1):
    y=w0+(w1*x)
    return y
