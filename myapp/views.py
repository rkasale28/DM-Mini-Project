from myapp.nb_functions import nb_test, nb_train, barlist,autolabel
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
import xlwings as xw
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
load=0

# Create your views here.
def home(request):
    return render(request, 'home.html')

def naive_bayes(request):
    xls = pd.ExcelFile('static/excel/INvideosNB.xlsx')
    df = pd.read_excel(xls, 'train_data')
    df1 = pd.read_excel(xls, 'test_data')

    unq_cat_id = np.sort(df1['category_id'].unique())
    columns = ['views', 'likes', 'dislikes', 'comment_count',
            'comments_disabled', 'ratings_disabled']
    pred=[]

    count, prob = nb_train(df, unq_cat_id)
    probdict = {key: [0] for key in unq_cat_id}

    for row_num in range(df1.shape[0]):
        print ("Test on row no.: ",row_num)
        prediction,probdict=nb_test(row_num,probdict,df1,unq_cat_id,prob,columns,df,count)
        pred.append(prediction)

    y=unq_cat_id
    y_pos=np.arange(len(y))
    x=barlist(probdict)

    fig=plt.figure()
    plt.barh(y_pos, x, align='center', alpha=0.5)
    plt.yticks(y_pos, y)
    plt.xlabel('Probability')
    plt.ylabel('Category Id')
    plt.title('Probabilities of category id')
    for i,v in enumerate(x):
        plt.text(v+0.00001,i-0.2,str(round(x[i],4)),color='blue')
    fig.set_size_inches(13, 5)
    plt.tight_layout()
    plt.savefig('static/images/naive_bayes.jpg')

    n_groups=len(unq_cat_id)
    true=list(df1['category_id'])
    
    fig, ax = plt.subplots()
    
    actual_category_id=list(true.count(i) for i in unq_cat_id)
    predicted_category_id=list(pred.count(i) for i in unq_cat_id)
    labels=unq_cat_id

    x = np.arange(len(labels))  # the label locations
    width = 0.5

    rects1 = ax.bar(x - width/2, actual_category_id, width, label='actual_category_id')
    rects2 = ax.bar(x + width/2, predicted_category_id, width, label='predicted_category_id')
    
    ax.set_ylabel('No. of rows')
    ax.set_xlabel('Category Id')
    ax.set_title('Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(ax,rects1)
    autolabel(ax,rects2)
    fig.set_size_inches(13, 5)
    fig.tight_layout()
    fig.savefig('static/images/naive_bayes_comparison.jpg')

    acc=accuracy_score(true,pred)*100
    return render(request,'naive_bayes.html',{'acc':round(acc,2)})

def sample(request):
    return render (request,'naive_bayes.html')