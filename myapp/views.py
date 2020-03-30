from myapp.nb_functions import nb_test, nb_train, barlist, autolabel
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from myapp.knn_functions import predict_classification
from myapp.slr_functions import slr_train,slr_test
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

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
    pred = []

    count, prob = nb_train(df, unq_cat_id)
    probdict = {key: [0] for key in unq_cat_id}

    for row_num in range(df1.shape[0]):
        print("Test on row no.: ", row_num)
        prediction, probdict = nb_test(
            row_num, probdict, df1, unq_cat_id, prob, columns, df, count)
        pred.append(prediction)

    y = unq_cat_id
    y_pos = np.arange(len(y))
    x = barlist(probdict)

    fig = plt.figure()
    plt.barh(y_pos, x, align='center', alpha=0.5)
    plt.yticks(y_pos, y)
    plt.xlabel('Probability')
    plt.ylabel('Category Id')
    plt.title('Probabilities of category id')
    for i, v in enumerate(x):
        plt.text(v+0.00001, i-0.2, str(round(x[i], 4)), color='blue')
    fig.set_size_inches(13, 5)
    plt.tight_layout()
    plt.savefig('static/images/nb/naive_bayes.jpg')

    n_groups = len(unq_cat_id)
    true = list(df1['category_id'])

    fig, ax = plt.subplots()

    actual_category_id = list(true.count(i) for i in unq_cat_id)
    predicted_category_id = list(pred.count(i) for i in unq_cat_id)
    labels = unq_cat_id

    x = np.arange(len(labels))  # the label locations
    width = 0.5

    rects1 = ax.bar(x - width/2, actual_category_id,
                    width, label='actual_category_id')
    rects2 = ax.bar(x + width/2, predicted_category_id,
                    width, label='predicted_category_id')

    ax.set_ylabel('No. of rows')
    ax.set_xlabel('Category Id')
    ax.set_title('Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(ax, rects1)
    autolabel(ax, rects2)
    fig.set_size_inches(13, 5)
    fig.tight_layout()
    fig.savefig('static/images/nb/naive_bayes_comparison.jpg')

    acc = accuracy_score(true, pred)*100
    return render(request, 'naive_bayes.html', {'acc': round(acc, 2)})


def knn(request):
    xls = pd.ExcelFile('static/excel/INvideosKNN.xlsx')
    df = pd.read_excel(xls, 'train_data')
    df1 = pd.read_excel(xls, 'test_data')

    train_dataset = df[['views', 'likes', 'dislikes', 'comment_count',
                        'comments_disabled', 'ratings_disabled', 'category_id']].to_numpy()
    test_dataset = df1[['views', 'likes', 'dislikes', 'comment_count',
                        'comments_disabled', 'ratings_disabled', 'category_id']].to_numpy()

    pred = []
    j = 0
    for i in test_dataset:
        print("Test on row no.: ", j)
        j += 1
        prediction = predict_classification(train_dataset, i, 50)
        pred.append(prediction)

    true = df1['category_id'].values.tolist()
    acc = accuracy_score(true, pred)*100

    n_neighbors = 50
    train_dataset = df1[['views', 'likes']].to_numpy()
    target = df1['category_id'].to_numpy()

    X = train_dataset
    y = target
    h = 10000  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Classification")

    plt.savefig('static/images/knn/knn_visualization.jpg')

    fig, ax = plt.subplots()

    unq_cat_id = np.sort(df1['category_id'].unique())
    actual_category_id = list(true.count(i) for i in unq_cat_id)
    predicted_category_id = list(pred.count(i) for i in unq_cat_id)
    labels = unq_cat_id

    x = np.arange(len(labels))  # the label locations
    width = 0.5

    rects1 = ax.bar(x - width/2, actual_category_id,
                    width, label='actual_category_id')
    rects2 = ax.bar(x + width/2, predicted_category_id,
                    width, label='predicted_category_id')

    ax.set_ylabel('No. of rows')
    ax.set_xlabel('Category Id')
    ax.set_title('Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(ax, rects1)
    autolabel(ax, rects2)
    fig.set_size_inches(13, 5)
    fig.tight_layout()
    fig.savefig('static/images/knn/knn_comparison.jpg')

    return render(request, 'knn.html', {'acc': round(acc, 2)})


def slr(request):
    xls = pd.ExcelFile('static/excel/INvideosSLR.xlsx')
    df = pd.read_excel(xls, 'train_data')
    df1 = pd.read_excel(xls, 'test_data')

    x = (df['views'].values.tolist())
    y = (df['likes'].values.tolist())

    rows=df.shape[0]
    w0,w1=slr_train(x,y,rows)
    pred=[]
    x=df1['views'].values.tolist()
    true=df1['likes'].values.tolist()

    for i in range (df1.shape[0]):
        print ("Test on row no.:",i)
        x1=(df1.iloc[i]['views'])
        pred.append(slr_test(x1,w0,w1))
    
    figure, axes = plt.subplots()
    plt.scatter(x,true, c='b', label='actual_likes')
    plt.legend(loc='upper left')
    plt.xlabel('Views')
    plt.ylabel('Likes')
    plt.title('Actual Likes')
    plt.savefig('static/images/slr/slr_actual.jpg')

    figure, axes = plt.subplots()
    plt.scatter(x,pred, c='r', label='predicted_likes')
    plt.legend(loc='upper left')
    plt.xlabel('Views')
    plt.ylabel('Likes')
    plt.title('Predicted Likes')
    plt.savefig('static/images/slr/slr_predicted.jpg')
    
    figure, axes = plt.subplots()
    plt.scatter(x,true, c='b', label='actual_likes')
    plt.scatter(x,pred, c='r', label='predicted_likes')
    plt.legend(loc='upper left')
    plt.xlabel('Views')
    plt.ylabel('Likes')
    plt.title('Comparison')
    plt.savefig('static/images/slr/slr_comparison.jpg')
    return render(request, 'slr.html')

def sample(request):
    return render(request,'slr.html')