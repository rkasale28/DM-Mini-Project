from statistics import mean
def nb_train(df, unq_cat_id):
    rows = df.shape[0]

    prob = dict()
    count = dict()

    for category_id in unq_cat_id:
        c = len(df.loc[df['category_id'] == category_id])
        count[category_id] = c
        prob[category_id] = c/float(rows)

    return count, prob

def barlist(probdict):
    return [mean(i) for i in list(probdict.values())]

def nb_test(row_num, probdict, df1, unq_cat_id, prob, columns, df, count):
    row = df1.iloc[row_num]
    plist = {}
    for category_id in unq_cat_id:
        p = prob[category_id]
        for j in columns:
            temp = len(df.loc[(df[j] == row[j]) & (
                df['category_id'] == category_id)])
            p *= (temp/float(count[category_id]))
        plist[category_id] = p
        probdict[category_id].append(p)
    prediction = max(plist, key=plist.get)

    return prediction,probdict

def autolabel(ax,rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')