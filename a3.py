import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

def normalize():
    df = pd.read_csv("mini.csv", encoding="utf8")

    df_uk = df.loc[df['Country'] == 'United Kingdom']

    df_uk['InvoiceDate'] = pd.to_datetime(df_uk['InvoiceDate'], errors='coerce')
    df_uk["Amount"] = df_uk["Quantity"] * df_uk["UnitPrice"]
    # print(df_uk["Amount"])

    temp = df_uk.groupby(['CustomerID', 'InvoiceNo'], as_index=False)['Amount'].sum()
    netprice = temp.rename(columns={'Amount': 'Net Spending'})
    return netprice

netprice=normalize()

temp1=netprice.groupby(['CustomerID'] ,as_index=False)['Net Spending'].sum()
netprice1 =temp1.rename(columns={'Net Spending': 'Net Spendings'})

#print(netprice1)

def plot(netprice1):
    price_range = [0, 100, 500, 1000, 5000]

    count_price = []

    for i, price in enumerate(price_range):
        if i == 0: continue
        val = netprice1[(netprice1['Net Spendings'] < price) &
                        (netprice1['Net Spendings'] > price_range[i - 1])]['Net Spendings'].count()
        count_price.append(val)

    plt.rc('font', weight='bold')
    f, ax = plt.subplots(figsize=(11, 6))
    colors = ['yellow', 'wheat', 'royalblue', 'c', 'violet']

    labels = ['{}-{}'.format(price_range[i - 1], s) for i, s in enumerate(price_range) if i != 0]
    sizes = count_price
    explode = [0.0 if sizes[i] < 100 else 0.0 for i in range(len(sizes))]
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct=lambda x: '{:1.0f}%'.format(x) if x > 1 else '',
           shadow=False, startangle=0)
    ax.axis('equal')
    f.text(0.5, 1.01, 'Distribution of orders', ha='center', fontsize=18);
    print(plt.show())

#plot(netprice1)

def customerType(netprice1):
    if netprice1['Net Spendings'] > 1000:
        return 'Best'
    elif netprice1['Net Spendings'] > 500 and netprice1['Net Spendings'] <=1000:
        return 'Average'
    elif netprice1['Net Spendings'] > 100 and netprice1['Net Spendings'] <=500:
        return 'Loyal'
    elif netprice1['Net Spendings'] > 0 and  netprice1['Net Spendings'] <=100:
        return 'New'
    else:
        return 'Get better deals for them'


netprice1['Customer_type'] = netprice1.apply(customerType, axis=1)


netprice1['InvoiceNo']= netprice['InvoiceNo']

netprice1=netprice1[['Net Spendings','Customer_type']]

lis2 =[]
# import re
# for val in netprice1.InvoiceNo.values:
#     if(type(val)== str):
#
#         nv= pd.to_numeric(re.sub('\D', '', val))
#         lis2.append(nv)
#     else:
#         lis2.append(val)
# netprice1['InvoiceNo']=lis2
#
feature_cols = ['Net Spendings']

X = netprice1[feature_cols]  # Features
y = netprice1.Customer_type  # Target variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

def EDA(netprice1):
    import seaborn as sns

    sns.set(style="whitegrid")
    ax = sns.pairplot(data=netprice1, hue='Customer_type')
    plt.title('EDA on dataset');

    # function to show plot
    print(plt.show())

#EDA(netprice1)
def logistic():
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)


    from sklearn import metrics
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred, average='macro'))
    print("Recall:", metrics.recall_score(y_test, y_pred, average='micro'))



#logistic()

def decison():

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    from sklearn.tree import export_graphviz
    from six import StringIO
    from IPython.display import Image
    import pydotplus


    dot_data = StringIO()

    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True,feature_names = ['CustomerID','Net Spendings'],class_names=['Loyal','New','Average','Best','Get better deals for them'])
    from sklearn import tree
    # text_representation = tree.export_text(clf)
    # print(text_representation)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('pic.png')
    Image(graph.create_png())





#decison()

def knn(netprice1):

    k_dict = {}


    for i in range(3, 17, 2):
        knn = KNeighborsClassifier(n_neighbors=i)

        knn.fit(X_train, y_train)

        print("score for k= " + str(i) + ":" + str(knn.score(X_test, y_test)))
        k_dict[i] = knn.score(X_test, y_test)

    names = list(k_dict.keys())
    values = list(k_dict.values())

    plt.plot(names, values, label='Model Accuracy')

    plt.legend()
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.show()

knn(netprice1)

def k_fold():
    overall={}
    for i in range(3, 17, 2):

        k = 10
        kf = KFold(n_splits=k, random_state=None)
        # model = LogisticRegression(solver='liblinear')
        model = KNeighborsClassifier(n_neighbors=i)
        acc_score = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred_values = model.predict(X_test)
            acc = accuracy_score(pred_values, y_test)
            acc_score.append(acc)

        avg_acc_score = sum(acc_score) / k
        maxim = max(acc_score)
        print('accuracy of each fold for k={} - {}'.format(i, acc_score))
        print('Avg accuracy for k={}: {}'.format(i, avg_acc_score))
        overall[i] = maxim
        print()



    names = list(overall.keys())
    values = list(overall.values())

    plt.plot(names, values, label='Model Accuracy')

    plt.legend()
    plt.xlabel('K - Neighbors')
    plt.ylabel('Max Accuracy in each fold')
    plt.show()
#k_fold()


def SVM():
    # scalling the input data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)


    classifier = SVC(kernel='linear')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("accuracy for SVM",accuracy_score(y_test, y_pred))




#SVM()

