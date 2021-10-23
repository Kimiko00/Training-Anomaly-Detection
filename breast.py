import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.svm import SVC

seed = 7
scoring = 'accuracy'

# DATASET URL AND MANIPULATING

#load adult dataset
url = "https://raw.githubusercontent.com/MateLabs/Public-Datasets/master/Datasets/diabetes.csv"
breast_df = pd.read_csv(url)
names = ["times_pregnant", "glucose_concentration", "blood_pressure", "skin_fold", "serum_insulin", "body_mass",
        "diabetes_pedigree", "age", "class"]
breast_df.columns = names
# print(breast_df.head())

#changing class column into 0 and 1
breast_df['class'][breast_df['class'] == 'positive'] = 1
breast_df['class'][breast_df['class'] == 'negative'] = 0
# print(breast_df.head())

print("Size: {}".format(breast_df.shape))
print(breast_df.describe())

breast_df.hist()
# plt.show()

# distribusi normal untuk setiap kolom didalam dataset tanpa colom terakhir
breast_norm = breast_df.iloc[:, 0:8]
for column in breast_norm.columns:
    mu, sigma = breast_norm[column].mean(), breast_norm[column].std()
    s = np.random.normal(mu, sigma, breast_norm[column].count())
    count, bins, ignored = plt.hist(s, 30, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * 
             np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
             linewidth=2, color='r')
    plt.xlabel(breast_norm[column].name)
    plt.ylabel("Probability")
    # plt.show()

# membagi dataset ke dalam x dan y sebagai train dan test data
X = breast_df.iloc[:,0:8]
y = breast_df.iloc[:,8]
y = y.astype('int')

# split ke dalam train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

# scaling data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# membangun list model
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate estimator peforma
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print("{} {} --> {}".format(name, cv_results.mean(), cv_results.std()))

# membuat prediksi
cls = LogisticRegression(random_state=7)
cls.fit(X_train,y_train)
y_pred = cls.predict(X_test)

print("accuracy score : {}".format(accuracy_score(y_test,y_pred)))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))