import pandas as pd
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

random_seed = 12
plt.rcParams["figure.figsize"] = (20, 10)
pd.set_option('display.max_columns', None)


data = pd.read_csv('indian_liver_patient.csv')
data_to_use = data
del data_to_use['Gender']
data_to_use.dropna(inplace=True)
# print(data_to_use.head())

# values = data_to_use.values
# Y = values[:,9] # use the columns as data predictions
# X = values[:,0:9]

X = data_to_use.iloc[:,0:8]
Y = data_to_use.iloc[:,8]
Y = Y.astype('int')

outcome = []
model_names = []
models  = [
    ('GaussianNB', GaussianNB()),
    ('Xboost', XGBClassifier(random_state=24)),
    ('Forest', RandomForestClassifier(random_state=24)),
    ('LogReg', LogisticRegression()),
    ('SVM', SVC()),
    ('DecTree', DecisionTreeClassifier()),
    ('KNN', KNeighborsClassifier()),
    ('LinDisc', LinearDiscriminantAnalysis())]

for model_name, model in models:
    k_fold_validation = model_selection.KFold(n_splits=10, random_state=random_seed, shuffle=True)
    results = model_selection.cross_val_score(model, X, Y, cv=k_fold_validation, scoring='accuracy')
    outcome.append(results)
    model_names.append(model_name)
    output_message = "%s| Mean=%f STD=%f" % (model_name, results.mean(), results.std())
    print(output_message)

print(model_names)
print(models)

fig = plt.figure()
fig.suptitle('Machine Learning Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome)
ax.set_xticklabels(model_names)
plt.show()

# train test split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# standart scaller 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

GaussianNB = GaussianNB()
Xboost = XGBClassifier(random_state=24)
Forest = RandomForestClassifier(random_state=24)
LogReg = LogisticRegression()
SVM = SVC()
DecTree = DecisionTreeClassifier()
KNN = KNeighborsClassifier()
LinDisc = LinearDiscriminantAnalysis()

model_classifier = [GaussianNB, Xboost, Forest, LogReg, SVM, DecTree, KNN, LinDisc]

for model in model_classifier:
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    clf_report = classification_report(Y_test, y_pred)
    print(f"The Accuracy of Models {type(model).__name__} is {accuracy:}")
    print(clf_report)
    print("\n")