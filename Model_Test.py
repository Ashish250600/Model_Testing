#Import all the required librabries and check the version
import sys
import numpy 
import pandas
import matplotlib
import scipy
import sklearn
print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Scipy: {}'.format(scipy.__version__))
print('Sklearn: {}'.format(sklearn.__version__))

#Import all the required librabries
import pandas
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier

#Read the dataset
url='https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=read_csv(url,names=names)

#Peek at the data
print(dataset.shape)
print(dataset.head())

#Statistical analysis of data
print(dataset.describe())

#Class distribution
print(dataset.groupby('class').size())

#Visualize the data (univariate plot)- box and whisker plots, histogram
dataset.plot(kind='box',subplots='True',layout=(2,2),sharex=False,sharey=False)
pyplot.show()
dataset.hist()
pyplot.show()

#Visualize the data (multivariate plot) - scatter_matrix plot
scatter_matrix(dataset)
pyplot.show()

#Split the dataset
arr=dataset.values
x=arr[:,0:4]
y=arr[:,4]
x_train,x_validation,y_train,y_validation=train_test_split(x,y,test_size=0.2,random_state=1)

#Build Models
models = []
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDS',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))

#Evaluate the created models
results = []
names = []
for name,model in models:
    kfold=StratifiedKFold(n_splits=10)
    cv_results=cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name,cv_results.mean(),cv_results.std()))
    
#Compare our models
pyplot.boxplot(results,labels=names)
pyplot.title('Modles Comparison')

#Test the SVM model
model=SVC(gamma='auto')
model.fit(x_train,y_train)
pred=model.predict(x_validation)
pyplot.show()
print(accuracy_score(y_validation,pred))
print(classification_report(y_validation,pred))
