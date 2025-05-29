#!/usr/bin/env python
# coding: utf-8

# In[2]:


# this code is a simple machine learning project using the iris dataset ']
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

print(df.head())
print(df.describe())
sns.pairplot(df, hue='species')
plt.show()

X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[202]:


df = pd.read_csv(r"C:\Users\prava\Downloads\Iris.csv")
df.head()


# In[88]:


y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o',
label='setosa')
plt.scatter(X[50:100,0], X[50:100,1], color='blue', marker='x',
label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()


    


# In[125]:


class Perceptron(object):
      def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
          self.eta = eta
          self.n_iter = n_iter
          self.random_state = random_state
      def fit(self, X, y):
          rgen = np.random.RandomState(self.random_state)
          self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
          self.errors_ = []
          for _ in range(self.n_iter):
              errors = 0
              for xi, target in zip(X, y):
                  update = self.eta * (target - self.predict(xi))
                  self.w_[1:] += update * xi
                  self.w_[0] += update
                  errors += int(update != 0.0)
                  self.errors_.append(errors)
                  return self
      def net_input(self, X):
          """Calculate net input"""
          return np.dot(X, self.w_[1:]) + self.w_[0]
      def predict(self, X):
          """return class label after unit setup"""
          return np.where(self.net_input(X) >= 0.0, 1, -1)
      ppn = Perceptron(eta=0.1, n_iter=10)
      ppn.fit(X, y);
      plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_,marker='o')
      plt.xlabel('Epochs')
      plt.ylabel('Number of updates')
      plt.show()


# In[77]:


df = pd.read_csv(r"C:\Users\prava\Downloads\Iris.csv")
df.head()


# In[80]:


df.value_counts("Species")


# In[82]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Species', data=df, )
plt.show()


# In[147]:


class Perceptron(object):
      def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
          self.eta = eta
          self.n_iter = n_iter
          self.random_state = random_state
      def fit(self, X, y):
          rgen = np.random.RandomState(self.random_state)
          self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
          self.errors_ = []
          for _ in range(self.n_iter):
              errors = 0
              for xi, target in zip(X, y):
                  update = self.eta * (target - self.predict(xi))
                  self.w_[1:] += update * xi
                  self.w_[0] += update
                  errors += int(update != 0.0)
                  self.errors_.append(errors)
                  return self
      def net_input(self, X):
          """Calculate net input"""
          return np.dot(X, self.w_[1:]) + self.w_[0]
      def predict(self, X):
          """return class label after unit setup"""
          return np.where(self.net_input(X) <= 0.0, 1, -1)
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y);
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()
      
             


# In[149]:


from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
# setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
# plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max,resolution),
    np.arange(x2_min, x2_max,resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
# plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],y=X[y == cl, 1],alpha=0.8,c=colors[idx],marker=markers[idx],label=cl,edgecolor='black')
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show();
        
            


# In[119]:


df.shape
df.info()
df.describe()


# In[151]:


df.isnull().sum()


# In[153]:


data = df.drop_duplicates(subset ="Species",)
data


# In[155]:


df.value_counts("Species")


# In[157]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Species', data=df, )
plt.show()


# In[159]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm',
hue='Species', data=df, )
# Placing Legend outside the Figure
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()


# In[161]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm',
hue='Species', data=df, )
# Placing Legend outside the Figure
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()


# In[163]:


import seaborn as sns
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(10,10))
axes[0,0].set_title("Sepal Length")
axes[0,0].hist(df['SepalLengthCm'], bins=7)
axes[0,1].set_title("Sepal Width")
axes[0,1].hist(df['SepalWidthCm'], bins=5);
axes[1,0].set_title("Petal Length")
axes[1,0].hist(df['PetalLengthCm'], bins=6);
axes[1,1].set_title("Petal Width")
axes[1,1].hist(df['PetalWidthCm'], bins=6);


# In[165]:


import seaborn as sns
import matplotlib.pyplot as plt
plot = sns.FacetGrid(df, hue="Species")
plot.map(sns.distplot, "SepalLengthCm").add_legend()
plot = sns.FacetGrid(df, hue="Species")
plot.map(sns.distplot, "SepalWidthCm").add_legend()
plot = sns.FacetGrid(df, hue="Species")
plot.map(sns.distplot, "PetalLengthCm").add_legend()
plot = sns.FacetGrid(df, hue="Species")
plot.map(sns.distplot, "PetalWidthCm").add_legend()
plt.show()

# In[196]:


import seaborn as sns
import matplotlib.pyplot as plt
def graph(y):
    sns.boxplot(x="Species", y=y, data=df)
plt.figure(figsize=(10,10))
# Adding the subplot at the specified
# grid position
plt.subplot(221)
graph('SepalLengthCm')
plt.subplot(222)
graph('SepalWidthCm')
plt.subplot(223)
graph('PetalLengthCm')
plt.subplot(224)
graph('PetalWidthCm')
plt.show()


# In[204]:


import seaborn as sns
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv(r"C:\Users\prava\Downloads\Iris.csv")
sns.boxplot(x='SepalWidthCm', data=df)


# In[230]:


import sklearn
import pandas as pd
import seaborn as sns
df = pd.read_csv(r"C:\Users\prava\Downloads\Iris.csv")
# IQR
Q1 = np.percentile(df['SepalWidthCm'], 25,interpolation = 'midpoint')
Q3 = np.percentile(df['SepalWidthCm'], 75,interpolation = 'midpoint')
IQR = Q3 - Q1
print("Old Shape: ", df.shape)
# Upper bound
upper = np.where(df['SepalWidthCm'] >= (Q3+1.5*IQR))
# Lower bound
lower = np.where(df['SepalWidthCm'] <= (Q1-1.5*IQR))
# Removing the Outliers
df.drop(upper[0], inplace = True)
df.drop(lower[0], inplace = True)
print("New Shape: ", df.shape)
sns.boxplot(x='SepalWidthCm', data=df)
OldShape: (150, 5)
NewShape: (146, 5)







