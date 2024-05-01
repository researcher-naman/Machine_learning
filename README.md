# ML-Sklearn Quick Guide

Discover the essentials of Scikit-Learn with our quick guide. From model import to dataset exploration, get started swiftly.


## Esentials

```python
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
```
how to inmport dataset

```python
data = pd.read_csv('dataset.csv')
x = data.drop('y',axis=1)
y = data['y']
```
how to use train test split data

```python
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.35,random_state=32)
```

How to use StandardScaler (preprocessing)
```python
scaler = StandardScaler()

x_train_std = scaler.fit_transform(x_train)
x_test_std = scaler.transform(x_test)
```

How to use accuracy_score

```python
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
```

How to Make Confussion Matrix

```python
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
```

How to get Classification Evaluation Summary

```python
from sklearn.metrics import classification_report

# Generate classification report
class_report = classification_report(y_test, y_pred)
print(class_report)
```
## How to import models and use them

#### Gaussian naive Bayesian classifier

```python
  from sklearn.naive_bayes import GaussianNB
  gnb = GaussianNB()
```

#### Perceptron classifier

```python
  from sklearn.linear_model import Perceptron
  perceptron_classifier = Perceptron()
```

#### SVM classifier

```python
  from sklearn.svm import SVC
  svm_classifier = SVC(kernel='rbf',C=1.0, gamma='scale')
```

####  k-Nearest Neighbour

```python
  from sklearn.neighbors import KNeighborsClassifier
  k = 4
  knn = KNeighborsClassifier(n_neighbors=k)
```

####  K-Means clustering

```python
  from sklearn.cluster import KMeans
  kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
  
  # give full data frame at place of 'your_df' while model fiting
  kmeans.fit(your_df)

  # for results use

  df['Cluster'] = kmeans.labels_
```

####  PCA on any data

```python
  from sklearn.decomposition import PCA
  
  # Apply PCA for dimensionality reduction
  pca = PCA(n_components=0.95)
  X_train_reduced = pca.fit_transform(X_train_scaled)
  X_test_reduced = pca.transform(X_test_scaled)
```

####  Decision Tree Classifier

```python
  from sklearn.tree import DecisionTreeClassifier
  
  clf = DecisionTreeClassifier()
```


## How to Import and use famous Datasets

#### MNIST dataset

```python
  from sklearn.datasets import fetch_openml

  mnist_data = fetch_openml('mnist_784', version=1, parser='auto')
  X_data = mnist_data.data.astype('float32')
  y_data = mnist_data.target.astype('int64')
    
  X_data /= 255.0
```

#### Iris dataset

```python
  from sklearn.datasets import load_iris

  iris = load_iris()
  x = iris.data
  y = iris.target
```
## Random DataSet Maker

```python
import pandas as pd
import numpy as np

data = pd.DataFrame(
    {
        'area':np.random.randint(1000,5000,size=(35,)),
        'price':np.random.randint(25000,50000,size=(35,)),
        'review':np.random.randint(1,10,size=(35,))
    }
)

data.to_csv('house_price.csv',index=False)
print()
```
