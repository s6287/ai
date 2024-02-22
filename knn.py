import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

plt.style.use('ggplot')

df = pd.read_csv('diabetes.csv')

#df.head()
#df.shape
#df.types

x = df.drop('Outcome', axis = 1).values
y = df['Outcome'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 42, shuffle = True)

neighbors = np.arange(1, 9)

train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
                knn = KNeighborsClassifier(n_neighbors = k)
                #Setup a KNN classifier with k neighbours
                knn.fit(x_train, y_train)
                #Fit the model
                train_accuracy[i] = knn.score(x_train, y_train)
                #Compute accuracy in the training set
                test_accuracy[i] = knn.score(x_test, y_test)
                #Compute accuracy in the test set

knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(x_train, y_train)

s = knn.score(x_test, y_test)
print(s)

y_pred = knn.predict(x_test)
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

'''
Precision : Accuracy of Positive Prediction
Recall : Fraction of Correctly Identification Positive Predictions
f1-score : Harmonic Mean of Precisions and Recall
Support : The Number of Occurrence of Each Class in your y test
'''

plt.title('KNN Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
