# Importing libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression


# Reading the file
df = pd.read_csv("raw_data.csv", index_col=0)

# Creating numpy arrays from the data
x = np.array(df.drop('BCA CGPA', axis=1))
y = np.array(df["BCA CGPA"])

# scaling the data for fast processing
X = preprocessing.scale(x)

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

clf = LinearRegression()
clf.fit(x_train, y_train)

# Please uncomment the below block to test the accuracy
# accuracy = clf.score(x_test, y_test)
# print(accuracy)

# New array for value prediction
predict_values = np.array([[77,67,2],
                        [89,79,2],
                        [66,56,2]])

final_predict = clf.predict(predict_values)
print(final_predict)
