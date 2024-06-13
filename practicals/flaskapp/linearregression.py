import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle

# Load the data
df = pd.read_csv('homeprices.csv')

# Assume 'area' as independent variable and 'price' as dependent variable
X = df['area'].values.reshape(-1,1)
y = df['price'].values.reshape(-1,1)

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a Linear Regression object
regressor = LinearRegression()  

# Train the model using the training sets
regressor.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regressor.predict(X_test)

# Print the coefficients
print('Coefficients: \n', regressor.coef_)

# Save the model to disk
filename = 'finalized_model.pkl'
pickle.dump(regressor, open(filename, 'wb'))