import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
dataset = pd.read_csv('H:\Harsh\VIT\CSE1021\Assignment\All_India_Index_Upto_Feb24.csv')
dataset.dropna(inplace=True)

all_combo = dataset.drop(columns=['Sector','Sector Division','Year','Month','General index'])
all_combo_list = all_combo.columns.tolist()

# List of different inputs
input_combo=[['General index','Health','Footwear'],
             ['Meat and fish','Cereals and products','Recreation and amusement'],
             all_combo_list,
             ['Transport and communication','Personal care and effects','Pan, tobacco and intoxicants'],
             ['Fuel and light','Non-alcoholic beverages','Spices','Education']]

for i in input_combo:
    X_input = dataset[i]
    Y_output = dataset['Sector Division']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_input, Y_output, test_size=0.5, random_state=1)

    # Scaling the data 
    scaler = StandardScaler() 
    X_train_scaled = scaler.fit_transform(X_train) 
    X_test_scaled = scaler.transform(X_test)

    # Initialize the Logistic Regression model with increased max_iter
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train_scaled, y_train)

    # Make predictions on the test data
    predictions = lr.predict(X_test_scaled)

    # Evaluate the model
    print('---------ATTRIBUTES USED AS INPUT----------\n',i)
    mse = metrics.mean_squared_error(y_test, predictions)
    print("\nMean Squared Error =>>> ", mse)
    print('Model Score ==========>>> ', lr.score(X_train_scaled, y_train))
    print("Accuracy =============>>> ", (metrics.accuracy_score(y_test, predictions))*100,'%')

    # Calculate R-squared to evaluate the model's performance
    r_squared = metrics.r2_score(y_test, predictions)
    print("R-squared ============>>> ", r_squared)
    print()
