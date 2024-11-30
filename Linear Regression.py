import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Converting CSV File data into
# pandas dataframe with 'dataset' variable
dataset = pd.read_csv('H:\Harsh\VIT\CSE1021\Assignment\All_India_Index_Upto_Feb24.csv')
dataset.dropna(inplace=True)

# Data Preprocessing
all_combo=dataset.drop(columns=['Sector','Sector Division','Year','Month','General index'])
all_combo_list=all_combo.columns.tolist()

# List of different inputs
input_combo=[['Health','Footwear'],
             ['Meat and fish','Cereals and products','Recreation and amusement'],
             all_combo_list,
             ['Transport and communication','Personal care and effects','Pan, tobacco and intoxicants'],
             ['Fuel and light','Non-alcoholic beverages','Spices','Education']]

# Training and predicting with different inputs
for i in input_combo:
    X_input=dataset[i]
    Y_output=dataset['General index']

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_input, Y_output, test_size=0.5, random_state=1)

    # Initialize the Linear Regression model
    lr = LinearRegression()

    # Train the model
    lr.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = lr.predict(X_test)

    print('---------ATTRIBUTES USED AS INPUT----------\n',i)

    # Evaluate the model
    mse = metrics.mean_squared_error(y_test, predictions)
    print("\nMean Squared Error =>>> ", mse)
    print('Model Score ==========>>> ',lr.score(X_train,y_train))
    r_squared = metrics.r2_score(y_test, predictions) 
    print("R-squared ============>>> ", r_squared*100,'%')
    print()

# Ploting the graph of values
sns.scatterplot(all_combo)
plt.show()