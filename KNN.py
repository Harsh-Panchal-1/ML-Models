import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# Converting CSV File data into
# pandas dataframe with 'data' variable
data = pd.read_csv('H:\Harsh\VIT\CSE1021\Assignment\All_India_Index_Upto_Feb24.csv')

# Data Preprocessing
data.dropna(inplace=True)
all_combo=data.drop(columns=['Sector','Sector Division','Year','Month'])
all_combo_list=all_combo.columns.tolist()

# List of different inputs
input_combo=[['General index','Health','Footwear'],
             ['Meat and fish','Cereals and products','Recreation and amusement'],
             all_combo_list,
             ['Transport and communication','Personal care and effects','Pan, tobacco and intoxicants'],
             ['Fuel and light','Non-alcoholic beverages','Spices','Education']]

# Training and predicting with different inputs
for i in input_combo:
    X_input=data[i]
    Y_output=data['Sector Division']

    X_train, X_test, y_train, y_test = train_test_split(X_input, Y_output, test_size=0.5, random_state=1)
    classifier_knn = KNeighborsClassifier(n_neighbors = 3)
    classifier_knn.fit(X_train, y_train)
    y_pred = classifier_knn.predict(X_test)

    # Finding accuracy by comparing actual response values(y_test) with predicted response val(y_pred)
    print('---------ATTRIBUTES USED AS INPUT----------\n',i)
    mse = metrics.mean_squared_error(y_test, y_pred)
    print("\nMean Squared Error =>>> ", mse)
    print("Accuracy =============>>> ", (metrics.accuracy_score(y_test, y_pred))*100,'%')
    print('Model Score ==========>>> ',classifier_knn.score(X_train,y_train))
    r_squared = metrics.r2_score(y_test, y_pred)
    print("R-squared ============>>> ", r_squared)
    print()
