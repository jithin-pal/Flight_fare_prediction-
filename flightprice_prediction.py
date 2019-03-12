import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import  Dense
from sklearn.metrics import  confusion_matrix
from keras.layers import Dropout

training_set = pd.read_excel("Data_Train.xlsx")
test_set = pd.read_excel("Test_set.xlsx")


def info_dataset():
    # Training set
    print("\nEDA on Training Set\n")
    print("#"*30)
    print("\nFeatures/Columns : \n", training_set.columns)
    print("\n\nNumber of Features/Columns : ", len(training_set.columns))
    print("\nNumber of Rows : ", len(training_set))
    print("\n\nData Types :\n", training_set.dtypes)
    print("\n Contains NaN/Empty cells : ", training_set.isnull().values.any())
    print("\n Total empty cells by column :\n", training_set.isnull().sum(), "\n\n")
    # Test Set
    print("#"*30)
    print("\nEDA on Test Set\n")
    print("#"*30)
    print("\nFeatures/Columns : \n", test_set.columns)
    print("\n\nNumber of Features/Columns : ", len(test_set.columns))
    print("\nNumber of Rows : ", len(test_set))
    print("\n\nData Types :\n", test_set.dtypes)
    print("\n Contains NaN/Empty cells : ", test_set.isnull().values.any())
    print("\n Total empty cells by column :\n", test_set.isnull().sum())

# Data Preprocessing
# Training Set : DATE
training_set = training_set.dropna()
training_set['Journey_Day'] = pd.to_datetime(training_set.Date_of_Journey, format='%d/%m/%Y').dt.day
training_set['Journey_Month'] = pd.to_datetime(training_set.Date_of_Journey, format='%d/%m/%Y').dt.month
training_set.drop(labels= 'Date_of_Journey', axis=1, inplace = True)
training_set.drop(labels = 'Additional_Info', axis=1, inplace=True)
# Test Set
test_set = test_set.dropna()
test_set['Journey_Day'] = pd.to_datetime(test_set.Date_of_Journey, format='%d/%m/%Y').dt.day
test_set['Journey_Month'] = pd.to_datetime(test_set.Date_of_Journey, format='%d/%m/%Y').dt.month
test_set.drop(labels= 'Date_of_Journey', axis=1, inplace = True)
test_set.drop(labels = 'Additional_Info', axis=1, inplace=True)

# Duration
duration = list(training_set['Duration'])
for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + ' 0m'
        elif 'm' in duration[i]:
            duration[i] = '0h {}'.format(duration[i].strip())
dur_hours = []
dur_minutes = []
for i in range(len(duration)):
    dur_hours.append(int(duration[i].split()[0][:-1]))
    dur_minutes.append(int(duration[i].split()[1][:-1]))
training_set['Duration_hours'] = dur_hours
training_set['Duration_minutes'] = dur_minutes
training_set.drop(labels='Duration', axis=1, inplace=True)


durationT = list(test_set['Duration'])
for i in range(len(durationT)):
    if len(durationT[i].split()) != 2:
        if 'h' in durationT[i]:
            durationT[i] = durationT[i].strip() + ' 0m'
        elif 'm' in durationT[i]:
            durationT[i] = '0h {}'.format(durationT[i].strip())

# Test Set
durationT = list(test_set['Duration'])
for i in range(len(durationT)):
    if len(durationT[i].split()) != 2:
        if 'h' in durationT[i]:
            durationT[i] = durationT[i].strip() + ' 0m'
        elif 'm' in durationT[i]:
            durationT[i] = '0h {}'.format(durationT[i].strip())
dur_hours = []
dur_minutes = []
for i in range(len(durationT)):
    dur_hours.append(int(durationT[i].split()[0][:-1]))
    dur_minutes.append(int(durationT[i].split()[1][:-1]))
test_set['Duration_hours'] = dur_hours
test_set['Duration_minutes'] = dur_minutes
# Departure:
# Training Set
training_set['Depart_Time_Hour'] = pd.to_datetime(training_set.Dep_Time).dt.hour
training_set['Depart_Time_Minutes'] = pd.to_datetime(training_set.Dep_Time).dt.minute
training_set.drop(labels = 'Dep_Time', axis = 1, inplace = True)
training_set['Arr_Time_Hour'] = pd.to_datetime(training_set.Arrival_Time).dt.hour
training_set['Arr_Time_Minutes'] = pd.to_datetime(training_set.Arrival_Time).dt.minute
training_set.drop(labels = 'Arrival_Time', axis = 1, inplace = True)
# Test Set
test_set['Depart_Time_Hour'] = pd.to_datetime(test_set.Dep_Time).dt.hour
test_set['Depart_Time_Minutes'] = pd.to_datetime(test_set.Dep_Time).dt.minute
test_set.drop(labels = 'Dep_Time', axis = 1, inplace = True)
test_set['Arr_Time_Hour'] = pd.to_datetime(test_set.Arrival_Time).dt.hour
test_set['Arr_Time_Minutes'] = pd.to_datetime(test_set.Arrival_Time).dt.minute
test_set.drop(labels = 'Arrival_Time', axis = 1, inplace = True)

X_train = training_set.iloc[:, training_set.columns != 'Price'].values
X_test = test_set.iloc[:, :].values
Y_train = training_set.iloc[:, 6].values

le1 = LabelEncoder()
le2 = LabelEncoder()

# Training Set
X_train[:, 0] = le1.fit_transform(X_train[:, 0])
X_train[:, 1] = le1.fit_transform(X_train[:, 1])
X_train[:, 2] = le1.fit_transform(X_train[:, 2])
X_train[:, 3] = le1.fit_transform(X_train[:, 3])
X_train[:, 4] = le1.fit_transform(X_train[:, 4])
X_train[:, 5] = le1.fit_transform(X_train[:, 5])
# Test Set
X_test[:, 0] = le2.fit_transform(X_test[:, 0])
X_test[:, 1] = le2.fit_transform(X_test[:, 1])
X_test[:, 2] = le2.fit_transform(X_test[:, 2])
X_test[:, 3] = le2.fit_transform(X_test[:, 3])
X_test[:, 4] = le2.fit_transform(X_test[:, 4])
X_test[:, 5] = le2.fit_transform(X_test[:, 5])

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
sc_y = StandardScaler()
Y_train = Y_train.reshape((len(Y_train), 1))
Y_train = sc_X.fit_transform(Y_train)
Y_train = Y_train.ravel()

# Model Begins:
classifier = Sequential()
classifier.add(Dense(7, activation='relu', kernel_initializer='uniform', input_dim=14))
classifier.add(Dense(7, activation='relu', kernel_initializer='uniform'))
classifier.add(Dense(7, activation='relu', kernel_initializer='uniform'))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, Y_train, batch_size=10, epochs=20)

Y_pred = sc_X.inverse_transform(classifier.predict(X_test))
pd.DataFrame(Y_pred, columns = ['Price']).to_excel("Final_Pred.xlsx", index = False)

