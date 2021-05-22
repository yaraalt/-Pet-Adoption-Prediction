import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

# Import the needed sklearn libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Lambda, Flatten, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
import keras
import numpy as np
from tensorflow.keras.layers import concatenate
from sklearn.metrics import classification_report
from PIL import Image
from sklearn.metrics import accuracy_score , recall_score , precision_score , f1_score , confusion_matrix
#Importing required libraries
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from numpy import mean
from keras.optimizers import SGD
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score




def visualize(path):

    data = pd.read_csv(path)
    data = pd.DataFrame(data)
    data = data.dropna()
    print(data.head())
    print(data[["Type","Name","Age","Breed1","Breed2","Gender","Color1","Color2","Color3","MaturitySize","FurLength","Vaccinated","Dewormed","Sterilized","Health","Quantity","Fee","State","RescuerID","VideoAmt","PhotoAmt","PetID"]].describe())
   

    data['AdoptionSpeed'].value_counts().plot.bar()
    plt.title('Class label distribution')
    print("We have class imbalance.")
    

    n_bins = 10

    fig, ((ax0, ax1), (ax2, ax3),(ax4, ax5)) = plt.subplots(nrows=3, ncols=2,figsize=(12,12))


    # continous attributes
    x = data['Age']
    ax0.hist(x,range=[0, 150], color = 'orange', edgecolor = 'black', bins = 10)
    ax0.legend(prop={'size': 10})
    ax0.set_title('Age')

    y = data['Fee']
    ax1.hist(y,range=[0,600], color = 'tan', edgecolor = 'black', bins = 10)
    ax1.set_title('Fee')
     
    z = data['Quantity']   
    ax2.hist(z,range=[0,30], color = 'lime', edgecolor = 'black', bins = 10)
    ax2.set_title('Quantity')

    n = data['VideoAmt']      
    ax3.hist(n,range=[0,30], color = 'pink', edgecolor = 'black', bins = 10)
    ax3.set_title('VideoAmt')
    
    m = data['PhotoAmt']      
    ax4.hist(m,range=[0,40], color = 'blue', edgecolor = 'black', bins = 10)
    ax4.set_title('PhotoAmt')
    
    ax5 = sns.heatmap(data.corr());
    ax5.set_title('Correlation heatmap')


    fig.tight_layout()
    plt.show()

    # dropping unwanted attributes
    data.drop('Name', inplace=True, axis=1)
    data.drop('RescuerID', inplace=True, axis=1)
    data.drop('VideoAmt', inplace=True, axis=1)
    data.drop('PhotoAmt', inplace=True, axis=1)
    data.drop('State', inplace=True, axis=1)


    return data




# splitting data
def split(data):
    training_data = data.dropna()

    x = training_data.iloc[:, 1:18].values
    y = training_data.iloc[:, 18].values
    

    trainX, testX, trainY, testY= train_test_split(x, y, test_size=0.2, random_state=0)


    return trainX, testX, trainY, testY





def preprocess_image(data, path_to_images= 'Images'):

    # split the images into training and testing sets following 80/20 partition
    train_X, test_X, train_y, test_Y = split(data)
    trainPet_ID = train_X[:, -1]
    testPet_ID = test_X[:, -1]


    # load and resize images then saving them into numpy array and return it from the method

    # train
    trainImages= np.zeros(shape=(len(trainPet_ID),32,32,3)) 

    counter = 0
    for i in range(len(trainPet_ID)):
        image = tf.keras.preprocessing.image.load_img(path_to_images+'/'+str(trainPet_ID[i])+"-1.jpg")
        image = image.resize((32,32), resample=0, box=None)
        trainImages[counter] = image

    # test
    testImages= np.zeros(shape=(len(testPet_ID),32,32,3)) 

    counter = 0 
    for i in range(len(testPet_ID)):
        image = tf.keras.preprocessing.image.load_img(path_to_images+'/'+str(testPet_ID[i])+"-1.jpg")
        image = image.resize((32,32), resample=0, box=None)
        testImages[counter] = image


    return trainImages, testImages





# preprocessing categorical and numerical data
def preprocess_data(trainX, testX, trainY, testY):
    # one hot representation of categorical data

    # train
    trainX= pd.DataFrame(trainX, columns=['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize',
                                          'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee', 'PetID'])
    trainY= pd.DataFrame(trainY, columns=['AdoptionSpeed'])


    # test
    testX= pd.DataFrame(testX, columns=['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize',
                                          'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee', 'PetID'])
    testY= pd.DataFrame(testY, columns=['AdoptionSpeed'])


    # finding categorical data that needs to be encoded
    numericData = ['Age','Quantity','Fee']
    ids= []
    temp= list(trainX.columns)
    dropped= ['Name', 'RescuerID', 'PetID','PhotoAmt', 'VideoAmt', 'State']
    for element in temp:
        if (element not in numericData) and (element not in dropped):
            ids.append(element)


    # encoding
    for element in ids:
        # train
        dummies = pd.get_dummies(trainX[element], prefix=element, drop_first=False)
        trainX = trainX.drop([element], axis= 'columns')
        trainX = pd.concat([trainX, dummies], axis=1)

        # test
        dummies = pd.get_dummies(testX[element], prefix=element, drop_first=False)
        testX = testX.drop([element], axis= 'columns')
        testX = pd.concat([testX, dummies], axis=1)


    # scaling numeric data
    scaler = MinMaxScaler()

    for element in numericData:
        # train
        scaled = scaler.fit_transform(trainX[[element]])
        trainX = trainX.drop([element], axis= 'columns')
        trainX[element]= scaled

        # test
        scaled = scaler.fit_transform(testX[[element]])
        testX = testX.drop([element], axis= 'columns')
        testX[element] = scaled

    # making train and test have same number of columns
    for col in trainX.columns:
        if col not in testX.columns:
            testX[col] = 0
        
    for col in testX.columns:
        if col not in trainX.columns:
            trainX[col] = 0


    # one hot representation for label

    # train
    trainY['AdoptionSpeed'] = trainY['AdoptionSpeed'].astype('category')
    oneH_trainY = pd.get_dummies(trainY)

    # test
    testY['AdoptionSpeed'] = testY['AdoptionSpeed'].astype('category')
    oneH_testY = pd.get_dummies(testY)
    

    return trainX, testX, oneH_trainY, oneH_testY





def train_mlp(input_dim, input_train, input_test, target_train, target_test):
    # merge inputs and targets
    inputs = np.concatenate((input_train, input_test), axis=0)
    targets = np.concatenate((target_train, target_test), axis=0)


    # define the K-fold Cross Validator
    kfold = KFold(n_splits=10, shuffle=True)


    max_Acc = 0
    max_model = 0

    acc_per_fold = []
    accuracies = {}

    hist_per_fold = []
    histories = {}
    max_hist = 0

    for x in [64,32,10]:

        # define the model architecture
        NN = Sequential(name='MLP')
        NN.add(Dense(64, input_shape=input_dim,activation='relu'))
        NN.add(Dense(x, activation='relu'))
        NN.add(Dropout(0.2))
        NN.add(Dense(5, activation='softmax'))

        NN.compile(optimizer='adam',loss = tf.losses.categorical_crossentropy , metrics=['accuracy'])

        fold_no = 1
        print('\n*******************************************')
        print('For: ', x)

        for train, test in kfold.split(inputs, targets):
            history = NN.fit(inputs[train], targets[train], epochs=10,batch_size=128, verbose=0)
            hist_per_fold.append(history.history['accuracy'][1] * 100)

            # generate generalization metrics
            scores = NN.evaluate(inputs[test], targets[test], verbose=0)
            print(f'Score for fold {fold_no}: {NN.metrics_names[0]} of {round(scores[0], 4)}     {NN.metrics_names[1]} of {round(scores[1]*100, 4)} %')
            acc_per_fold.append(scores[1] * 100)

            # increase fold number
            fold_no = fold_no + 1

        accuracies[x] = acc_per_fold
        accuracy = mean(acc_per_fold)
        acc_per_fold = []

        histories[x] = hist_per_fold
        hist_per_fold = []

        print('\nAverage accuracy: ', accuracy)


        if (accuracy > max_Acc):
            max_Acc = accuracy
            max_model = NN
            max_hist = history


    # plotting
    plt.plot(max_hist.history['accuracy'], color='blue')
    plt.title('Accuracy for MLP model')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    plt.plot(max_hist.history['loss'], color='red')
    plt.title('Loss for MLP model')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    print('\n*******************************************')
    print(f'Maximum accuracy: {max_Acc}%')
    return max_model






# creating CNN model
def train_cnn(train_images, test_images, trainY, testY, filters=(16, 32, 64)):
    # merge inputs and targets
    inputs = np.concatenate((train_images, test_images), axis=0)
    targets = np.concatenate((trainY, testY), axis=0)

    inputShape= (32, 32, 3)

    # define the K-fold Cross Validator
    kfold = KFold(n_splits=10, shuffle=True)


    max_Acc = 0
    max_model = 0
    max_hist = 0

    acc_per_fold = []
    accuracies = {}

    hist_per_fold = []
    histories = {}


    for x in [0.01, 0.001, 0.0001]:

        # define the model architecture
        model = Sequential()

        model.add(Conv2D(filters[0], (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=inputShape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters[1], (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters[2], (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(5, activation='softmax'))

        # compile model
        opt = SGD(lr= x, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        print('\n*******************************************')
        print('For: ', x)

        fold_no = 1
        for train, test in kfold.split(inputs, targets):
            history = model.fit(inputs[train], targets[train], epochs=10, batch_size=128, verbose=0)
            hist_per_fold.append(history.history['accuracy'][1]*100)

            # generate generalization metrics
            scores = model.evaluate(inputs[test], targets[test], verbose=0)
            acc_per_fold.append(scores[1] * 100)
            print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {round(scores[0], 4)}     {model.metrics_names[1]} of {round(scores[1]*100, 4)} %')

            # increase fold number
            fold_no = fold_no + 1

        accuracies[x] = acc_per_fold
        accuracy= mean(acc_per_fold)
        acc_per_fold= []

        histories[x] = hist_per_fold
        hist_per_fold = []


        print('\nAverage accuracy: ', accuracy)


        if (accuracy > max_Acc):
            max_Acc = accuracy
            max_model = model
            max_hist = history


    # plotting
    plt.plot(max_hist.history['accuracy'], color='blue')
    plt.title('Accuracy for CNN model')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    plt.plot(max_hist.history['loss'], color='red')
    plt.title('loss for CNN model')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    print('\n*******************************************')
    print(f'Maximum accuracy: {max_Acc}%')
    return max_model





def predict_evalute (test_images, test_X, test_Y, model):
    if len(test_images) == 0:
        prediction = model.predict(test_X)
    else:
        prediction = model.predict(test_images)

    # take max value from vector
    prediction = np.argmax(prediction, axis=1)
    testY = test_Y.values.argmax(axis=1)


    # confusion Matrix
    confusion = confusion_matrix(testY, prediction)
    print('Confusion matrix: \n', confusion)


    # accuracy
    accuracy = accuracy_score(testY, prediction)
    print('Accuracy: ', accuracy)


    # recall
    recall = recall_score(testY, prediction, average=None)
    print('\nRecall: ', recall)

    average = sum(recall)/5
    print('Recall average: ', average)

    # precision
    precision = precision_score(testY, prediction, average=None)
    print('\nPrecision: ', precision)

    average = sum(precision) / 5
    print('Precision average: ', average)

    # f-score
    fscore = f1_score(testY, prediction, average=None)
    print('\nF-score: ', fscore)

    average = sum(fscore) / 5
    print('F-score average: ', average)





path_to_data = "dataset.csv"
images_path= "images"
data = pd.read_csv(path_to_data)


# data visualization
print('Visualizing the data.\n')
data = visualize(path_to_data)
print('-------------------------------------------\n')


# split the dataset into training and testing sets following 80/20 partition
print('Splitting data.')
train_X, test_X, train_y, test_Y = split(data)
print('-------------------------------------------\n')



# preprocess categorical and continuous data
print('Preprocessing categorical and continuous data.')
pr_train_X, pr_test_x, pr_train_y, pr_test_y = preprocess_data(train_X, test_X, train_y, test_Y)
pr_train_X.drop('PetID', inplace=True, axis=1)
pr_test_x.drop('PetID', inplace=True, axis=1)
print('\nTrain set (x) after preprocessing:\n', pr_train_X.head())
print('\nTest set (x) after preprocessing:\n', pr_test_x.head())
print('-------------------------------------------\n')


# preprocess images
print('Preprocessing images.')
train_images, test_images = preprocess_image(data, images_path)
print('-------------------------------------------\n')


# create MLP model
print('Building MLP model.')
mlp_model = train_mlp([pr_train_X.shape[1]], pr_train_X, pr_test_x, pr_train_y, pr_test_y)
print('-------------------------------------------\n')


# MLP summary
print('MLP summary:')
mlp_model.summary()


# evaluating MLP model
print('Evaluating MLP model: \n')
empty= []
predict_evalute(empty, pr_test_x, pr_test_y, mlp_model)
print('-------------------------------------------\n')


# create CNN model
print('Building CNN model.')
cnn_model = train_cnn(train_images, test_images, pr_train_y, pr_test_y)
print('-------------------------------------------\n')


# CNN summary
print('CNN summary:')
cnn_model.summary()


# evaluating CNN model
print('Evaluating CNN model: \n')
predict_evalute(test_images, empty, pr_test_y, cnn_model)







