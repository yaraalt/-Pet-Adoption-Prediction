# Pet-Adoption-Prediction
Pet adoption prediction is a multi-class classification problem to predict the speed at which a pet is adopted, based on the pet’s listing on PetFinder. The model should predict one of the following classes:
- 0: Pet was adopted on the same day as it was listed.
- 1: Pet was adopted between 1 and 7 days (1st week) after being listed.
- 2: Pet was adopted between 8 and 30 days (1st month) after being listed.
- 3: Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed.
- 4: No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days).
The dataset was adopted from Kaggle competition "PetFinder.my Adoption Prediction". The dataset includes categorical, numerical, and image data.

## Data Preprocessing 
The [dataset](https://www.kaggle.com/c/petfinder-adoption-prediction) is split into training and testing sets folling 80/20 partition. Categorical data is preprocessed using one-hot representation. Whereas, continuous data by scaling their values to be in the range [0,1]. Finally, the label is preprocessed as one-hot representation.

## Multi-Layer Perceptron (MLP)
Trains the model on categorical and numerical data found in "dataset.csv". Moreover, the model contains three layers (including input and output layers). 10-fold cross validation is performed to select the number of units [10, 32 or 64] in the middle hidden layer, such that the number of units that produces the highest accuracy is chosen. Finally, the accuracy of the model is plotted against each epoch.

## Convolutional Neural Network (CNN)
Trains the model on image data. 10-fold cross validation is performed to select the best learning rate from [1e-2, 1e-3, 1e-4], such that the learning rate that produces the highest accuracy is chosen. Finally, the accuracy of the model is plotted against each epoch.
