import numpy as np
import pandas as pd
import scipy.io as sp
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def to_classlabel(z):
    return z.argmax(axis = 1)

def one_hot_encode(y):
    n_class = np.unique(y).shape[0]
    y_encode = np.zeros((y.shape[0], n_class))
    for idx, val in enumerate(y):
        y_encode[idx, val] = 1.0
    return y_encode

# Define Accuracy
def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc

def loss(X, Y, W):
    """
    Y: onehot encoded
    """
    Z = - X @ W
    N = X.shape[0]
    loss = 1/N * (np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
    return loss

def gradient(X, Y, W, mu):
    """
    Y: onehot encoded
    """
    Z = - X @ W
    P = softmax(Z, axis=1)
    N = X.shape[0]
    gd = 1/N * (X.T @ (Y - P)) + 2 * mu * W
    return gd

def gradient_descent(X, Y, max_iter=1000, eta=0.1, mu=0.01):
    """
    Very basic gradient descent algorithm with fixed eta and mu
    """
    Y_onehot = one_hot_encode(Y)
    W = np.zeros((X.shape[1], Y_onehot.shape[1]))
    step = 0
    step_lst = []
    loss_lst = []
    W_lst = []

    while step < max_iter:
        step += 1
        W -= eta * gradient(X, Y_onehot, W, mu)
        step_lst.append(step)
        W_lst.append(W)
        loss_lst.append(loss(X, Y_onehot, W))

    df = pd.DataFrame({
        'step': step_lst,
        'loss': loss_lst
    })
    return df, W

# Multiclass logistic regression
class Multiclass:
    def fit(self, X, Y):
        self.loss_steps, self.W = gradient_descent(X, Y)

    def loss_plot(self):
        return self.loss_steps.plot(
            x='step',
            y='loss',
            xlabel='Epochs',
            ylabel='MCCE loss'
        )

    def predict(self, H):
        Z = - H @ self.W
        P = softmax(Z, axis=1)
        return to_classlabel(P)

# load dataset
mat = sp.loadmat("C:/Users/benit/Downloads/iris_dataset.mat")
inputs = mat['irisInputs'].T
targets = to_classlabel(mat['irisTargets'].T)

# Split the data
X_train,X_test,Y_train,Y_test = train_test_split(inputs,targets,test_size=0.30,random_state=1234)

# Standardize the data
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.fit_transform(X_test)

# fit model
model = Multiclass()
model.fit(X_train, Y_train)

# plot loss
model.loss_plot()

# predict
train_pred = model.predict(X_train)

# Calculate accuracy
train_acc = accuracy(Y_train, train_pred)
print(f'Accuracy on training set: {train_acc}')

# Calculate metrics
cm_train = confusion_matrix(Y_train, train_pred)
train_report = classification_report(Y_train, train_pred)

print("Performance on training set:\n")
print(f'Confusion Matrix:\n {cm_train}\n')
print(f'Classification Report:\n {train_report}')

# Create predictions on test set
test_pred = model.predict(X_test)

# Calculate accuracy
test_acc = accuracy(Y_test, test_pred)
print(f'Accuracy on test set: {test_acc}')

# Calculate metrics
cm_test = confusion_matrix(Y_test, test_pred)
test_report = classification_report(Y_test, test_pred)

print("Performance on test set:\n")
print(f'Confusion Matrix:\n {cm_test}\n')
print(f'Classification Report:\n {test_report}')

THETA = model.W
print(targets.shape)
print(train_pred.shape)
print(Y_train.shape)
