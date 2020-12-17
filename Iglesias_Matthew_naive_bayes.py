#Created by: Dr. Olac Fuentes
#Modified by: Matthew Iglesias
#Exercise 2 - Naive Bayes

import numpy as np
import matplotlib.pyplot as plt
import time 

class naive_bayes():
    def __init__(self):  
        self.n_classes = None
        self.p_att_given_class = None
        
    def fit(self,X,y): 
        # Assumes X is binary
        self.n_classes = np.amax(y)+1
        self.p_class = np.zeros(self.n_classes)
        self.p_att_given_class = np.zeros((self.n_classes,X.shape[1]))
        for i in range(self.n_classes):
            self.p_class[i] = np.sum(y==i)/len(y)
            self.p_att_given_class[i] = np.mean(X[y==i],axis=0)
            
    def predict(self,x_test):
        pred =  np.zeros(x_test.shape[0],dtype=int)
        probs =  np.zeros((x_test.shape[0],self.n_classes))
        for i,x in enumerate(x_test):
            p = self.p_att_given_class*x + (1-self.p_att_given_class)*(1-x)
            p = np.log(p)
            result = np.sum(p,axis =1)
            #m = np.prod(p,axis=1)
            p2 = np.log(self.p_class) + result #Question 2 
            probs[i] = p2
        probs += 1e-200 #Smoothing out the values, help from Fuentes
        pred = np.argmax(probs,axis=1)
        return pred, probs
   
def display_probabilities(P):
    fig, ax = plt.subplots(1,10,figsize=(10,1))
    for i in range(10):
        ax[i].imshow(P[i].reshape((28,28)),cmap='gray')
        ax[i].axis('off')    
    
def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/y_true.shape[0]
   
def split_train_test(X,y,percent_train=0.9):
    ind = np.random.permutation(X.shape[0])
    train = ind[:int(X.shape[0]*percent_train)]
    test = ind[int(X.shape[0]*percent_train):]
    return X[train],X[test], y[train], y[test]  

def Q3_fit(X,y):
    stand_dev = np.zeros((10,784), dtype=float)
    m = np.zeros((10,784), dtype=float)
    for i in range(10):
        stand_dev[i] = np.std(X[y==i],axis=0)
        m[i] = np.mean(X[y==i],axis=0)
    stand_dev += 1e-1
    return stand_dev,m

def Q3_predict(X_test,stand_dev,m):
    pred = np.zeros(X_test.shape[0],dtype=int)
    p = []
    for i,x in enumerate(X_test):
        for j in range(10):
            result = np.e**(-1*((x-m[j])**2)/(stand_dev[j]**2))/(np.sqrt(2*3.14159*stand_dev[j]))
            p.append(np.prod(result)) #Product
        pred[i] = np.argmax(p)
        p = []
    return pred
    
if __name__ == "__main__":  
    plt.close('all')
   
    X = np.load('mnist_X.npy').astype(np.float32).reshape(-1,28*28)
    X2 = np.load('mnist_X.npy').astype(np.float32).reshape(-1,28*28)/255 #Smoothing the values by dividing X by 255 help from Fuentes
    y = np.load('mnist_y.npy')
    
    thr = 127.5
    X = (X>thr).astype(int)
     
    X_train, X_test, y_train, y_test = split_train_test(X,y)
    X_train2, X_test2, y_train2, y_test2 = split_train_test(X2,y)
    
    
    model = naive_bayes()
    start = time.time()
    model.fit(X_train, y_train)
    
    stand_dev,m = Q3_fit(X_train2, y_train2)
    
    elapsed_time = time.time()-start
    print('Elapsed_time training:  {0:.6f} secs'.format(elapsed_time))  
    
    plt.close('all')
    display_probabilities(model.p_att_given_class)
    
    start = time.time()       
    pred, probs = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing: {0:.6f} secs'.format(elapsed_time))   
    print('Accuracy: {0:.6f} '.format(accuracy(pred,y_test)))
    
    start = time.time()       
    Q3_pred = Q3_predict(X_test2,stand_dev,m)
    elapsed_time = time.time()-start
    print('Elapsed_time testing: {0:.6f} secs'.format(elapsed_time))   
    print('Accuracy: {0:.6f} '.format(accuracy(Q3_pred,y_test2)))
    
    
    
    