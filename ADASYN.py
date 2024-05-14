from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
# from keras.utils.np_utils import to_categorical # For older versions of Keras {keras<=2.3.1 tensorflow<=1.14.0}
from tensorflow.keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
import pickle
from sklearn.model_selection import train_test_split
import cv2
import os
from keras.callbacks import ModelCheckpoint
import webbrowser
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
#from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.preprocessing import LabelEncoder


main = tkinter.Tk()
main.title("Adaptive Synthetic Sampling") 
main.geometry("1300x1200")

global filename, spc_cnn, X, Y, dataset
global X_train, X_test, y_train, y_test
global accuracy, precision, recall, fscore
global labels, scaler, le1, le2, le3, le4

def uploadDataset(): 
    global dataset, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n")
    dataset = pd.read_csv(filename)
    text.insert(END, str(dataset))
    labels = np.unique(dataset['label'])
    label = dataset.groupby('label').size()
    label.plot(kind="bar")
    plt.title("NSL-KDD Attack Graph")
    plt.xlabel("Attack Name")
    plt.ylabel("Count")
    plt.show()


def preprocess():
    global dataset, X, Y, le1, le2, le3, le4
    text.delete('1.0', END)
    cols = ['protocol_type','service','flag','label']
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    le4 = LabelEncoder()
    dataset.fillna(0, inplace = True)
    dataset[cols[0]] = pd.Series(le1.fit_transform(dataset[cols[0]].astype(str)))
    dataset[cols[1]] = pd.Series(le2.fit_transform(dataset[cols[1]].astype(str)))
    dataset[cols[2]] = pd.Series(le3.fit_transform(dataset[cols[2]].astype(str)))
    dataset[cols[3]] = pd.Series(le4.fit_transform(dataset[cols[3]].astype(str)))
    data = dataset.values
    X = data[:,0:data.shape[1]-1]
    Y = data[:,data.shape[1]-1]
    print(X)
    print(Y)
    indices = np.arange(X.shape[0]) #shuffling the images
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    text.insert(END,"Dataset preprocessing & normalization & shuffling process completed\n\n")
    text.insert(END,str(dataset.head())) 

def augmentation():
    text.delete('1.0', END)
    global X, Y
    #now apply smote augmentation to balance data by adding synthesize records to fewer class
    sm = SMOTE(random_state = 2) #defining smote object
    X, Y = sm.fit_sample(X, Y)
    unique, count = np.unique(Y, return_counts=True)
    text.insert(END,"Records in each class after applying ADASYN Augmentation\n\n")
    text.insert(END,labels[0]+" Number of records : "+str(count[0])+"\n")
    text.insert(END,labels[1]+" Number of records : "+str(count[1])+"\n")
    text.insert(END,labels[2]+" Number of records : "+str(count[2])+"\n")
    text.insert(END,labels[3]+" Number of records : "+str(count[3])+"\n")
    
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("NSL-KDD Attack Graph After ADASYN Augmentation")
    plt.xlabel("Attack Name")
    plt.ylabel("Count")
    plt.show()
    plt.show()

def trainTestSplit():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, X, Y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset Train and Test Split\n\n")
    text.insert(END,"80% records used to train Algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% records used to test Algorithms : "+str(X_test.shape[0])+"\n")
    text.update_idletasks()
    
       
def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def runSPCCNN():
    global X_train, X_test, y_train, y_test, spc_cnn
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    accuracy = []
    precision = []
    recall = []
    fscore = []
    X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
    X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    #defining CNN2D layers
    spc_cnn = Sequential()
    #cnn layer to filtered features and then send to max pooling layer
    spc_cnn.add(Convolution2D(32, (1, 1), input_shape = (X_train1.shape[1],X_train1.shape[2],X_train1.shape[3]), activation = 'relu'))
    #max pooling layer remove irrrelavant and redundant fetaures and then select only optimized features to train SPC-CNN model
    spc_cnn.add(MaxPooling2D(pool_size = (1, 1)))
    spc_cnn.add(Convolution2D(32, (1, 1), activation = 'relu'))
    spc_cnn.add(MaxPooling2D(pool_size = (1, 1)))
    #flatten will be used to convert dataset features in to 2D format
    spc_cnn.add(Flatten())
    spc_cnn.add(Dense(output_dim = 256, activation = 'relu'))
    spc_cnn.add(Dense(output_dim = y_train1.shape[1], activation = 'softmax'))
    spc_cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) #compiling custome model
    if os.path.exists('model/model_weights.hdf5') == False:
        model_check_point = ModelCheckpoint(filepath='model/model_weights.hdf5', verbose = 1, save_best_only = True)
        #training the model
        hist = spc_cnn.fit(X_train1, y_train1, batch_size = 16, epochs = 20, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        spc_cnn.load_weights('model/model_weights.hdf5')
    predict = spc_cnn.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    target = np.argmax(y_test1, axis=1)
    calculateMetrics("Propose ADASYN-SPC-CNN", predict, target)

def runNaiveBayes():
    global X_train, X_test, y_train, y_test
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    predict = nb.predict(X_test)
    calculateMetrics("Naive Bayes", predict, y_test)

def runSVM():
    global X_train, X_test, y_train, y_test
    svm_cls = svm.SVC()
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM", predict, y_test)


def graph():
    output = "<html><body><table align=center border=1><tr><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th>"
    output+="<th>FSCORE</th></tr>"
    output+="<tr><td>Propose ADASYN-SPC-CNN</td><td>"+str(accuracy[0])+"</td><td>"+str(precision[0])+"</td><td>"+str(recall[0])+"</td><td>"+str(fscore[0])+"</td></tr>"
    output+="<tr><td>Naive Bayes</td><td>"+str(accuracy[1])+"</td><td>"+str(precision[1])+"</td><td>"+str(recall[1])+"</td><td>"+str(fscore[1])+"</td></tr>"
    output+="<tr><td>SVM</td><td>"+str(accuracy[2])+"</td><td>"+str(precision[2])+"</td><td>"+str(recall[2])+"</td><td>"+str(fscore[2])+"</td></tr>"
    output+="</table></body></html>"
    f = open("table.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("table.html",new=2)
    
    df = pd.DataFrame([['Propose ADASYN-SPC-CNN','Precision',precision[0]],['Propose ADASYN-SPC-CNN','Recall',recall[0]],['Propose ADASYN-SPC-CNN','F1 Score',fscore[0]],['Propose ADASYN-SPC-CNN','Accuracy',accuracy[0]],
                       ['Existing Naive Bayes','Precision',precision[1]],['Existing Naive Bayes','Recall',recall[1]],['Existing Naive Bayes','F1 Score',fscore[1]],['Existing Naive Bayes','Accuracy',accuracy[1]],
                       ['SVM','Precision',precision[2]],['SVM','Recall',recall[2]],['SVM','F1 Score',fscore[2]],['SVM','Accuracy',accuracy[2]],
                      ],columns=['Algorithms','Performance Output','Value'])
    df.pivot("Algorithms", "Performance Output", "Value").plot(kind='bar')
    plt.title("All Algorithms Performance Graph")
    plt.show()


def predict():
    global spc_cnn, scaler, le1, le2, le3, le4, labels
    cols = ['protocol_type','service','flag']
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    temp = dataset.values
    dataset[cols[0]] = pd.Series(le1.transform(dataset[cols[0]].astype(str)))
    dataset[cols[1]] = pd.Series(le2.transform(dataset[cols[1]].astype(str)))
    dataset[cols[2]] = pd.Series(le3.transform(dataset[cols[2]].astype(str)))
    dataset = dataset.values
    #dataset = scaler.transform(dataset)
    dataset = np.reshape(dataset, (dataset.shape[0], dataset.shape[1], 1, 1))
    predict = spc_cnn.predict(dataset)
    predict = np.argmax(predict, axis=1)
    print(predict)
    for i in range(len(predict)):
        text.insert(END,"Test Data = "+str(temp[i])+" Predicted As ====> "+labels[predict[i]]+"\n\n")
    


font = ('times', 15, 'bold')
title = Label(main, text='Wireless Network Intrusion Detection Method Based on Adaptive Synthetic Sampling and an Improved Convolutional Neural Network')
title.config(bg='PaleGreen2', fg='Khaki4')
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload NSL-KDD Dataset", command=uploadDataset, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=preprocess, bg='#ffb3fe')
processButton.place(x=340,y=550)
processButton.config(font=font1) 



splitButton = Button(main, text="Dataset Train & Test Split", command=trainTestSplit, bg='#ffb3fe')
splitButton.place(x=870,y=550)
splitButton.config(font=font1) 

spccnnButton = Button(main, text="Train SPC-CNN Algorithm", command=runSPCCNN, bg='#ffb3fe')
spccnnButton.place(x=50,y=600)
spccnnButton.config(font=font1) 

existingButton = Button(main, text="Run Naive Bayes Algorithms", command=runNaiveBayes, bg='#ffb3fe')
existingButton.place(x=340,y=600)
existingButton.config(font=font1)

graphButton = Button(main, text="Run SVM Algorithm", command=runSVM, bg='#ffb3fe')
graphButton.place(x=570,y=600)
graphButton.config(font=font1)

predictButton = Button(main, text="Comparison Graph", command=graph, bg='#ffb3fe')
predictButton.place(x=870,y=600)
predictButton.config(font=font1)

predictButton = Button(main, text="Predict Attack from Test Data", command=predict, bg='#ffb3fe')
predictButton.place(x=50,y=650)
predictButton.config(font=font1)

main.config(bg='PeachPuff2')
main.mainloop()
