from tkinter import * 
from tkinter import ttk as ttk
import tkinter as tk
import tkinter.filedialog as fd
import tkinter.messagebox as mb
import pandas as pd
import numpy as np
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from nltk.stem.porter import PorterStemmer
import nltk
import random
from collections import Counter
import matplotlib.pyplot as plt

ps = PorterStemmer()
stop_words=set(stopwords.words("english"))
cv = CountVectorizer(max_features = 4000) #to select top 4000 words most used
reg=LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=1001)
lab=LabelEncoder()


def info():
    mb.showinfo("Info","Please browse a file first")

def openfile():
    filename=fd.askopenfilename()
    e1.insert(0,filename)


def pie():
    plt.show()
    

def model():
    fh=open("input_data.csv")
    fh2=open("random_data.csv","w+")
    fh2.write("id,text,emotions\n")
    contents=[]
    for line in fh:
        contents.append(line)
    for i in range(0,50000):
        i=random.randint(1,416809)
        fh2.write(contents[i])
    fh.close()
    fh2.close()
    dataset=pd.read_csv("random_data.csv",encoding='cp1252')
    processed_list = []
    

    for i in range(50000):
        contents=re.sub('@[\w]*',' ',dataset['text'][i])
        contents = re.sub('[^a-zA-Z]', ' ', contents)
        contents = contents.lower()
        contents = contents.split()
        filtered_sent=[]
        for w in contents:
            if w not in stop_words:
                filtered_sent.append(ps.stem(w))
        
        filtered_sent = ' '.join(filtered_sent)
        processed_list.append(filtered_sent)

    
    X = cv.fit_transform(processed_list) #convert it in string and store data in X
 

    y=dataset["emotions"]  
    y=y[0:50000]
    
    y=lab.fit_transform(y) #to  make y as interger type label
    reg.fit(X,y)

def DisplayOnGUI(tone):
    ta=Text(root,height=1,width=40,bg="slategray1")
    ta.insert(tk.END,tone)
    ta.place(x=100,y=250)
    bt3=Button(root,text="Details",fg="white",bg="SteelBlue",width=10,font="Arial 10 bold",command=pie)
    bt3.place(x=330,y=272)
    
def display_result(result):
    result_list=result.tolist()
    result2=Counter(result_list)
    count=0
    for ele in result_list:
        curr_freq=result_list.count(ele)
        if curr_freq>count:
            count=curr_freq
            label=ele
    tone="The tone of the essay is: "+str(label)
    DisplayOnGUI(tone)

    unique_label=[]
    sizes=[]
    
    for ele in result_list:
        if ele not in unique_label:
            unique_label.append(ele)
            sizes.append(result2[ele])
  
    plt.pie(sizes,labels=unique_label, autopct='%1.1f%%',shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    
def train(X_test): 
    y_pred=reg.predict(X_test)
    result=lab.inverse_transform(y_pred)
    display_result(result)    
    #print("accuracy",metrics.accuracy_score(y_test,y_pred))


def convert_into_words(contents):

    tokenized_text=sent_tokenize(contents)

    processed_list=[]
    for i in tokenized_text:
        con=re.sub('@[\w]*',' ',i)
        con = re.sub('[^a-zA-Z]', ' ', con)
        con = con.lower()
        con = con.split()
        filtered_sent=[]
        for w in con:
            if w not in stop_words:
                filtered_sent.append(ps.stem(w))
        
        filtered_sent = ' '.join(filtered_sent)
        processed_list.append(filtered_sent)
    
    X_test = cv.transform(processed_list) #convert it in string and store data in X
    return X_test

    
def textmining(event):
    filename=str(e1.get())
    if filename =="":
        info()   
    else:
        result=re.search(r'\.([A-Za-z0-9]+$)',filename)
        if result:
            if str(result.group(1))!="txt":
                
                e1.delete(0,'end')
                mb.showerror("Error","Only .txt files supported!")
                root.destroy()
            else:
                e1.delete(0,'end')
                fh=open(filename,"r")
                contents=fh.read()

                model()
                X_test=convert_into_words(contents)
                #print("Filterd Sentence:",X_test)
                train(X_test)


root=Tk()
root.title("Essay Tone Detector")
root.geometry("500x500")
root.geometry("500x500+100+100")
root.resizable(False,False)
root.config(background="slategray1")
logo = tk.PhotoImage(file="logo.png")

w1 = tk.Label(root, image=logo)
w1.place(x=405,y=20)

l1=Label(root,text="Essay Tone Detector",fg="white",bg="Skyblue4",font="Calibri 20 bold",relief=RIDGE,padx=10)
l1.place(x=100,y=40)

bt=Button(root,text="Click here to browse File",fg="white",bg="SteelBlue",width=20,font="Arial 10 bold",command=openfile)
bt.place(x=20,y=150)

e1=Entry(root,width=45)
e1.place(x=210,y=155)

bt2=Button(root,text="Upload",fg="white",bg="SteelBlue",width=10,font="Arial 10 bold")
bt2.place(x=180,y=210)
bt2.bind('<Button>',textmining)
