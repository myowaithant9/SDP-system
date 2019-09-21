import sys
from ui2 import *
# from app import *
from main import *

# import matplotlib
# matplotlib.use('TkAgg')
# import numpy as np
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.figure import Figure

import tkinter as tk
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# for table ui
from tkinter import *
import tkinter.ttk as ttk
import csv

import pandas as pd

from scipy.io import arff

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

import interface_support
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

from PIL import ImageTk,Image 

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


dataset = 0
# modelctrl = 0
# select data-> view data
dataset_path = ""
datactrl = 0

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    top = Toplevel1 (root)
    interface_support.init(root, top)
    root.mainloop()

w = None
def create_Toplevel1(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = tk.Toplevel (root)
    interface_support.set_Tk_var()
    top = Toplevel1 (w)
    interface_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_Toplevel1():
    global w
    w.destroy()
    w = None

class Toplevel1:
    # def uitwo(self):
    #     vp2_start_gui()

    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'

        top.geometry("868x507+321+158")
        top.title("New Toplevel")
        top.configure(background="#a6ddf4")
        top.configure(highlightbackground="#d9d9d9")
        top.configure(highlightcolor="black")

        prj_info = prjInfo()

    # def prj_info(self, top=None):
        # self.Frame2 = tk.Frame(top)
        # self.Frame2.place(relx=0.259, rely=-0.02, relheight=1.016
        #         , relwidth=0.743)
        # self.Frame2.configure(relief='groove')
        # self.Frame2.configure(borderwidth="2")
        # self.Frame2.configure(relief="groove")
        # self.Frame2.configure(background="#a6ddf4")
        # self.Frame2.configure(highlightbackground="#d9d9d9")
        # self.Frame2.configure(highlightcolor="black")      

class prjInfo:
    def __init__(self, top=None):
        # -----------------------------siedebar start-----------------------------#
        self.Frame1 = tk.Frame(top)
        self.Frame1.place(relx=0.0, rely=-0.02, relheight=1.036, relwidth=0.259)
        self.Frame1.configure(relief='groove')
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief="groove")
        self.Frame1.configure(background="#21b5ff")
        self.Frame1.configure(highlightbackground="#d9d9d9")
        self.Frame1.configure(highlightcolor="black")

        self.Label2 = tk.Label(self.Frame1)
        self.Label2.place(relx=0.178, rely=0.057, height=52, width=141)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(activeforeground="black")
        self.Label2.configure(background="#21b5ff")
        self.Label2.configure(font="-family {Al Bayan} -size 20 -weight bold")
        self.Label2.configure(foreground="white")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''SDP System''')

        self.Button2 = tk.Button(self.Frame1)
        self.Button2.place(relx=0.0, rely=0.419, height=62, width=227)
        self.Button2.configure(activebackground="#ececec")
        self.Button2.configure(activeforeground="#000000")
        self.Button2.configure(background="#d9d9d9")
        self.Button2.configure(font="-family {Al Bayan} -size 16")
        self.Button2.configure(foreground="blue")
        self.Button2.configure(highlightbackground="#d9d9d9")
        self.Button2.configure(highlightcolor="black")
        self.Button2.configure(justify='left')
        self.Button2.configure(relief="raised")
        self.Button2.configure(text='''Preprocessing''')

        self.Button3 = tk.Button(self.Frame1)
        self.Button3.place(relx=0.0, rely=0.305, height=62, width=227)
        self.Button3.configure(activebackground="#ececec")
        self.Button3.configure(activeforeground="#000000")
        self.Button3.configure(background="#d9d9d9")
        self.Button3.configure(font="-family {Al Bayan} -size 16")
        self.Button3.configure(foreground="blue")
        self.Button3.configure(highlightbackground="#d9d9d9")
        self.Button3.configure(highlightcolor="black")
        self.Button3.configure(justify='left')
        self.Button3.configure(relief="raised")
        self.Button3.configure(text='''Dataset Selection''')

        self.Button4 = tk.Button(self.Frame1)
        self.Button4.place(relx=0.0, rely=0.533, height=62, width=227)
        self.Button4.configure(activebackground="#ececec")
        self.Button4.configure(activeforeground="#000000")
        self.Button4.configure(background="#d9d9d9")
        self.Button4.configure(font="-family {Al Bayan} -size 16")
        self.Button4.configure(foreground="blue")
        self.Button4.configure(highlightbackground="#d9d9d9")
        self.Button4.configure(highlightcolor="black")
        self.Button4.configure(justify='left')
        self.Button4.configure(relief="raised")
        self.Button4.configure(text='''Discretization''')

        self.Button5 = tk.Button(self.Frame1)
        self.Button5.place(relx=0.0, rely=0.648, height=62, width=227)
        self.Button5.configure(activebackground="#ececec")
        self.Button5.configure(activeforeground="#000000")
        self.Button5.configure(background="#d9d9d9")
        self.Button5.configure(font="-family {Al Bayan} -size 16")
        self.Button5.configure(foreground="blue")
        self.Button5.configure(highlightbackground="#d9d9d9")
        self.Button5.configure(highlightcolor="black")
        self.Button5.configure(justify='left')
        self.Button5.configure(relief="raised")
        self.Button5.configure(text='''Model Selection''')

        self.Button1 = tk.Button(self.Frame1)
        self.Button1.place(relx=0.0, rely=0.19, height=62, width=227)
        self.Button1.configure(activebackground="#ececec")#21b5ff
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")

        self.Button1.configure(font="-family {Al Bayan} -size 16")
        self.Button1.configure(foreground="green")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(justify='left')
        self.Button1.configure(overrelief="flat")
        self.Button1.configure(relief="raised")
        self.Button1.configure(text='''*** Project Info ***''')

        self.Button6 = tk.Button(self.Frame1)
        self.Button6.place(relx=0.0, rely=0.762, height=62, width=227)
        self.Button6.configure(activebackground="#ececec")
        self.Button6.configure(activeforeground="#000000")
        self.Button6.configure(background="#d9d9d9")
        self.Button6.configure(font="-family {Al Bayan} -size 16")
        self.Button6.configure(foreground="blue")
        self.Button6.configure(highlightbackground="#d9d9d9")
        self.Button6.configure(highlightcolor="black")
        self.Button6.configure(justify='left')
        self.Button6.configure(relief="raised")
        self.Button6.configure(text='''Feature Selection''')

        self.Button7 = tk.Button(self.Frame1)
        self.Button7.place(relx=0.0, rely=0.876, height=62, width=227)
        self.Button7.configure(activebackground="#ececec")
        self.Button7.configure(activeforeground="#000000")
        self.Button7.configure(background="#d9d9d9")
        self.Button7.configure(font="-family {Al Bayan} -size 16")
        self.Button7.configure(foreground="blue")
        self.Button7.configure(highlightbackground="#d9d9d9")
        self.Button7.configure(highlightcolor="black")
        self.Button7.configure(justify='left')
        self.Button7.configure(relief="raised")
        self.Button7.configure(text='''Outcome''')
        # -----------------------------siedebar end-----------------------------#

        self.Frame2 = tk.Frame(top)
        self.Frame2.place(relx=0.258, rely=-0.01, relheight=1.016
                , relwidth=0.743)
        self.Frame2.configure(relief='groove')
        self.Frame2.configure(borderwidth="2")
        self.Frame2.configure(relief="groove")
        self.Frame2.configure(background="#a6ddf4")
        self.Frame2.configure(highlightbackground="#d9d9d9")
        self.Frame2.configure(highlightcolor="black")

        self.Label1 = tk.Label(self.Frame2)
        self.Label1.place(relx=0.13, rely=0.07, height=46, width=475)
        self.Label1.configure(activebackground="#a6ddf4")
        self.Label1.configure(activeforeground="#1c38ed")
        self.Label1.configure(background="#a6ddf4")
        self.Label1.configure(font="-family {Al Bayan} -size 20 -weight bold")
        self.Label1.configure(foreground="#171fff")
        self.Label1.configure(highlightbackground="#ffffffffffff")
        self.Label1.configure(highlightcolor="#2b1582")
        self.Label1.configure(text='''Welcome to Software Defect Prediction System''')

        self.Label3 = tk.Label(self.Frame2)
        self.Label3.place(relx=0.01, rely=0.214, height=31, width=161)
        # self.Label3.place(relx=0.265, rely=0.197, height=31, width=161)
        self.Label3.configure(activebackground="#f9f9f9")
        self.Label3.configure(activeforeground="black")
        self.Label3.configure(background="#a6ddf4")
        self.Label3.configure(font="-family {Al Bayan} -size 16")
        self.Label3.configure(foreground="#000000")
        self.Label3.configure(highlightbackground="#d9d9d9")
        self.Label3.configure(highlightcolor="black")
        self.Label3.configure(text='''Project Description :''')
        
        self.Button1 = tk.Button(self.Frame2)
        self.Button1.place(relx=0.8, rely=0.204, height=31, width=120)
        self.Button1.configure(activebackground="#ececec")
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")
        self.Button1.configure(font="-family {Al Bayan} -size 16")
        self.Button1.configure(foreground="blue")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(relief="raised")
        self.Button1.configure(text='''Start''')
        self.Button1.configure(command=selectData)

        self.Message1 = tk.Message(self.Frame2)
        self.Message1.place(relx=0.015, rely=0.31, relheight=0.105
                , relwidth=0.96)
        # self.Message1.place(relx=0.265, rely=0.286, relheight=0.105
        #         , relwidth=0.718)
        self.Message1.configure(background="#a6ddf4")
        self.Message1.configure(font="-family {Al Bayan} -size 16")
        self.Message1.configure(foreground="#000000")
        self.Message1.configure(highlightbackground="#d9d9d9")
        self.Message1.configure(highlightcolor="black")
        self.Message1.configure(text='''          Software Defect Prediction mechanisms are used to enhance the work of SQA process through the prediction of defect modules comprised with 5 stages :''')
        # self.Message1.configure(width=823)
        self.Message1.configure(width=623)

        self.Label4 = tk.Label(self.Frame2)
        self.Label4.place(relx=0.0138, rely=0.43, height=41, width=595)
        self.Label4.configure(activebackground="#f9f9f9")
        self.Label4.configure(activeforeground="black")
        self.Label4.configure(background="#a6ddf4")
        self.Label4.configure(font="-family {Al Bayan} -size 16")
        self.Label4.configure(foreground="#000000")
        self.Label4.configure(highlightbackground="#d9d9d9")
        self.Label4.configure(highlightcolor="black")
        self.Label4.configure(text='''1. Preprocessing       :   Remove null value, missing value and impossible instance''')

        self.Label5 = tk.Label(self.Frame2)
        self.Label5.place(relx=0.006, rely=0.525, height=31, width=586)
        self.Label5.configure(activebackground="#f9f9f9")
        self.Label5.configure(activeforeground="black")
        self.Label5.configure(background="#a6ddf4")
        self.Label5.configure(font="-family {Al Bayan} -size 16")
        self.Label5.configure(foreground="#000000")
        self.Label5.configure(highlightbackground="#d9d9d9")
        self.Label5.configure(highlightcolor="black")
        self.Label5.configure(justify='left')
        self.Label5.configure(text='''2. Discretization        :   Convert continuous data values into finite set of values''')

        self.Message2 = tk.Message(self.Frame2)
        self.Message2.place(relx=0.01, rely=0.74, relheight=0.099
                , relwidth=0.96)
        self.Message2.configure(background="#a6ddf4")
        self.Message2.configure(font="-family {Al Bayan} -size 16")
        self.Message2.configure(foreground="#000000")
        self.Message2.configure(highlightbackground="#d9d9d9")
        self.Message2.configure(highlightcolor="black")
        self.Message2.configure(text='''4. Model Building      :   Adaptive Boosting based on Support Vector Machine-Radial \t\t Basis Function''')
        self.Message2.configure(width=625)

        self.Message3 = tk.Message(self.Frame2)
        self.Message3.place(relx=0.005, rely=0.615, relheight=0.085
                , relwidth=0.96)
        self.Message3.configure(background="#a6ddf4")
        self.Message3.configure(font="-family {Al Bayan} -size 16")
        self.Message3.configure(foreground="#000000")
        self.Message3.configure(highlightbackground="#d9d9d9")
        self.Message3.configure(highlightcolor="black")
        self.Message3.configure(text='''3. Feature Selection :  Select features using Minimum Redundancy and Maximum \t\t Relevance (MRMR)''')
        self.Message3.configure(width=624)

        self.Message4 = tk.Message(self.Frame2)
        self.Message4.place(relx=0.0045, rely=0.858, relheight=0.114
                , relwidth=0.96)
        self.Message4.configure(background="#a6ddf4")
        self.Message4.configure(font="-family {Al Bayan} -size 16")
        self.Message4.configure(foreground="#000000")
        self.Message4.configure(highlightbackground="#d9d9d9")
        self.Message4.configure(highlightcolor="black")
        self.Message4.configure(text='''5. Comparison           :  Accuracy and performance measures used to compare with \t\t hybrid method against single classifier results''')
        self.Message4.configure(width=624)

class selectData:
    def viewData(self, top=None):
        global dataset_path
        global datactrl
        print('self.TCombobox1.get()-->', self.TCombobox1.get())
        if self.TCombobox1.get() == "PC1":
            datactrl = 1
            dataset_path = 'MDP csv/PC01.csv'
            dataset = pd.read_csv(dataset_path)

        if self.TCombobox1.get() == "PC2":
            datactrl = 2
            dataset_path = 'MDP csv/PC02.csv'
            dataset = pd.read_csv(dataset_path)

        if self.TCombobox1.get() == "PC3":
            datactrl = 3
            dataset_path = 'MDP csv/PC03.csv'
            dataset = pd.read_csv(dataset_path)

        if self.TCombobox1.get() == "PC4":
            datactrl = 4
            dataset_path = 'MDP csv/PC04.csv'
            dataset = pd.read_csv(dataset_path)

        if self.TCombobox1.get() == "PC5":
            datactrl = 5
            dataset_path = 'MDP csv/PC05.csv'
            dataset = pd.read_csv(dataset_path)

        print("dataset_path-->", dataset_path)

        
        self.Frame3 = tk.Frame(self.Frame2)
        self.Frame3.place(relx=0.0, rely=0.35, relheight=0.65, relwidth=1.0)
        self.Frame3.configure(relief='groove')
        self.Frame3.configure(borderwidth="2")
        self.Frame3.configure(relief="groove")
        self.Frame3.configure(background="#d9d9d9")

        TableMargin = Frame(self.Frame3, width=500)
        TableMargin.pack(side=TOP)
        scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
        scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
        tree = ttk.Treeview(TableMargin, columns=dataset.columns, height=400, selectmode="extended", yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
        scrollbary.config(command=tree.yview)
        scrollbary.pack(side=RIGHT, fill=Y)
        scrollbarx.config(command=tree.xview)
        scrollbarx.pack(side=BOTTOM, fill=X)
        # print('dataset.columns', dataset.shape[1])
      
        for col in range(dataset.shape[1]):
            # print('col-->', col)
            # print('column name', dataset.columns[col])
            tree.heading(col, text=dataset.columns[col], anchor=W)
            tree.column('#'+str(col), stretch=NO, minwidth=0, width=100)
        tree.column('#0', stretch=NO, minwidth=0, width=0)
        tree.pack()

        for i in range(dataset.shape[0]):
            print('val', dataset.values[i, :])
            tree.insert("", 0, values= dataset.values[i, :])

    def __init__(self, top=None):
        global dataset_path
        global datactrl
        datactrl = 0
        dataset_path = ''
        print('datactrl in init-->', datactrl)
        # -----------------------------siedebar start-----------------------------#
        self.Frame1 = tk.Frame(top)
        self.Frame1.place(relx=0.0, rely=-0.02, relheight=1.036, relwidth=0.259)
        self.Frame1.configure(relief='groove')
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief="groove")
        self.Frame1.configure(background="#21b5ff")
        self.Frame1.configure(highlightbackground="#d9d9d9")
        self.Frame1.configure(highlightcolor="black")

        self.Label2 = tk.Label(self.Frame1)
        self.Label2.place(relx=0.178, rely=0.057, height=52, width=141)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(activeforeground="black")
        self.Label2.configure(background="#21b5ff")
        self.Label2.configure(font="-family {Al Bayan} -size 20 -weight bold")
        self.Label2.configure(foreground="white")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''SDP System''')

        self.Button2 = tk.Button(self.Frame1)
        self.Button2.place(relx=0.0, rely=0.419, height=62, width=227)
        self.Button2.configure(activebackground="#ececec")
        self.Button2.configure(activeforeground="#000000")
        self.Button2.configure(background="#d9d9d9")
        self.Button2.configure(font="-family {Al Bayan} -size 16")
        self.Button2.configure(foreground="blue")
        self.Button2.configure(highlightbackground="#d9d9d9")
        self.Button2.configure(highlightcolor="black")
        self.Button2.configure(justify='left')
        self.Button2.configure(relief="raised")
        self.Button2.configure(text='''Preprocessing''')

        self.Button3 = tk.Button(self.Frame1)
        self.Button3.place(relx=0.0, rely=0.305, height=62, width=227)
        self.Button3.configure(activebackground="#ececec")
        self.Button3.configure(activeforeground="#000000")
        self.Button3.configure(background="#d9d9d9")
        self.Button3.configure(font="-family {Al Bayan} -size 16")
        self.Button3.configure(foreground="green")
        self.Button3.configure(highlightbackground="#d9d9d9")
        self.Button3.configure(highlightcolor="black")
        self.Button3.configure(justify='left')
        self.Button3.configure(relief="raised")
        self.Button3.configure(text='''*** Dataset Selection ***''')

        self.Button4 = tk.Button(self.Frame1)
        self.Button4.place(relx=0.0, rely=0.533, height=62, width=227)
        self.Button4.configure(activebackground="#ececec")
        self.Button4.configure(activeforeground="#000000")
        self.Button4.configure(background="#d9d9d9")
        self.Button4.configure(font="-family {Al Bayan} -size 16")
        self.Button4.configure(foreground="blue")
        self.Button4.configure(highlightbackground="#d9d9d9")
        self.Button4.configure(highlightcolor="black")
        self.Button4.configure(justify='left')
        self.Button4.configure(relief="raised")
        self.Button4.configure(text='''Discretization''')

        self.Button5 = tk.Button(self.Frame1)
        self.Button5.place(relx=0.0, rely=0.648, height=62, width=227)
        self.Button5.configure(activebackground="#ececec")
        self.Button5.configure(activeforeground="#000000")
        self.Button5.configure(background="#d9d9d9")
        self.Button5.configure(font="-family {Al Bayan} -size 16")
        self.Button5.configure(foreground="blue")
        self.Button5.configure(highlightbackground="#d9d9d9")
        self.Button5.configure(highlightcolor="black")
        self.Button5.configure(justify='left')
        self.Button5.configure(relief="raised")
        self.Button5.configure(text='''Model Selection''')

        self.Button1 = tk.Button(self.Frame1)
        self.Button1.place(relx=0.0, rely=0.19, height=62, width=227)
        self.Button1.configure(activebackground="#ececec")#21b5ff
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")

        self.Button1.configure(font="-family {Al Bayan} -size 16")
        self.Button1.configure(foreground="blue")

        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(justify='left')
        self.Button1.configure(overrelief="flat")
        self.Button1.configure(relief="raised")
        self.Button1.configure(text='''Project Info''')

        self.Button6 = tk.Button(self.Frame1)
        self.Button6.place(relx=0.0, rely=0.762, height=62, width=227)
        self.Button6.configure(activebackground="#ececec")
        self.Button6.configure(activeforeground="#000000")
        self.Button6.configure(background="#d9d9d9")
        self.Button6.configure(font="-family {Al Bayan} -size 16")
        self.Button6.configure(foreground="blue")
        self.Button6.configure(highlightbackground="#d9d9d9")
        self.Button6.configure(highlightcolor="black")
        self.Button6.configure(justify='left')
        self.Button6.configure(relief="raised")
        self.Button6.configure(text='''Feature Selection''')

        self.Button7 = tk.Button(self.Frame1)
        self.Button7.place(relx=0.0, rely=0.876, height=62, width=227)
        self.Button7.configure(activebackground="#ececec")
        self.Button7.configure(activeforeground="#000000")
        self.Button7.configure(background="#d9d9d9")
        self.Button7.configure(font="-family {Al Bayan} -size 16")
        self.Button7.configure(foreground="blue")
        self.Button7.configure(highlightbackground="#d9d9d9")
        self.Button7.configure(highlightcolor="black")
        self.Button7.configure(justify='left')
        self.Button7.configure(relief="raised")
        self.Button7.configure(text='''Outcome''')
        # -----------------------------siedebar end-----------------------------#

        self.Frame2 = tk.Frame(top)
        self.Frame2.place(relx=0.258, rely=-0.01, relheight=1.016
                , relwidth=0.743)
        self.Frame2.configure(relief='groove')
        self.Frame2.configure(borderwidth="2")
        self.Frame2.configure(relief="groove")
        self.Frame2.configure(background="#a6ddf4")
        self.Frame2.configure(highlightbackground="#d9d9d9")
        self.Frame2.configure(highlightcolor="black")

        self.Label1 = tk.Label(self.Frame2)
        self.Label1.place(relx=0.13, rely=0.07, height=46, width=475)
        self.Label1.configure(activebackground="#a6ddf4")
        self.Label1.configure(activeforeground="#1c38ed")
        self.Label1.configure(background="#a6ddf4")
        self.Label1.configure(font="-family {Al Bayan} -size 20 -weight bold")
        self.Label1.configure(foreground="#171fff")
        self.Label1.configure(highlightbackground="#ffffffffffff")
        self.Label1.configure(highlightcolor="#2b1582")
        self.Label1.configure(text='''Welcome to Software Defect Prediction System''')

        self.Frame3 = tk.Frame(self.Frame2)
        self.Frame3.place(relx=0.0, rely=0.35, relheight=0.65, relwidth=1.0)
        self.Frame3.configure(relief='groove')
        self.Frame3.configure(borderwidth="2")
        self.Frame3.configure(relief="groove")
        self.Frame3.configure(background="#d9d9d9")

        self.Label4 = tk.Label(self.Frame3)
        self.Label4.place(relx=0.13, rely=0.07, height=46, width=475)
        self.Label4.configure(background="#d9d9d9")
        self.Label4.configure(font="-family {Al Bayan} -size 16")
        self.Label4.configure(foreground="#171fff")
        self.Label4.configure(text='''Click "Import" button to view dataset''')


        self.Label2 = tk.Label(self.Frame2)
        self.Label2.place(relx=0.031, rely=0.214, height=31, width=126)
        self.Label2.configure(background="#a6ddf4")
        self.Label2.configure(font="-family {Al Bayan} -size 16")
        self.Label2.configure(foreground="#000000")
        self.Label2.configure(text='''Choose Dataset''')

        self.TCombobox1 = ttk.Combobox(self.Frame2)
        self.TCombobox1.place(relx=0.264, rely=0.214, relheight=0.052
                , relwidth=0.319)
        # self.TCombobox1.configure(textvariable=interface_support.combobox)
        self.TCombobox1['values']=("PC1", 'PC2', 'PC3', 'PC4', 'PC5')
        # self.TCombobox1['values']=(1,2,3)
        self.TCombobox1.configure(takefocus="")

        self.Button1 = tk.Button(self.Frame2)
        self.Button1.place(relx=0.62, rely=0.204, height=31, width=110)
        self.Button1.configure(activebackground="#ececec")
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")
        self.Button1.configure(font="-family {Al Bayan} -size 16")
        self.Button1.configure(foreground="blue")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(relief="raised")
        self.Button1.configure(text='''Import''')
        self.Button1.configure(command=self.viewData)

        self.Button2 = tk.Button(self.Frame2)
        self.Button2.place(relx=0.822, rely=0.204, height=31, width=110)
        self.Button2.configure(activebackground="#ececec")
        self.Button2.configure(activeforeground="#000000")
        self.Button2.configure(background="#d9d9d9")
        self.Button2.configure(font="-family {Al Bayan} -size 16")
        self.Button2.configure(foreground="blue")
        self.Button2.configure(highlightbackground="#d9d9d9")
        self.Button2.configure(highlightcolor="black")
        self.Button2.configure(relief="raised")
        self.Button2.configure(text='''Preprocess''')
        self.Button2.configure(command=preprocess)

class preprocess:
    def ui(self, top=None):
        # -----------------------------siedebar start-----------------------------#
        self.Frame1 = tk.Frame(top)
        self.Frame1.place(relx=0.0, rely=-0.02, relheight=1.036, relwidth=0.259)
        self.Frame1.configure(relief='groove')
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief="groove")
        self.Frame1.configure(background="#21b5ff")
        self.Frame1.configure(highlightbackground="#d9d9d9")
        self.Frame1.configure(highlightcolor="black")

        self.Label2 = tk.Label(self.Frame1)
        self.Label2.place(relx=0.178, rely=0.057, height=52, width=141)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(activeforeground="black")
        self.Label2.configure(background="#21b5ff")
        self.Label2.configure(font="-family {Al Bayan} -size 20 -weight bold")
        self.Label2.configure(foreground="white")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''SDP System''')

        self.Button2 = tk.Button(self.Frame1)
        self.Button2.place(relx=0.0, rely=0.419, height=62, width=227)
        self.Button2.configure(activebackground="#ececec")
        self.Button2.configure(activeforeground="#000000")
        self.Button2.configure(background="#d9d9d9")
        self.Button2.configure(font="-family {Al Bayan} -size 16")
        self.Button2.configure(foreground="green")
        self.Button2.configure(highlightbackground="#d9d9d9")
        self.Button2.configure(highlightcolor="black")
        self.Button2.configure(justify='left')
        self.Button2.configure(relief="raised")
        self.Button2.configure(text='''*** Preprocessing ***''')

        self.Button3 = tk.Button(self.Frame1)
        self.Button3.place(relx=0.0, rely=0.305, height=62, width=227)
        self.Button3.configure(activebackground="#ececec")
        self.Button3.configure(activeforeground="#000000")
        self.Button3.configure(background="#d9d9d9")
        self.Button3.configure(font="-family {Al Bayan} -size 16")
        self.Button3.configure(foreground="blue")
        self.Button3.configure(highlightbackground="#d9d9d9")
        self.Button3.configure(highlightcolor="black")
        self.Button3.configure(justify='left')
        self.Button3.configure(relief="raised")
        self.Button3.configure(text='''Dataset Selection''')

        self.Button4 = tk.Button(self.Frame1)
        self.Button4.place(relx=0.0, rely=0.533, height=62, width=227)
        self.Button4.configure(activebackground="#ececec")
        self.Button4.configure(activeforeground="#000000")
        self.Button4.configure(background="#d9d9d9")
        self.Button4.configure(font="-family {Al Bayan} -size 16")
        self.Button4.configure(foreground="blue")
        self.Button4.configure(highlightbackground="#d9d9d9")
        self.Button4.configure(highlightcolor="black")
        self.Button4.configure(justify='left')
        self.Button4.configure(relief="raised")
        self.Button4.configure(text='''Discretization''')

        self.Button5 = tk.Button(self.Frame1)
        self.Button5.place(relx=0.0, rely=0.648, height=62, width=227)
        self.Button5.configure(activebackground="#ececec")
        self.Button5.configure(activeforeground="#000000")
        self.Button5.configure(background="#d9d9d9")
        self.Button5.configure(font="-family {Al Bayan} -size 16")
        self.Button5.configure(foreground="blue")
        self.Button5.configure(highlightbackground="#d9d9d9")
        self.Button5.configure(highlightcolor="black")
        self.Button5.configure(justify='left')
        self.Button5.configure(relief="raised")
        self.Button5.configure(text='''Model Selection''')

        self.Button1 = tk.Button(self.Frame1)
        self.Button1.place(relx=0.0, rely=0.19, height=62, width=227)
        self.Button1.configure(activebackground="#ececec")#21b5ff
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")

        self.Button1.configure(font="-family {Al Bayan} -size 16")
        self.Button1.configure(foreground="blue")

        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(justify='left')
        self.Button1.configure(overrelief="flat")
        self.Button1.configure(relief="raised")
        self.Button1.configure(text='''Project Info''')

        self.Button6 = tk.Button(self.Frame1)
        self.Button6.place(relx=0.0, rely=0.762, height=62, width=227)
        self.Button6.configure(activebackground="#ececec")
        self.Button6.configure(activeforeground="#000000")
        self.Button6.configure(background="#d9d9d9")
        self.Button6.configure(font="-family {Al Bayan} -size 16")
        self.Button6.configure(foreground="blue")
        self.Button6.configure(highlightbackground="#d9d9d9")
        self.Button6.configure(highlightcolor="black")
        self.Button6.configure(justify='left')
        self.Button6.configure(relief="raised")
        self.Button6.configure(text='''Feature Selection''')

        self.Button7 = tk.Button(self.Frame1)
        self.Button7.place(relx=0.0, rely=0.876, height=62, width=227)
        self.Button7.configure(activebackground="#ececec")
        self.Button7.configure(activeforeground="#000000")
        self.Button7.configure(background="#d9d9d9")
        self.Button7.configure(font="-family {Al Bayan} -size 16")
        self.Button7.configure(foreground="blue")
        self.Button7.configure(highlightbackground="#d9d9d9")
        self.Button7.configure(highlightcolor="black")
        self.Button7.configure(justify='left')
        self.Button7.configure(relief="raised")
        self.Button7.configure(text='''Outcome''')
        # -----------------------------siedebar end-----------------------------#
        
        self.Frame2 = tk.Frame(top)
        self.Frame2.place(relx=0.258, rely=-0.01, relheight=1.016
                , relwidth=0.743)
        self.Frame2.configure(relief='groove')
        self.Frame2.configure(borderwidth="2")
        self.Frame2.configure(relief="groove")
        self.Frame2.configure(background="#a6ddf4")
        self.Frame2.configure(highlightbackground="#d9d9d9")
        self.Frame2.configure(highlightcolor="black")

        self.Label1 = tk.Label(self.Frame2)
        self.Label1.place(relx=0.13, rely=0.07, height=46, width=475)
        self.Label1.configure(activebackground="#a6ddf4")
        self.Label1.configure(activeforeground="#1c38ed")
        self.Label1.configure(background="#a6ddf4")
        self.Label1.configure(font="-family {Al Bayan} -size 20 -weight bold")
        self.Label1.configure(foreground="#171fff")
        self.Label1.configure(highlightbackground="#ffffffffffff")
        self.Label1.configure(highlightcolor="#2b1582")
        self.Label1.configure(text='''Welcome to Software Defect Prediction System''')

        self.Frame3 = tk.Frame(self.Frame2)
        self.Frame3.place(relx=0.0, rely=0.35, relheight=0.65, relwidth=1.0)
        self.Frame3.configure(relief='groove')
        self.Frame3.configure(borderwidth="2")
        self.Frame3.configure(relief="groove")
        self.Frame3.configure(background="#d9d9d9")

        self.Message1 = tk.Message(self.Frame2)
        self.Message1.place(relx=0.031, rely=0.19, relheight=0.11, relwidth=0.8)
        self.Message1.configure(background="#a6ddf4")
        self.Message1.configure(font="-family {Al Bayan} -size 16")
        self.Message1.configure(foreground="#000000")
        self.Message1.configure(highlightbackground="#d9d9d9")
        self.Message1.configure(highlightcolor="black")
        self.Message1.configure(anchor='nw')
        self.Message1.configure(text='''Preprocessing is done by removing item with impossible occurences and feature when all instances have same value''')
        self.Message1.configure(width=500)

        # self.Label2 = tk.Label(self.Frame2)
        # self.Label2.place(relx=0.031, rely=0.214, height=62, width=300)
        # self.Label2.configure(background="#a6ddf4")
        # self.Label2.configure(font="-family {Al Bayan} -size 16")
        # self.Label2.configure(foreground="#000000")
        # self.Label2.configure(justify='left')
        # self.Label2.configure(text='''Preprocessing is done by removing item with impossible occurences and remove feature when all instances have same value''')

        self.Button1 = tk.Button(self.Frame2)
        self.Button1.place(relx=0.8, rely=0.204, height=31, width=120)
        self.Button1.configure(activebackground="#ececec")
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")
        self.Button1.configure(font="-family {Al Bayan} -size 16")
        self.Button1.configure(foreground="blue")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(relief="raised")
        self.Button1.configure(text='''Discretize''')
        self.Button1.configure(command=discretize)

        TableMargin = Frame(self.Frame3, width=500)
        TableMargin.pack(side=TOP)
        scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
        scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
        tree = ttk.Treeview(TableMargin, columns=pre_dataset.columns, height=400, selectmode="extended", yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
        scrollbary.config(command=tree.yview)
        scrollbary.pack(side=RIGHT, fill=Y)
        scrollbarx.config(command=tree.xview)
        scrollbarx.pack(side=BOTTOM, fill=X)
        # print('dataset.columns', pre_dataset.shape[1])
      
        for col in range(pre_dataset.shape[1]):
            # print('col-->', col)
            # print('column name', pre_dataset.columns[col])
            tree.heading(col, text=pre_dataset.columns[col], anchor=W)
            tree.column('#'+str(col), stretch=NO, minwidth=0, width=100)

        tree.column('#0', stretch=NO, minwidth=0, width=0)
        # tree.column('#', stretch=NO, minwidth=0, width=100)

        tree.pack()

        for i in range(pre_dataset.shape[0]):
            print('val', pre_dataset.values[i, :])
            tree.insert("", 0, values= pre_dataset.values[i, :])

    def fun(self, top=None):
        print('dataset_path', dataset_path)
        print('datactrl', datactrl)
        global feature_data
        global target_data
        global pre_dataset

        pre_dataset, feature_data, target_data = main_preprocess(dataset_path, datactrl)
        # fun_preprocess(dataset_path)
        print('feature_data')
        print(feature_data)
        print(feature_data.shape)

        print('target_data')
        print(target_data)
        print(target_data.shape)

    def __init__(self, top=None):
        print('datactrl in pre-->', datactrl)

        self.fun()
        self.ui()

class discretize:
    def ui(self, top=None):
        # -----------------------------siedebar start-----------------------------#
        self.Frame1 = tk.Frame(top)
        self.Frame1.place(relx=0.0, rely=-0.02, relheight=1.036, relwidth=0.259)
        self.Frame1.configure(relief='groove')
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief="groove")
        self.Frame1.configure(background="#21b5ff")
        self.Frame1.configure(highlightbackground="#d9d9d9")
        self.Frame1.configure(highlightcolor="black")

        self.Label2 = tk.Label(self.Frame1)
        self.Label2.place(relx=0.178, rely=0.057, height=52, width=141)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(activeforeground="black")
        self.Label2.configure(background="#21b5ff")
        self.Label2.configure(font="-family {Al Bayan} -size 20 -weight bold")
        self.Label2.configure(foreground="white")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''SDP System''')

        self.Button2 = tk.Button(self.Frame1)
        self.Button2.place(relx=0.0, rely=0.419, height=62, width=227)
        self.Button2.configure(activebackground="#ececec")
        self.Button2.configure(activeforeground="#000000")
        self.Button2.configure(background="#d9d9d9")
        self.Button2.configure(font="-family {Al Bayan} -size 16")
        self.Button2.configure(foreground="blue")
        self.Button2.configure(highlightbackground="#d9d9d9")
        self.Button2.configure(highlightcolor="black")
        self.Button2.configure(justify='left')
        self.Button2.configure(relief="raised")
        self.Button2.configure(text='''Preprocessing''')

        self.Button3 = tk.Button(self.Frame1)
        self.Button3.place(relx=0.0, rely=0.305, height=62, width=227)
        self.Button3.configure(activebackground="#ececec")
        self.Button3.configure(activeforeground="#000000")
        self.Button3.configure(background="#d9d9d9")
        self.Button3.configure(font="-family {Al Bayan} -size 16")
        self.Button3.configure(foreground="blue")
        self.Button3.configure(highlightbackground="#d9d9d9")
        self.Button3.configure(highlightcolor="black")
        self.Button3.configure(justify='left')
        self.Button3.configure(relief="raised")
        self.Button3.configure(text='''Dataset Selection''')

        self.Button4 = tk.Button(self.Frame1)
        self.Button4.place(relx=0.0, rely=0.533, height=62, width=227)
        self.Button4.configure(activebackground="#ececec")
        self.Button4.configure(activeforeground="#000000")
        self.Button4.configure(background="#d9d9d9")
        self.Button4.configure(font="-family {Al Bayan} -size 16")
        self.Button4.configure(foreground="green")
        self.Button4.configure(highlightbackground="#d9d9d9")
        self.Button4.configure(highlightcolor="black")
        self.Button4.configure(justify='left')
        self.Button4.configure(relief="raised")
        self.Button4.configure(text='''*** Discretization ***''')

        self.Button5 = tk.Button(self.Frame1)
        self.Button5.place(relx=0.0, rely=0.648, height=62, width=227)
        self.Button5.configure(activebackground="#ececec")
        self.Button5.configure(activeforeground="#000000")
        self.Button5.configure(background="#d9d9d9")
        self.Button5.configure(font="-family {Al Bayan} -size 16")
        self.Button5.configure(foreground="blue")
        self.Button5.configure(highlightbackground="#d9d9d9")
        self.Button5.configure(highlightcolor="black")
        self.Button5.configure(justify='left')
        self.Button5.configure(relief="raised")
        self.Button5.configure(text='''Model Selection''')

        self.Button1 = tk.Button(self.Frame1)
        self.Button1.place(relx=0.0, rely=0.19, height=62, width=227)
        self.Button1.configure(activebackground="#ececec")#21b5ff
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")

        self.Button1.configure(font="-family {Al Bayan} -size 16")
        self.Button1.configure(foreground="blue")

        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(justify='left')
        self.Button1.configure(overrelief="flat")
        self.Button1.configure(relief="raised")
        self.Button1.configure(text='''Project Info''')

        self.Button6 = tk.Button(self.Frame1)
        self.Button6.place(relx=0.0, rely=0.762, height=62, width=227)
        self.Button6.configure(activebackground="#ececec")
        self.Button6.configure(activeforeground="#000000")
        self.Button6.configure(background="#d9d9d9")
        self.Button6.configure(font="-family {Al Bayan} -size 16")
        self.Button6.configure(foreground="blue")
        self.Button6.configure(highlightbackground="#d9d9d9")
        self.Button6.configure(highlightcolor="black")
        self.Button6.configure(justify='left')
        self.Button6.configure(relief="raised")
        self.Button6.configure(text='''Feature Selection''')

        self.Button7 = tk.Button(self.Frame1)
        self.Button7.place(relx=0.0, rely=0.876, height=62, width=227)
        self.Button7.configure(activebackground="#ececec")
        self.Button7.configure(activeforeground="#000000")
        self.Button7.configure(background="#d9d9d9")
        self.Button7.configure(font="-family {Al Bayan} -size 16")
        self.Button7.configure(foreground="blue")
        self.Button7.configure(highlightbackground="#d9d9d9")
        self.Button7.configure(highlightcolor="black")
        self.Button7.configure(justify='left')
        self.Button7.configure(relief="raised")
        self.Button7.configure(text='''Outcome''')
        # -----------------------------siedebar end-----------------------------#
        self.Frame2 = tk.Frame(top)
        self.Frame2.place(relx=0.258, rely=-0.01, relheight=1.016
                , relwidth=0.743)
        self.Frame2.configure(relief='groove')
        self.Frame2.configure(borderwidth="2")
        self.Frame2.configure(relief="groove")
        self.Frame2.configure(background="#a6ddf4")
        self.Frame2.configure(highlightbackground="#d9d9d9")
        self.Frame2.configure(highlightcolor="black")

        self.Label1 = tk.Label(self.Frame2)
        self.Label1.place(relx=0.13, rely=0.07, height=46, width=475)
        self.Label1.configure(activebackground="#a6ddf4")
        self.Label1.configure(activeforeground="#1c38ed")
        self.Label1.configure(background="#a6ddf4")
        self.Label1.configure(font="-family {Al Bayan} -size 20 -weight bold")
        self.Label1.configure(foreground="#171fff")
        self.Label1.configure(highlightbackground="#ffffffffffff")
        self.Label1.configure(highlightcolor="#2b1582")
        self.Label1.configure(text='''Welcome to Software Defect Prediction System''')

        self.Frame3 = tk.Frame(self.Frame2)
        self.Frame3.place(relx=0.0, rely=0.35, relheight=0.65, relwidth=1.0)
        self.Frame3.configure(relief='groove')
        self.Frame3.configure(borderwidth="2")
        self.Frame3.configure(relief="groove")
        self.Frame3.configure(background="#d9d9d9")

        self.Message1 = tk.Message(self.Frame2)
        self.Message1.place(relx=0.031, rely=0.19, relheight=0.11, relwidth=0.8)
        self.Message1.configure(background="#a6ddf4")
        self.Message1.configure(font="-family {Al Bayan} -size 16")
        self.Message1.configure(foreground="#000000")
        self.Message1.configure(highlightbackground="#d9d9d9")
        self.Message1.configure(highlightcolor="black")
        self.Message1.configure(anchor='nw')
        self.Message1.configure(text='''Discretization is done by transforming continuous data into discrete data ''')
        self.Message1.configure(width=500)

        # self.Label2 = tk.Label(self.Frame2)
        # self.Label2.place(relx=0.031, rely=0.214, height=31, width=226)
        # self.Label2.configure(background="#a6ddf4")
        # self.Label2.configure(font="-family {Al Bayan} -size 16")
        # self.Label2.configure(foreground="#000000")
        # self.Label2.configure(justify='left')
        # self.Label2.configure(text='''This is discretize dataset.''')

        self.Button1 = tk.Button(self.Frame2)
        self.Button1.place(relx=0.8, rely=0.204, height=31, width=120)
        self.Button1.configure(activebackground="#ececec")
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")
        self.Button1.configure(font="-family {Al Bayan} -size 16")
        self.Button1.configure(foreground="blue")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(relief="raised")
        self.Button1.configure(text='''Build Model''')
        self.Button1.configure(command=selectModel)

        TableMargin = Frame(self.Frame3, width=500)
        TableMargin.pack(side=TOP)
        scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
        scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
        tree = ttk.Treeview(TableMargin, columns=pre_dataset.columns, height=400, selectmode="extended", yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
        scrollbary.config(command=tree.yview)
        scrollbary.pack(side=RIGHT, fill=Y)
        scrollbarx.config(command=tree.xview)
        scrollbarx.pack(side=BOTTOM, fill=X)
        # print('dataset.columns', pre_dataset.shape[1])
      
        for col in range(pre_dataset.shape[1]):
            # print('col-->', col)
            # print('column name', pre_dataset.columns[col])
            tree.heading(col, text=pre_dataset.columns[col], anchor=W)
            tree.column('#'+str(col), stretch=NO, minwidth=0, width=100)

        tree.column('#0', stretch=NO, minwidth=0, width=0)
        # tree.column('#', stretch=NO, minwidth=0, width=100)

        tree.pack()

        for i in range(discretize_data.shape[0]):
            # print('val', discretize_data[i, :])
            tree.insert("", 0, values= discretize_data[i, :])

    def fun(self, top=None):
        global discretize_data
        print('feature data-->', feature_data)
        discretize_data = main_discretize(feature_data)
        print('\n')
        print ("*** Discretize Data ***")
        print (discretize_data)
        print(discretize_data.shape)

    def __init__(self, top=None):
        self.fun()
        self.ui()

class selectModel:
    def ui(self, top=None):
        # -----------------------------siedebar start-----------------------------#
        self.Frame1 = tk.Frame(top)
        self.Frame1.place(relx=0.0, rely=-0.02, relheight=1.036, relwidth=0.259)
        self.Frame1.configure(relief='groove')
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief="groove")
        self.Frame1.configure(background="#21b5ff")
        self.Frame1.configure(highlightbackground="#d9d9d9")
        self.Frame1.configure(highlightcolor="black")

        self.Label2 = tk.Label(self.Frame1)
        self.Label2.place(relx=0.178, rely=0.057, height=52, width=141)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(activeforeground="black")
        self.Label2.configure(background="#21b5ff")
        self.Label2.configure(font="-family {Al Bayan} -size 20 -weight bold")
        self.Label2.configure(foreground="white")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''SDP System''')

        self.Button2 = tk.Button(self.Frame1)
        self.Button2.place(relx=0.0, rely=0.419, height=62, width=227)
        self.Button2.configure(activebackground="#ececec")
        self.Button2.configure(activeforeground="#000000")
        self.Button2.configure(background="#d9d9d9")
        self.Button2.configure(font="-family {Al Bayan} -size 16")
        self.Button2.configure(foreground="blue")
        self.Button2.configure(highlightbackground="#d9d9d9")
        self.Button2.configure(highlightcolor="black")
        self.Button2.configure(justify='left')
        self.Button2.configure(relief="raised")
        self.Button2.configure(text='''Preprocessing''')

        self.Button3 = tk.Button(self.Frame1)
        self.Button3.place(relx=0.0, rely=0.305, height=62, width=227)
        self.Button3.configure(activebackground="#ececec")
        self.Button3.configure(activeforeground="#000000")
        self.Button3.configure(background="#d9d9d9")
        self.Button3.configure(font="-family {Al Bayan} -size 16")
        self.Button3.configure(foreground="blue")
        self.Button3.configure(highlightbackground="#d9d9d9")
        self.Button3.configure(highlightcolor="black")
        self.Button3.configure(justify='left')
        self.Button3.configure(relief="raised")
        self.Button3.configure(text='''Dataset Selection''')

        self.Button4 = tk.Button(self.Frame1)
        self.Button4.place(relx=0.0, rely=0.533, height=62, width=227)
        self.Button4.configure(activebackground="#ececec")
        self.Button4.configure(activeforeground="#000000")
        self.Button4.configure(background="#d9d9d9")
        self.Button4.configure(font="-family {Al Bayan} -size 16")
        self.Button4.configure(foreground="blue")
        self.Button4.configure(highlightbackground="#d9d9d9")
        self.Button4.configure(highlightcolor="black")
        self.Button4.configure(justify='left')
        self.Button4.configure(relief="raised")
        self.Button4.configure(text='''Discretization''')

        self.Button5 = tk.Button(self.Frame1)
        self.Button5.place(relx=0.0, rely=0.648, height=62, width=227)
        self.Button5.configure(activebackground="#ececec")
        self.Button5.configure(activeforeground="#000000")
        self.Button5.configure(background="#d9d9d9")
        self.Button5.configure(font="-family {Al Bayan} -size 16")
        self.Button5.configure(foreground="green")
        self.Button5.configure(highlightbackground="#d9d9d9")
        self.Button5.configure(highlightcolor="black")
        self.Button5.configure(justify='left')
        self.Button5.configure(relief="raised")
        self.Button5.configure(text='''*** Model Selection ***''')

        self.Button1 = tk.Button(self.Frame1)
        self.Button1.place(relx=0.0, rely=0.19, height=62, width=227)
        self.Button1.configure(activebackground="#ececec")#21b5ff
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")
        self.Button1.configure(font="-family {Al Bayan} -size 16")
        self.Button1.configure(foreground="blue")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(justify='left')
        self.Button1.configure(overrelief="flat")
        self.Button1.configure(relief="raised")
        self.Button1.configure(text='''Project Info''')

        self.Button6 = tk.Button(self.Frame1)
        self.Button6.place(relx=0.0, rely=0.762, height=62, width=227)
        self.Button6.configure(activebackground="#ececec")
        self.Button6.configure(activeforeground="#000000")
        self.Button6.configure(background="#d9d9d9")
        self.Button6.configure(font="-family {Al Bayan} -size 16")
        self.Button6.configure(foreground="blue")
        self.Button6.configure(highlightbackground="#d9d9d9")
        self.Button6.configure(highlightcolor="black")
        self.Button6.configure(justify='left')
        self.Button6.configure(relief="raised")
        self.Button6.configure(text='''Feature Selection''')

        self.Button7 = tk.Button(self.Frame1)
        self.Button7.place(relx=0.0, rely=0.876, height=62, width=227)
        self.Button7.configure(activebackground="#ececec")
        self.Button7.configure(activeforeground="#000000")
        self.Button7.configure(background="#d9d9d9")
        self.Button7.configure(font="-family {Al Bayan} -size 16")
        self.Button7.configure(foreground="blue")
        self.Button7.configure(highlightbackground="#d9d9d9")
        self.Button7.configure(highlightcolor="black")
        self.Button7.configure(justify='left')
        self.Button7.configure(relief="raised")
        self.Button7.configure(text='''Outcome''')
        # -----------------------------siedebar end-----------------------------#
        self.Frame2 = tk.Frame(top)
        self.Frame2.place(relx=0.258, rely=-0.01, relheight=1.016
                , relwidth=0.743)
        self.Frame2.configure(relief='groove')
        self.Frame2.configure(borderwidth="2")
        self.Frame2.configure(relief="groove")
        self.Frame2.configure(background="#a6ddf4")
        self.Frame2.configure(highlightbackground="#d9d9d9")
        self.Frame2.configure(highlightcolor="black")

        self.Label1 = tk.Label(self.Frame2)
        self.Label1.place(relx=0.13, rely=0.07, height=46, width=475)
        self.Label1.configure(activebackground="#a6ddf4")
        self.Label1.configure(activeforeground="#1c38ed")
        self.Label1.configure(background="#a6ddf4")
        self.Label1.configure(font="-family {Al Bayan} -size 20 -weight bold")
        self.Label1.configure(foreground="#171fff")
        self.Label1.configure(highlightbackground="#ffffffffffff")
        self.Label1.configure(highlightcolor="#2b1582")
        self.Label1.configure(text='''Welcome to Software Defect Prediction System''')

        self.Frame3 = tk.Frame(self.Frame2)
        self.Frame3.place(relx=0.0, rely=0.35, relheight=0.65, relwidth=1.0)
        self.Frame3.configure(relief='groove')
        self.Frame3.configure(borderwidth="2")
        self.Frame3.configure(relief="groove")
        self.Frame3.configure(background="#a6ddf4")

        self.Message1 = tk.Message(self.Frame3)
        self.Message1.place(relx=0.031, rely=0.03, relheight=0.937
                , relwidth=0.936)
        self.Message1.configure(background="#a6ddf4")
        self.Message1.configure(font='-family {Al Bayan} -size 16')
        self.Message1.configure(foreground="blue")
        self.Message1.configure(highlightbackground="#d9d9d9")
        self.Message1.configure(highlightcolor="#a6ddf4")
        self.Message1.configure(text='''SVM Method
-   Predict data using Support Vector Machine with Radial Basis Function (RBF)

Hybrid Approach
-   Select 10 features using Minimum-Redundancy-Maximum-Relevance (MRMR) method
-   Build model using Adaptive Boosting method based on RBF_SVM

Comparison Chart
-   Implement these 2 methods and show comparison chart for Accuracy, Precision, Recall and F-score measures between these methods''')
        self.Message1.configure(width=604)

        self.Label1 = tk.Label(self.Frame2)
        self.Label1.place(relx=0.031, rely=0.214, height=31, width=126)
        self.Label1.configure(background="#a6ddf4")
        self.Label1.configure(font="-family {Al Bayan} -size 16")
        self.Label1.configure(foreground="#000000")
        self.Label1.configure(text='''Select Model''')

        self.TCombobox2 = ttk.Combobox(self.Frame2)
        self.TCombobox2.place(relx=0.264, rely=0.214, relheight=0.052
                , relwidth=0.319)
        # self.TCombobox1.configure(textvariable=interface_support.combobox)
        self.TCombobox2["values"]=("SVM", "Hybrid Approach", "Compare both methods")
        self.TCombobox2.configure(takefocus="")

        self.Button8 = tk.Button(self.Frame2)
        self.Button8.place(relx=0.62, rely=0.204, height=31, width=113)
        self.Button8.configure(activebackground="#ececec")
        self.Button8.configure(activeforeground="#000000")
        self.Button8.configure(background="#d9d9d9")
        self.Button8.configure(font="-family {Al Bayan} -size 16")
        self.Button8.configure(foreground="blue")
        self.Button8.configure(highlightbackground="#d9d9d9")
        self.Button8.configure(highlightcolor="black")
        self.Button8.configure(relief="raised")
        self.Button8.configure(text='''Predict''')
        self.Button8.configure(command=self.fun)

    def fun(self, top=None):
        global modelctrl
        print('self.TCombobox2.get()', self.TCombobox2.get())
        if self.TCombobox2.get() == "SVM":
            modelctrl = 0
            outcome_ui = outcome()
        if self.TCombobox2.get() == "Hybrid Approach":
            modelctrl = 1
            show_feature = showFeature()
        if self.TCombobox2.get() == "Compare both methods":
            modelctrl = 2
            outcome_ui = outcome()

    def __init__(self, top=None):
        self.ui()
        # self.fun()      
      
class showFeature:
    def ui(self, top=None):
        # -----------------------------siedebar start-----------------------------#
        self.Frame1 = tk.Frame(top)
        self.Frame1.place(relx=0.0, rely=-0.02, relheight=1.036, relwidth=0.259)
        self.Frame1.configure(relief='groove')
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief="groove")
        self.Frame1.configure(background="#21b5ff")
        self.Frame1.configure(highlightbackground="#d9d9d9")
        self.Frame1.configure(highlightcolor="black")

        self.Label2 = tk.Label(self.Frame1)
        self.Label2.place(relx=0.178, rely=0.057, height=52, width=141)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(activeforeground="black")
        self.Label2.configure(background="#21b5ff")
        self.Label2.configure(font="-family {Al Bayan} -size 20 -weight bold")
        self.Label2.configure(foreground="white")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''SDP System''')

        self.Button2 = tk.Button(self.Frame1)
        self.Button2.place(relx=0.0, rely=0.419, height=62, width=227)
        self.Button2.configure(activebackground="#ececec")
        self.Button2.configure(activeforeground="#000000")
        self.Button2.configure(background="#d9d9d9")
        self.Button2.configure(font="-family {Al Bayan} -size 16")
        self.Button2.configure(foreground="blue")
        self.Button2.configure(highlightbackground="#d9d9d9")
        self.Button2.configure(highlightcolor="black")
        self.Button2.configure(justify='left')
        self.Button2.configure(relief="raised")
        self.Button2.configure(text='''Preprocessing''')

        self.Button3 = tk.Button(self.Frame1)
        self.Button3.place(relx=0.0, rely=0.305, height=62, width=227)
        self.Button3.configure(activebackground="#ececec")
        self.Button3.configure(activeforeground="#000000")
        self.Button3.configure(background="#d9d9d9")
        self.Button3.configure(font="-family {Al Bayan} -size 16")
        self.Button3.configure(foreground="blue")
        self.Button3.configure(highlightbackground="#d9d9d9")
        self.Button3.configure(highlightcolor="black")
        self.Button3.configure(justify='left')
        self.Button3.configure(relief="raised")
        self.Button3.configure(text='''Dataset Selection''')

        self.Button4 = tk.Button(self.Frame1)
        self.Button4.place(relx=0.0, rely=0.533, height=62, width=227)
        self.Button4.configure(activebackground="#ececec")
        self.Button4.configure(activeforeground="#000000")
        self.Button4.configure(background="#d9d9d9")
        self.Button4.configure(font="-family {Al Bayan} -size 16")
        self.Button4.configure(foreground="blue")
        self.Button4.configure(highlightbackground="#d9d9d9")
        self.Button4.configure(highlightcolor="black")
        self.Button4.configure(justify='left')
        self.Button4.configure(relief="raised")
        self.Button4.configure(text='''Discretization''')

        self.Button5 = tk.Button(self.Frame1)
        self.Button5.place(relx=0.0, rely=0.648, height=62, width=227)
        self.Button5.configure(activebackground="#ececec")
        self.Button5.configure(activeforeground="#000000")
        self.Button5.configure(background="#d9d9d9")
        self.Button5.configure(font="-family {Al Bayan} -size 16")
        self.Button5.configure(foreground="blue")
        self.Button5.configure(highlightbackground="#d9d9d9")
        self.Button5.configure(highlightcolor="black")
        self.Button5.configure(justify='left')
        self.Button5.configure(relief="raised")
        self.Button5.configure(text='''Model Selection''')

        self.Button1 = tk.Button(self.Frame1)
        self.Button1.place(relx=0.0, rely=0.19, height=62, width=227)
        self.Button1.configure(activebackground="#ececec")#21b5ff
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")

        self.Button1.configure(font="-family {Al Bayan} -size 16")
        self.Button1.configure(foreground="blue")

        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(justify='left')
        self.Button1.configure(overrelief="flat")
        self.Button1.configure(relief="raised")
        self.Button1.configure(text='''Project Info''')

        self.Button6 = tk.Button(self.Frame1)
        self.Button6.place(relx=0.0, rely=0.762, height=62, width=227)
        self.Button6.configure(activebackground="#ececec")
        self.Button6.configure(activeforeground="#000000")
        self.Button6.configure(background="#d9d9d9")
        self.Button6.configure(font="-family {Al Bayan} -size 16")
        self.Button6.configure(foreground="green")
        self.Button6.configure(highlightbackground="#d9d9d9")
        self.Button6.configure(highlightcolor="black")
        self.Button6.configure(justify='left')
        self.Button6.configure(relief="raised")
        self.Button6.configure(text='''*** Feature Selection ***''')

        self.Button7 = tk.Button(self.Frame1)
        self.Button7.place(relx=0.0, rely=0.876, height=62, width=227)
        self.Button7.configure(activebackground="#ececec")
        self.Button7.configure(activeforeground="#000000")
        self.Button7.configure(background="#d9d9d9")
        self.Button7.configure(font="-family {Al Bayan} -size 16")
        self.Button7.configure(foreground="blue")
        self.Button7.configure(highlightbackground="#d9d9d9")
        self.Button7.configure(highlightcolor="black")
        self.Button7.configure(justify='left')
        self.Button7.configure(relief="raised")
        self.Button7.configure(text='''Outcome''')
        # -----------------------------siedebar end-----------------------------#

        self.Frame2 = tk.Frame(top)
        self.Frame2.place(relx=0.258, rely=-0.01, relheight=1.016
                , relwidth=0.743)
        self.Frame2.configure(relief='groove')
        self.Frame2.configure(borderwidth="2")
        self.Frame2.configure(relief="groove")
        self.Frame2.configure(background="#a6ddf4")
        self.Frame2.configure(highlightbackground="#d9d9d9")
        self.Frame2.configure(highlightcolor="black")

        self.Label1 = tk.Label(self.Frame2)
        self.Label1.place(relx=0.13, rely=0.07, height=46, width=475)
        self.Label1.configure(activebackground="#a6ddf4")
        self.Label1.configure(activeforeground="#1c38ed")
        self.Label1.configure(background="#a6ddf4")
        self.Label1.configure(font="-family {Al Bayan} -size 20 -weight bold")
        self.Label1.configure(foreground="#171fff")
        self.Label1.configure(highlightbackground="#ffffffffffff")
        self.Label1.configure(highlightcolor="#2b1582")
        self.Label1.configure(text='''Welcome to Software Defect Prediction System''')
        
        self.Label3 = tk.Label(self.Frame2)
        self.Label3.place(relx=0.031, rely=0.214, height=31, width=216)
        self.Label3.configure(activebackground="#f9f9f9")
        self.Label3.configure(activeforeground="black")
        self.Label3.configure(background="#a6ddf4")
        self.Label3.configure(font="-family {Al Bayan} -size 16")
        self.Label3.configure(foreground="#000000")
        self.Label3.configure(highlightbackground="#d9d9d9")
        self.Label3.configure(highlightcolor="black")
        self.Label3.configure(justify='left')
        self.Label3.configure(text='''Selected features list are :''')

        self.Button9 = tk.Button(self.Frame2)
        self.Button9.place(relx=0.822, rely=0.214, height=31, width=87)
        self.Button9.configure(activebackground="#ececec")
        self.Button9.configure(activeforeground="#000000")
        self.Button9.configure(background="#d9d9d9")
        self.Button9.configure(font="-family {Al Bayan} -size 16")
        self.Button9.configure(foreground="blue")
        self.Button9.configure(highlightbackground="#d9d9d9")
        self.Button9.configure(highlightcolor="black")
        self.Button9.configure(relief="raised")
        self.Button9.configure(text='''Next Step''')
        self.Button9.configure(command=outcome)

        self.Message1 = tk.Message(self.Frame2)
        self.Message1.place(relx=0.005, rely=0.311, relheight=0.105, relwidth=0.5)
        self.Message1.configure(background="#a6ddf4")
        self.Message1.configure(font="-family {Al Bayan} -size 16")
        self.Message1.configure(foreground="#000000")
        self.Message1.configure(highlightbackground="#d9d9d9")
        self.Message1.configure(highlightcolor="black")
        self.Message1.configure(anchor='sw')
        f1 = "1."+ pre_dataset.columns[feature_extraction[0][0]]
        self.Message1.configure(text=f1)
        self.Message1.configure(width=500)

        # self.Label4 = tk.Label(self.Frame2, justify='left')
        # self.Label4.place(relx=0.005, rely=0.311, height=31, width=350)
        # self.Label4.configure(background="blue")
        # self.Label4.configure(font="-family {Al Bayan} -size 16")
        # self.Label4.configure(foreground="#000000")
        # # self.Label4.configure(justify=left)
        # f1 = "1. "+ pre_dataset.columns[feature_extraction[0][0]]
        # self.Label4.configure(text=f1)

        self.Message2 = tk.Message(self.Frame2)
        self.Message2.place(relx=0.005, rely=0.447, relheight=0.105, relwidth=0.5)
        self.Message2.configure(background="#a6ddf4")
        self.Message2.configure(font="-family {Al Bayan} -size 16")
        self.Message2.configure(foreground="#000000")
        self.Message2.configure(highlightbackground="#d9d9d9")
        self.Message2.configure(highlightcolor="black")
        self.Message2.configure(anchor='sw')
        f2 = "2. "+ pre_dataset.columns[feature_extraction[0][1]]
        self.Message2.configure(text=f2)
        self.Message2.configure(width=500)

        # self.Label5 = tk.Label(self.Frame2)
        # self.Label5.place(relx=0.005, rely=0.447, height=31, width=350)
        # self.Label5.configure(background="#a6ddf4")
        # self.Label5.configure(font="-family {Al Bayan} -size 16")
        # self.Label5.configure(foreground="#000000")
        # self.Label5.configure(justify='left')
        # f2 = "2. "+ pre_dataset.columns[feature_extraction[0][1]]
        # self.Label5.configure(text=f2)

        self.Message3 = tk.Message(self.Frame2)
        self.Message3.place(relx=0.005, rely=0.583, relheight=0.105, relwidth=0.5)
        self.Message3.configure(background="#a6ddf4")
        self.Message3.configure(font="-family {Al Bayan} -size 16")
        self.Message3.configure(foreground="#000000")
        self.Message3.configure(highlightbackground="#d9d9d9")
        self.Message3.configure(highlightcolor="black")
        self.Message3.configure(anchor='sw')
        f3 = "3. "+ pre_dataset.columns[feature_extraction[0][2]]
        self.Message3.configure(text=f3)
        self.Message3.configure(width=500)

        # self.Label6 = tk.Label(self.Frame2)
        # self.Label6.place(relx=0.005, rely=0.583, height=31, width=350)
        # self.Label6.configure(background="#a6ddf4")
        # self.Label6.configure(font="-family {Al Bayan} -size 16")
        # self.Label6.configure(foreground="#000000")
        # self.Label6.configure(justify='left')
        # f3 = "3. "+ pre_dataset.columns[feature_extraction[0][2]]
        # self.Label6.configure(text=f3)

        self.Message4 = tk.Message(self.Frame2)
        self.Message4.place(relx=0.005, rely=0.718, relheight=0.105, relwidth=0.5)
        self.Message4.configure(background="#a6ddf4")
        self.Message4.configure(font="-family {Al Bayan} -size 16")
        self.Message4.configure(foreground="#000000")
        self.Message4.configure(highlightbackground="#d9d9d9")
        self.Message4.configure(highlightcolor="black")
        self.Message4.configure(anchor='sw')
        f4 = "4. "+ pre_dataset.columns[feature_extraction[0][3]]
        self.Message4.configure(text=f4)
        self.Message4.configure(width=500)

        # self.Label7 = tk.Label(self.Frame2)
        # self.Label7.place(relx=0.005, rely=0.718, height=31, width=350)
        # self.Label7.configure(background="#a6ddf4")
        # self.Label7.configure(font="-family {Al Bayan} -size 16")
        # self.Label7.configure(foreground="#000000")
        # self.Label7.configure(justify='left')
        # f4 = "4. "+ pre_dataset.columns[feature_extraction[0][3]]
        # self.Label7.configure(text=f4)

        self.Message5 = tk.Message(self.Frame2)
        self.Message5.place(relx=0.005, rely=0.854, relheight=0.105, relwidth=0.5)
        self.Message5.configure(background="#a6ddf4")
        self.Message5.configure(font="-family {Al Bayan} -size 16")
        self.Message5.configure(foreground="#000000")
        self.Message5.configure(highlightbackground="#d9d9d9")
        self.Message5.configure(highlightcolor="black")
        self.Message5.configure(anchor='sw')
        f5 = "5. "+ pre_dataset.columns[feature_extraction[0][4]]
        self.Message5.configure(text=f5)
        self.Message5.configure(width=500)

        # self.Label8 = tk.Label(self.Frame2)
        # self.Label8.place(relx=0.005, rely=0.854, height=31, width=350)
        # self.Label8.configure(background="#a6ddf4")
        # self.Label8.configure(font="-family {Al Bayan} -size 16")
        # self.Label8.configure(foreground="#000000")
        # self.Label8.configure(justify='left')
        # f5 = "5. "+ pre_dataset.columns[feature_extraction[0][4]]
        # self.Label8.configure(text=f5)

        self.Message6 = tk.Message(self.Frame2)
        self.Message6.place(relx=0.5, rely=0.311, relheight=0.105, relwidth=0.5)
        self.Message6.configure(background="#a6ddf4")
        self.Message6.configure(font="-family {Al Bayan} -size 16")
        self.Message6.configure(foreground="#000000")
        self.Message6.configure(highlightbackground="#d9d9d9")
        self.Message6.configure(highlightcolor="black")
        self.Message6.configure(anchor='sw')
        f6 = "6. "+ pre_dataset.columns[feature_extraction[0][5]]
        self.Message6.configure(text=f6)
        self.Message6.configure(width=500)

        # self.Label9 = tk.Label(self.Frame2)
        # self.Label9.place(relx=0.558, rely=0.311, height=31, width=350)
        # self.Label9.configure(background="#a6ddf4")
        # self.Label9.configure(font="-family {Al Bayan} -size 16")
        # self.Label9.configure(foreground="#000000")
        # self.Label9.configure(justify='left')
        # f6 = "6. "+ pre_dataset.columns[feature_extraction[0][5]]
        # self.Label9.configure(text=f6)

        self.Message7 = tk.Message(self.Frame2)
        self.Message7.place(relx=0.5, rely=0.447, relheight=0.105, relwidth=0.5)
        self.Message7.configure(background="#a6ddf4")
        self.Message7.configure(font="-family {Al Bayan} -size 16")
        self.Message7.configure(foreground="#000000")
        self.Message7.configure(highlightbackground="#d9d9d9")
        self.Message7.configure(highlightcolor="black")
        self.Message7.configure(anchor='sw')
        f7 = "7. "+ pre_dataset.columns[feature_extraction[0][6]]
        self.Message7.configure(text=f7)
        self.Message7.configure(width=500)

        # self.Label10 = tk.Label(self.Frame2)
        # self.Label10.place(relx=0.558, rely=0.447, height=31, width=350)
        # self.Label10.configure(background="#a6ddf4")
        # self.Label10.configure(font="-family {Al Bayan} -size 16")
        # self.Label10.configure(foreground="#000000")
        # self.Label10.configure(justify='left')
        # f7 = "7. "+ pre_dataset.columns[feature_extraction[0][6]]
        # self.Label10.configure(text=f7)

        self.Message8 = tk.Message(self.Frame2)
        self.Message8.place(relx=0.5, rely=0.583, relheight=0.105, relwidth=0.5)
        self.Message8.configure(background="#a6ddf4")
        self.Message8.configure(font="-family {Al Bayan} -size 16")
        self.Message8.configure(foreground="#000000")
        self.Message8.configure(highlightbackground="#d9d9d9")
        self.Message8.configure(highlightcolor="black")
        self.Message8.configure(anchor='sw')
        f8 = "8. "+ pre_dataset.columns[feature_extraction[0][7]]
        self.Message8.configure(text=f8)
        self.Message8.configure(width=500)

        # self.Label11 = tk.Label(self.Frame2)
        # self.Label11.place(relx=0.558, rely=0.583, height=31, width=350)
        # self.Label11.configure(background="#a6ddf4")
        # self.Label11.configure(font="-family {Al Bayan} -size 16")
        # self.Label11.configure(foreground="#000000")
        # self.Label11.configure(justify='left')
        # f8 = "8. "+ pre_dataset.columns[feature_extraction[0][7]]
        # self.Label11.configure(text=f8)

        self.Message9 = tk.Message(self.Frame2)
        self.Message9.place(relx=0.5, rely=0.718, relheight=0.105, relwidth=0.5)
        self.Message9.configure(background="#a6ddf4")
        self.Message9.configure(font="-family {Al Bayan} -size 16")
        self.Message9.configure(foreground="#000000")
        self.Message9.configure(highlightbackground="#d9d9d9")
        self.Message9.configure(highlightcolor="black")
        self.Message9.configure(anchor='sw')
        f9 = "9. "+ pre_dataset.columns[feature_extraction[0][8]]
        self.Message9.configure(text=f9)
        self.Message9.configure(width=500)


        # self.Label12 = tk.Label(self.Frame2)
        # self.Label12.place(relx=0.558, rely=0.718, height=31, width=350)
        # self.Label12.configure(background="#a6ddf4")
        # self.Label12.configure(cursor="fleur")
        # self.Label12.configure(font="-family {Al Bayan} -size 16")
        # self.Label12.configure(foreground="#000000")
        # self.Label12.configure(justify='left')
        # f9 = "9. "+ pre_dataset.columns[feature_extraction[0][8]]
        # self.Label12.configure(text=f9)

        self.Message10 = tk.Message(self.Frame2)
        self.Message10.place(relx=0.5, rely=0.854, relheight=0.105, relwidth=0.5)
        self.Message10.configure(background="#a6ddf4")
        self.Message10.configure(font="-family {Al Bayan} -size 16")
        self.Message10.configure(foreground="#000000")
        self.Message10.configure(highlightbackground="#d9d9d9")
        self.Message10.configure(highlightcolor="black")
        self.Message10.configure(anchor='sw')
        f10 = "10. "+ pre_dataset.columns[feature_extraction[0][9]]
        self.Message10.configure(text=f10)
        self.Message10.configure(width=500)

        # self.Label13 = tk.Label(self.Frame2)
        # self.Label13.place(relx=0.543, rely=0.854, height=31, width=350)
        # self.Label13.configure(background="#a6ddf4")
        # self.Label13.configure(font="-family {Al Bayan} -size 16")
        # self.Label13.configure(foreground="#000000")
        # self.Label13.configure(justify='left')
        # f10 = "10. "+ pre_dataset.columns[feature_extraction[0][9]]
        # self.Label13.configure(text=f10)
    
    def fun(self, top=None):
        global selected_data
        global feature_extraction
        num_fea = 10
        feature_extraction = feature_extract(discretize_data, target_data, num_fea)
        print('\n')
        print("*** Selected Feature ***")
        print(feature_extraction)

        selected_data = discretize_data[:, feature_extraction[0]]

        # if datactrl == 1:
        #     selected_data = discretize_data[:, [36, 29, 16,  3, 32,  9, 19,  4, 20, 22]]  #=> transform manual to auto
        # if datactrl == 2:
        #     selected_data = discretize_data[:, [3, 28, 14, 13,  8,  4, 21, 25, 16, 15]]  #=> transform manual to auto
        # if datactrl == 3:
        #     selected_data = discretize_data[:, [0, 16, 9,  8, 17, 35,  2, 22, 20, 19]]  #=> transform manual to auto
        # if datactrl == 4:
        #     selected_data = discretize_data[:, [35, 25,  3, 16, 17,  9,  7, 32, 10, 29]]  #=> transform manual to auto
        # if datactrl == 5:
        #     selected_data = discretize_data[:, [34, 18,  1,  3,  2,  7, 19, 15, 26, 12]]  #=> transform manual to auto

    def __init__(self, top=None):
        self.fun()
        self.ui()

class outcome:
    def ui(self, top=None):
        # -----------------------------siedebar start-----------------------------#
        self.Frame1 = tk.Frame(top)
        self.Frame1.place(relx=0.0, rely=-0.02, relheight=1.036, relwidth=0.259)
        self.Frame1.configure(relief='groove')
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief="groove")
        self.Frame1.configure(background="#21b5ff")
        self.Frame1.configure(highlightbackground="#d9d9d9")
        self.Frame1.configure(highlightcolor="black")

        self.Label2 = tk.Label(self.Frame1)
        self.Label2.place(relx=0.178, rely=0.057, height=52, width=141)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(activeforeground="black")
        self.Label2.configure(background="#21b5ff")
        self.Label2.configure(font="-family {Al Bayan} -size 20 -weight bold")
        self.Label2.configure(foreground="white")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''SDP System''')

        self.Button2 = tk.Button(self.Frame1)
        self.Button2.place(relx=0.0, rely=0.419, height=62, width=227)
        self.Button2.configure(activebackground="#ececec")
        self.Button2.configure(activeforeground="#000000")
        self.Button2.configure(background="#d9d9d9")
        self.Button2.configure(font="-family {Al Bayan} -size 16")
        self.Button2.configure(foreground="blue")
        self.Button2.configure(highlightbackground="#d9d9d9")
        self.Button2.configure(highlightcolor="black")
        self.Button2.configure(justify='left')
        self.Button2.configure(relief="raised")
        self.Button2.configure(text='''Preprocessing''')

        self.Button3 = tk.Button(self.Frame1)
        self.Button3.place(relx=0.0, rely=0.305, height=62, width=227)
        self.Button3.configure(activebackground="#ececec")
        self.Button3.configure(activeforeground="#000000")
        self.Button3.configure(background="#d9d9d9")
        self.Button3.configure(font="-family {Al Bayan} -size 16")
        self.Button3.configure(foreground="blue")
        self.Button3.configure(highlightbackground="#d9d9d9")
        self.Button3.configure(highlightcolor="black")
        self.Button3.configure(justify='left')
        self.Button3.configure(relief="raised")
        self.Button3.configure(text='''Dataset Selection''')

        self.Button4 = tk.Button(self.Frame1)
        self.Button4.place(relx=0.0, rely=0.533, height=62, width=227)
        self.Button4.configure(activebackground="#ececec")
        self.Button4.configure(activeforeground="#000000")
        self.Button4.configure(background="#d9d9d9")
        self.Button4.configure(font="-family {Al Bayan} -size 16")
        self.Button4.configure(foreground="blue")
        self.Button4.configure(highlightbackground="#d9d9d9")
        self.Button4.configure(highlightcolor="black")
        self.Button4.configure(justify='left')
        self.Button4.configure(relief="raised")
        self.Button4.configure(text='''Discretization''')

        self.Button5 = tk.Button(self.Frame1)
        self.Button5.place(relx=0.0, rely=0.648, height=62, width=227)
        self.Button5.configure(activebackground="#ececec")
        self.Button5.configure(activeforeground="#000000")
        self.Button5.configure(background="#d9d9d9")
        self.Button5.configure(font="-family {Al Bayan} -size 16")
        self.Button5.configure(foreground="blue")
        self.Button5.configure(highlightbackground="#d9d9d9")
        self.Button5.configure(highlightcolor="black")
        self.Button5.configure(justify='left')
        self.Button5.configure(relief="raised")
        self.Button5.configure(text='''Model Selection''')

        self.Button1 = tk.Button(self.Frame1)
        self.Button1.place(relx=0.0, rely=0.19, height=62, width=227)
        self.Button1.configure(activebackground="#ececec")#21b5ff
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")

        self.Button1.configure(font="-family {Al Bayan} -size 16")
        self.Button1.configure(foreground="blue")

        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(justify='left')
        self.Button1.configure(overrelief="flat")
        self.Button1.configure(relief="raised")
        self.Button1.configure(text='''Project Info''')

        self.Button6 = tk.Button(self.Frame1)
        self.Button6.place(relx=0.0, rely=0.762, height=62, width=227)
        self.Button6.configure(activebackground="#ececec")
        self.Button6.configure(activeforeground="#000000")
        self.Button6.configure(background="#d9d9d9")
        self.Button6.configure(font="-family {Al Bayan} -size 16")
        self.Button6.configure(foreground="blue")
        self.Button6.configure(highlightbackground="#d9d9d9")
        self.Button6.configure(highlightcolor="black")
        self.Button6.configure(justify='left')
        self.Button6.configure(relief="raised")
        self.Button6.configure(text='''Feature Selection''')

        self.Button7 = tk.Button(self.Frame1)
        self.Button7.place(relx=0.0, rely=0.876, height=62, width=227)
        self.Button7.configure(activebackground="#ececec")
        self.Button7.configure(activeforeground="#000000")
        self.Button7.configure(background="#d9d9d9")
        self.Button7.configure(font="-family {Al Bayan} -size 16")
        self.Button7.configure(foreground="green")
        self.Button7.configure(highlightbackground="#d9d9d9")
        self.Button7.configure(highlightcolor="black")
        self.Button7.configure(justify='left')
        self.Button7.configure(relief="raised")
        self.Button7.configure(text='''*** Outcome ***''')
        # -----------------------------siedebar end-----------------------------#

        self.Frame2 = tk.Frame(top)
        self.Frame2.place(relx=0.258, rely=-0.01, relheight=1.016
                , relwidth=0.743)
        self.Frame2.configure(relief='groove')
        self.Frame2.configure(borderwidth="2")
        self.Frame2.configure(relief="groove")
        self.Frame2.configure(background="#a6ddf4")
        self.Frame2.configure(highlightbackground="#d9d9d9")
        self.Frame2.configure(highlightcolor="black")

        self.Label1 = tk.Label(self.Frame2)
        self.Label1.place(relx=0.13, rely=0.07, height=46, width=475)
        self.Label1.configure(activebackground="#a6ddf4")
        self.Label1.configure(activeforeground="#1c38ed")
        self.Label1.configure(background="#a6ddf4")
        self.Label1.configure(font="-family {Al Bayan} -size 20 -weight bold")
        self.Label1.configure(foreground="#171fff")
        self.Label1.configure(highlightbackground="#ffffffffffff")
        self.Label1.configure(highlightcolor="#2b1582")
        self.Label1.configure(text='''Welcome to Software Defect Prediction System''')

        self.Label3 = tk.Label(self.Frame2)
        self.Label3.configure(activebackground="#f9f9f9")
        self.Label3.configure(activeforeground="black")
        self.Label3.configure(background="#a6ddf4")
        self.Label3.configure(cursor="fleur")
        self.Label3.configure(font="-family {Al Bayan} -size 16")
        self.Label3.configure(foreground="#000000")
        self.Label3.configure(highlightbackground="#d9d9d9")
        self.Label3.configure(highlightcolor="black")
        self.Label3.configure(justify='left')
        if datactrl == 1:
            tmp = "PC1 Dataset"
        if datactrl == 2:
            tmp = "PC2 Dataset"
        if datactrl == 3:
            tmp = "PC3 Dataset"
        if datactrl == 4:
            tmp = "PC4 Dataset"
        if datactrl == 5:
            tmp = "PC5 Dataset"
        if modelctrl == 0:
            modeltmp = tmp + " using SVM"
            self.Label3.place(relx=0.15, rely=0.214, height=31, width=416)

        else:
            modeltmp = tmp + " using AdaBoost SVM-RBF with MRMR"
            self.Label3.place(relx=0.18, rely=0.214, height=31, width=416)

            
        self.Label3.configure(text=modeltmp)

        self.Label4 = tk.Label(self.Frame2)
        self.Label4.place(relx=0.023, rely=0.311, height=31, width=85)
        self.Label4.configure(background="#a6ddf4")
        self.Label4.configure(font="-family {Al Bayan} -size 16")
        self.Label4.configure(foreground="#000000")
        self.Label4.configure(text='''Accuracy :''')

        self.Label9 = tk.Label(self.Frame2)
        self.Label9.place(relx=0.215, rely=0.311, height=31, width=100)
        self.Label9.configure(background="#a6ddf4")
        self.Label9.configure(font="-family {Al Bayan} -size 16")
        self.Label9.configure(foreground="#000000")
        self.Label9.configure(justify='left')
        self.Label9.configure(text=float("{0:.2f}".format(accuracy)))

        self.Label5 = tk.Label(self.Frame2)
        self.Label5.place(relx=0.016, rely=0.408, height=31, width=96)
        self.Label5.configure(background="#a6ddf4")
        self.Label5.configure(font="-family {Al Bayan} -size 16")
        self.Label5.configure(foreground="#000000")
        self.Label5.configure(justify='left')
        self.Label5.configure(text='''Precision :''')
        

        self.Label10 = tk.Label(self.Frame2)
        self.Label10.place(relx=0.215, rely=0.408, height=31, width=100)
        self.Label10.configure(background="#a6ddf4")
        self.Label10.configure(font="-family {Al Bayan} -size 16")
        self.Label10.configure(foreground="#000000")
        self.Label10.configure(justify='left')
        self.Label10.configure(text=float("{0:.2f}".format(precision)))

        self.Label6 = tk.Label(self.Frame2)
        self.Label6.place(relx=0.026, rely=0.505, height=31, width=61)
        self.Label6.configure(background="#a6ddf4")
        self.Label6.configure(font="-family {Al Bayan} -size 16")
        self.Label6.configure(foreground="#000000")
        self.Label6.configure(text='''Recall :''')
        
        self.Label11 = tk.Label(self.Frame2)
        self.Label11.place(relx=0.215, rely=0.505, height=31, width=100)
        self.Label11.configure(background="#a6ddf4")
        self.Label11.configure(font="-family {Al Bayan} -size 16")
        self.Label11.configure(foreground="#000000")
        self.Label11.configure(justify='left')
        self.Label11.configure(text=float("{0:.2f}".format(recall)))

        self.Label7 = tk.Label(self.Frame2)
        self.Label7.place(relx=0.028, rely=0.602, height=31, width=73)
        self.Label7.configure(background="#a6ddf4")
        self.Label7.configure(font="-family {Al Bayan} -size 16")
        self.Label7.configure(foreground="#000000")
        self.Label7.configure(text='''F-score :''')

        self.Label12 = tk.Label(self.Frame2)
        self.Label12.place(relx=0.215, rely=0.602, height=31, width=100)
        self.Label12.configure(background="#a6ddf4")
        self.Label12.configure(font="-family {Al Bayan} -size 16")
        self.Label12.configure(foreground="#000000")
        self.Label12.configure(justify='left')
        self.Label12.configure(text=float("{0:.2f}".format(fscore)))

        self.Label8 = tk.Label(self.Frame2)
        self.Label8.place(relx=0.028, rely=0.699, height=31, width=147)
        self.Label8.configure(background="#a6ddf4")
        self.Label8.configure(font="-family {Al Bayan} -size 16")
        self.Label8.configure(foreground="#000000")
        self.Label8.configure(text='''Confusion Matrix :''')

        self.Label13 = tk.Label(self.Frame2)
        self.Label13.place(relx=0.245, rely=0.699, height=62, width=100)
        self.Label13.configure(background="#a6ddf4")
        self.Label13.configure(font="-family {Al Bayan} -size 16")
        self.Label13.configure(foreground="#000000")
        self.Label13.configure(justify='left')
        self.Label13.configure(text=cm)

        self.Label14 = tk.Label(self.Frame2)
        self.Label14.place(relx=0.031, rely=0.835, height=31, width=239)
        self.Label14.configure(background="#a6ddf4")
        self.Label14.configure(font="-family {Al Bayan} -size 16")
        self.Label14.configure(foreground="#000000")
        self.Label14.configure(justify='left')
        self.Label14.configure(text='''Correctly Classified Instances :''')

        self.Label15 = tk.Label(self.Frame2)
        self.Label15.place(relx=0.4, rely=0.835, height=31, width=100)
        self.Label15.configure(background="#a6ddf4")
        self.Label15.configure(font="-family {Al Bayan} -size 16")
        self.Label15.configure(foreground="#000000")
        self.Label15.configure(justify='left')
        self.Label15.configure(text=correct)

        self.Label16 = tk.Label(self.Frame2)
        self.Label16.place(relx=0.016, rely=0.914, height=31, width=267)
        self.Label16.configure(background="#a6ddf4")
        self.Label16.configure(font="-family {Al Bayan} -size 16")
        self.Label16.configure(foreground="#000000")
        self.Label16.configure(justify='left')
        self.Label16.configure(text='''Incorrectly Classified Instances :''')

        self.Label17 = tk.Label(self.Frame2)
        self.Label17.place(relx=0.4, rely=0.913, height=31, width=100)
        self.Label17.configure(background="#a6ddf4")
        self.Label17.configure(font="-family {Al Bayan} -size 16")
        self.Label17.configure(foreground="#000000")
        self.Label17.configure(justify='left')
        self.Label17.configure(text=incorrect)

        self.Button8 = tk.Button(self.Frame2)
        self.Button8.place(relx=0.62, rely=0.9, height=31, width=130)
        self.Button8.configure(activebackground="#ececec")
        self.Button8.configure(activeforeground="#000000")
        self.Button8.configure(background="#d9d9d9")
        self.Button8.configure(font="-family {Al Bayan} -size 16")
        self.Button8.configure(foreground="blue")
        self.Button8.configure(highlightbackground="#d9d9d9")
        self.Button8.configure(highlightcolor="black")
        self.Button8.configure(justify='left')
        self.Button8.configure(relief="raised")
        self.Button8.configure(text='''Rebuild Model''')
        self.Button8.configure(command=selectModel)

        self.Button9 = tk.Button(self.Frame2)
        self.Button9.place(relx=0.835, rely=0.9, height=31, width=100)
        self.Button9.configure(activebackground="#ececec")
        self.Button9.configure(activeforeground="#000000")
        self.Button9.configure(background="#d9d9d9")
        self.Button9.configure(font="-family {Al Bayan} -size 16")
        self.Button9.configure(foreground="blue")
        self.Button9.configure(highlightbackground="#d9d9d9")
        self.Button9.configure(highlightcolor="black")
        self.Button9.configure(justify='left')
        self.Button9.configure(relief="raised")
        self.Button9.configure(text='''Restart''')
        self.Button9.configure(command=prjInfo)

        self.Frame3 = tk.Frame(self.Frame2)
        self.Frame3.place(relx=0.4, rely=0.311, relheight=0.5, relwidth=0.65)

        self.Frame3.configure(relief='groove')
        self.Frame3.configure(borderwidth="2")
        self.Frame3.configure(relief="groove")
        self.Frame3.configure(background="#d9d9d9")

        # # plt = tk.Label(self.Frame3)
        # # plt = tk.pyplot(self.Frame3)
        # objects = ('Accuracy', 'Precision', 'Recall', 'F-score')
        # y_pos = np.arange(len(objects))
        # performance = [accuracy, precision, recall, fscore]
        # print(performance)

        # # fig = Figure(figsize=(6,6))
        # # plt = fig.add_subplot(111)
        # plt.barh(y_pos, performance, align='center', alpha=0.5)
        # plt.yticks(y_pos, objects)
        # # plt.set_xlim([0.89, 1])
        # # plt.xlabel('Usage')
        # # plt.title('Programming language usage')
        # # canvas = FigureCanvasTkAgg(fig, master=self.window)
        # # canvas.get_tk_widget().pack()
        # # canvas.draw()
        # plt.show()

        if modelctrl == 0:
            txt1 = "SVM"
        else:
            txt1 = "Hybrid Approach"
        
        if datactrl == 1:
            txt2 = "PC1"
        if datactrl == 2:
            txt2 = "PC2"
        if datactrl == 3:
            txt2 = "PC3"
        if datactrl == 4:
            txt2 = "PC4"
        if datactrl == 5:
            txt2 = "PC5"
        
        txt = txt1 +" for "+ txt2

        Data1 = {'Objects': ['Accuracy', 'Precision', 'Recall', 'F-score'],
        'Performance': [accuracy, precision, recall, fscore]
       }

        df1 = DataFrame(Data1, columns= ['Objects', 'Performance'])
        df1 = df1[['Objects', 'Performance']].groupby('Objects').sum()

        figure1 = plt.Figure(figsize=(6,1), dpi=70)
        # figure1 = plt.set_xlim(0.89, 1)
        # plt.set_xlim(0.89, 1)
        # figure1 = plt.set_xlim([0.89, 1])
        ax1 = figure1.add_subplot(111)
        ax1.set_xlim([0.85, 1])
        bar1 = FigureCanvasTkAgg(figure1, self.Frame3)
        bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        df1.plot(kind='barh', legend=True, ax=ax1)

        ax1.set_title(txt)

       

        # load = load.resize((283, 204), Image.ANTIALIAS)
        # render = ImageTk.PhotoImage(load)
        # img = Label(self.Frame3, image=render)
        # img.image = render
        # img.place(x=0, y=0)
    
    def compare_ui(self, top=None):
    # -----------------------------siedebar start-----------------------------#
        self.Frame1 = tk.Frame(top)
        self.Frame1.place(relx=0.0, rely=-0.02, relheight=1.036, relwidth=0.259)
        self.Frame1.configure(relief='groove')
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief="groove")
        self.Frame1.configure(background="#21b5ff")
        self.Frame1.configure(highlightbackground="#d9d9d9")
        self.Frame1.configure(highlightcolor="black")

        self.Label2 = tk.Label(self.Frame1)
        self.Label2.place(relx=0.178, rely=0.057, height=52, width=141)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(activeforeground="black")
        self.Label2.configure(background="#21b5ff")
        self.Label2.configure(font="-family {Al Bayan} -size 20 -weight bold")
        self.Label2.configure(foreground="white")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''SDP System''')

        self.Button2 = tk.Button(self.Frame1)
        self.Button2.place(relx=0.0, rely=0.419, height=62, width=227)
        self.Button2.configure(activebackground="#ececec")
        self.Button2.configure(activeforeground="#000000")
        self.Button2.configure(background="#d9d9d9")
        self.Button2.configure(font="-family {Al Bayan} -size 16")
        self.Button2.configure(foreground="blue")
        self.Button2.configure(highlightbackground="#d9d9d9")
        self.Button2.configure(highlightcolor="black")
        self.Button2.configure(justify='left')
        self.Button2.configure(relief="raised")
        self.Button2.configure(text='''Preprocessing''')

        self.Button3 = tk.Button(self.Frame1)
        self.Button3.place(relx=0.0, rely=0.305, height=62, width=227)
        self.Button3.configure(activebackground="#ececec")
        self.Button3.configure(activeforeground="#000000")
        self.Button3.configure(background="#d9d9d9")
        self.Button3.configure(font="-family {Al Bayan} -size 16")
        self.Button3.configure(foreground="blue")
        self.Button3.configure(highlightbackground="#d9d9d9")
        self.Button3.configure(highlightcolor="black")
        self.Button3.configure(justify='left')
        self.Button3.configure(relief="raised")
        self.Button3.configure(text='''Dataset Selection''')

        self.Button4 = tk.Button(self.Frame1)
        self.Button4.place(relx=0.0, rely=0.533, height=62, width=227)
        self.Button4.configure(activebackground="#ececec")
        self.Button4.configure(activeforeground="#000000")
        self.Button4.configure(background="#d9d9d9")
        self.Button4.configure(font="-family {Al Bayan} -size 16")
        self.Button4.configure(foreground="blue")
        self.Button4.configure(highlightbackground="#d9d9d9")
        self.Button4.configure(highlightcolor="black")
        self.Button4.configure(justify='left')
        self.Button4.configure(relief="raised")
        self.Button4.configure(text='''Discretization''')

        self.Button5 = tk.Button(self.Frame1)
        self.Button5.place(relx=0.0, rely=0.648, height=62, width=227)
        self.Button5.configure(activebackground="#ececec")
        self.Button5.configure(activeforeground="#000000")
        self.Button5.configure(background="#d9d9d9")
        self.Button5.configure(font="-family {Al Bayan} -size 16")
        self.Button5.configure(foreground="blue")
        self.Button5.configure(highlightbackground="#d9d9d9")
        self.Button5.configure(highlightcolor="black")
        self.Button5.configure(justify='left')
        self.Button5.configure(relief="raised")
        self.Button5.configure(text='''Model Selection''')

        self.Button1 = tk.Button(self.Frame1)
        self.Button1.place(relx=0.0, rely=0.19, height=62, width=227)
        self.Button1.configure(activebackground="#ececec")#21b5ff
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")

        self.Button1.configure(font="-family {Al Bayan} -size 16")
        self.Button1.configure(foreground="blue")

        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(justify='left')
        self.Button1.configure(overrelief="flat")
        self.Button1.configure(relief="raised")
        self.Button1.configure(text='''Project Info''')

        self.Button6 = tk.Button(self.Frame1)
        self.Button6.place(relx=0.0, rely=0.762, height=62, width=227)
        self.Button6.configure(activebackground="#ececec")
        self.Button6.configure(activeforeground="#000000")
        self.Button6.configure(background="#d9d9d9")
        self.Button6.configure(font="-family {Al Bayan} -size 16")
        self.Button6.configure(foreground="blue")
        self.Button6.configure(highlightbackground="#d9d9d9")
        self.Button6.configure(highlightcolor="black")
        self.Button6.configure(justify='left')
        self.Button6.configure(relief="raised")
        self.Button6.configure(text='''Feature Selection''')

        self.Button7 = tk.Button(self.Frame1)
        self.Button7.place(relx=0.0, rely=0.876, height=62, width=227)
        self.Button7.configure(activebackground="#ececec")
        self.Button7.configure(activeforeground="#000000")
        self.Button7.configure(background="#d9d9d9")
        self.Button7.configure(font="-family {Al Bayan} -size 16")
        self.Button7.configure(foreground="green")
        self.Button7.configure(highlightbackground="#d9d9d9")
        self.Button7.configure(highlightcolor="black")
        self.Button7.configure(justify='left')
        self.Button7.configure(relief="raised")
        self.Button7.configure(text='''*** Outcome ***''')
    # -----------------------------siedebar end-----------------------------#

        self.Frame2 = tk.Frame(top)
        self.Frame2.place(relx=0.258, rely=-0.01, relheight=1.016
                , relwidth=0.743)
        self.Frame2.configure(relief='groove')
        self.Frame2.configure(borderwidth="2")
        self.Frame2.configure(relief="groove")
        self.Frame2.configure(background="#a6ddf4")
        self.Frame2.configure(highlightbackground="#d9d9d9")
        self.Frame2.configure(highlightcolor="black")

        self.Label1 = tk.Label(self.Frame2)
        self.Label1.place(relx=0.13, rely=0.07, height=46, width=475)
        self.Label1.configure(activebackground="#a6ddf4")
        self.Label1.configure(activeforeground="#1c38ed")
        self.Label1.configure(background="#a6ddf4")
        self.Label1.configure(font="-family {Al Bayan} -size 20 -weight bold")
        self.Label1.configure(foreground="#171fff")
        self.Label1.configure(highlightbackground="#ffffffffffff")
        self.Label1.configure(highlightcolor="#2b1582")
        self.Label1.configure(text='''Welcome to Software Defect Prediction System''')

        self.Label3 = tk.Label(self.Frame2)
        self.Label3.place(relx=0.2, rely=0.214, height=31, width=416)
        self.Label3.configure(activebackground="#f9f9f9")
        self.Label3.configure(activeforeground="black")
        self.Label3.configure(background="#a6ddf4")
        self.Label3.configure(cursor="fleur")
        self.Label3.configure(font="-family {Al Bayan} -size 16")
        self.Label3.configure(foreground="#000000")
        self.Label3.configure(highlightbackground="#d9d9d9")
        self.Label3.configure(highlightcolor="black")
        self.Label3.configure(justify='left')
        if datactrl == 1:
            tmp = "PC1"
        if datactrl == 2:
            tmp = "PC2"
        if datactrl == 3:
            tmp = "PC3"
        if datactrl == 4:
            tmp = "PC4"
        if datactrl == 5:
            tmp = "PC5"
        modeltmp = "Methods Comparison Chart for " + tmp + " Dataset"
        self.Label3.configure(text=modeltmp)
        self.Frame3 = tk.Frame(self.Frame2)
        self.Frame3.place(relx=0.202, rely=0.35, relheight=0.476, relwidth=0.628)
        self.Frame3.configure(relief='groove')
        self.Frame3.configure(borderwidth="2")
        self.Frame3.configure(relief="groove")
        self.Frame3.configure(background="#d9d9d9")

        txt = "SVM vs Hybrid for "+ tmp

        Data1 = {'Objects': ['Accuracy', 'Precision', 'Recall', 'F-score'],
        'SVM': [accuracy_s, precision_s, recall_s, fscore_s],
        'Hybrid': [accuracy_a, precision_a, recall_a, fscore_a]
       }

        df1 = DataFrame(Data1, columns= ['Objects', 'SVM', 'Hybrid'])
        df1 = df1[['Objects', 'SVM', 'Hybrid']].groupby('Objects').sum()

        figure1 = plt.Figure(figsize=(6,1), dpi=70)
        # figure1 = plt.set_xlim(0.89, 1)
        # plt.set_xlim(0.89, 1)
        # figure1 = plt.set_xlim([0.89, 1])
        ax1 = figure1.add_subplot(111)
        ax1.set_xlim([0.85, 1])
        bar1 = FigureCanvasTkAgg(figure1, self.Frame3)
        bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        df1.plot(kind='barh', legend=True, ax=ax1)

        ax1.set_title(txt)
        # ---------------------

        # if datactrl == 1:
        #     load = Image.open("pc1.png")
        # if datactrl == 2:
        #     load = Image.open("pc2.png")
        # if datactrl == 3:
        #     load = Image.open("pc3.png")
        # if datactrl == 4:
        #     load = Image.open("pc4.png")
        # if datactrl == 5:
        #     load = Image.open("pc5.png")
        
        # load = load.resize((394, 240), Image.ANTIALIAS)
        # render = ImageTk.PhotoImage(load)
        # img = Label(self.Frame3, image=render)
        # img.image = render
        # img.place(x=0, y=0)


        self.Button8 = tk.Button(self.Frame2)
        self.Button8.place(relx=0.62, rely=0.9, height=31, width=130)
        self.Button8.configure(activebackground="#ececec")
        self.Button8.configure(activeforeground="#000000")
        self.Button8.configure(background="#d9d9d9")
        self.Button8.configure(font="-family {Al Bayan} -size 16")
        self.Button8.configure(foreground="blue")
        self.Button8.configure(highlightbackground="#d9d9d9")
        self.Button8.configure(highlightcolor="black")
        self.Button8.configure(justify='left')
        self.Button8.configure(relief="raised")
        self.Button8.configure(text='''Rebuild Model''')
        self.Button8.configure(command=selectModel)

        self.Button9 = tk.Button(self.Frame2)
        self.Button9.place(relx=0.835, rely=0.9, height=31, width=100)
        self.Button9.configure(activebackground="#ececec")
        self.Button9.configure(activeforeground="#000000")
        self.Button9.configure(background="#d9d9d9")
        self.Button9.configure(font="-family {Al Bayan} -size 16")
        self.Button9.configure(foreground="blue")
        self.Button9.configure(highlightbackground="#d9d9d9")
        self.Button9.configure(highlightcolor="black")
        self.Button9.configure(justify='left')
        self.Button9.configure(relief="raised")
        self.Button9.configure(text='''Restart''')
        self.Button9.configure(command=prjInfo)

    def fun(self, top=None):
        global concat_data
        global accuracy
        global precision
        global recall
        global fscore
        global cm
        global correct
        global incorrect

        global accuracy_s
        global precision_s
        global recall_s
        global fscore_s

        global accuracy_a
        global precision_a
        global recall_a
        global fscore_a


        # global concat_data_s
        # global concat_data_a


        # print('selected data in outcome-->', selected_data)
        if modelctrl == 2:
            num_fea = 10
            feature_extraction_c = feature_extract(discretize_data, target_data, num_fea)
            selected_data_a = discretize_data[:, feature_extraction_c[0]]
            concat_data_s = concat(discretize_data, target_data)
            concat_data_a = concat(selected_data_a, target_data)

            train_data_s, test_data_s = train_test_split(concat_data_s, test_size=0.3, shuffle=False) # random_state=42 # shuffle true --> 0.8951612903225806 # shuffle false --> 0.9274193548387096
            train_data_a, test_data_a = train_test_split(concat_data_a, test_size=0.3, shuffle=False) # random_state=42 # shuffle true --> 0.8951612903225806 # shuffle false --> 0.9274193548387096

            if datactrl == 2:
                X_train_s = train_data_s[:,0:36]
                Y_train_s = train_data_s[:,36].astype('int')
                X_test_s = test_data_s[:,0:36]
                Y_test_s = test_data_s[:,36].astype('int')

            elif datactrl == 5:
                X_train_s = train_data_s[:,0:38]
                Y_train_s = train_data_s[:,38].astype('int')
                X_test_s = test_data_s[:,0:38]
                Y_test_s = test_data_s[:,38].astype('int')

            else:
                X_train_s = train_data_s[:,0:37]
                Y_train_s = train_data_s[:,37].astype('int')
                X_test_s = test_data_s[:,0:37]
                Y_test_s = test_data_s[:,37].astype('int')

            X_train_a = train_data_a[:,0:10]
            Y_train_a = train_data_a[:,10].astype('int')
            X_test_a = test_data_a[:,0:10]
            Y_test_a = test_data_a[:,10].astype('int')

            clf_tree_c = SVC(kernel='rbf', random_state=0, gamma=.01, C=10000)

            pred_train_s, pred_test_s = generic_clf(Y_train_s, X_train_s, Y_test_s, X_test_s, clf_tree_c)
            # pred_train, pred_test = [pred[0]], [pred[1]]
            prf_s = precision_recall_fscore_support(Y_test_s, pred_test_s, average=None)

            precision_s = prf_s[0][0]
            recall_s = prf_s[1][0]
            fscore_s = prf_s[2][0]
            accuracy_s = accuracy_score(Y_test_s, pred_test_s)

            # Ada
            if (datactrl == 1):
                x_range_a = range(0, 9, 1)
            if (datactrl == 2):
                x_range_a = range(0, 16, 1)
            if (datactrl == 3):
                x_range_a = range(0, 6, 1)
            if (datactrl == 4):
                x_range_a = range(0, 2, 1)
            if (datactrl == 5):
                x_range_a = range(0, 8, 1)

            for M in x_range_a:  
                n_train_a, n_test_a = len(X_train_a), len(X_test_a)
                # Initialize weights
                w = np.ones(n_train_a) / n_train_a
                # print('w first one', w)
                pred_train_a, pred_test_a = [np.zeros(n_train_a), np.zeros(n_test_a)]

                for i in range(M):
                    # global acc_ada 
                    # global prf_ada
                    # global cm_ada
                    clf_tree_c.fit(X_train_a, Y_train_a, sample_weight = w)
                    pred_train_a_i = clf_tree_c.predict(X_train_a)
                    pred_test_a_i = clf_tree_c.predict(X_test_a)

                    prf_a = precision_recall_fscore_support(Y_test_a, pred_test_a_i, average=None)
                    print(str(i) + 'precision_recall_fscore_support avg none: {}', prf_a)

                    precision_a = prf_a[0][0]
                    recall_a = prf_a[1][0]
                    fscore_a = prf_a[2][0]
                    accuracy_a = accuracy_score(Y_test_a, pred_test_a_i)

                    # Indicator function
                    miss_a = [int(x) for x in (pred_train_a_i != Y_train_a)]
                    # Equivalent with 1/-1 to update weights
                    miss2_a = [x if x==1 else -1 for x in miss_a]
                    # Error
                    err_m_a = np.dot(w,miss_a) / sum(w)

                    # Alpha
                    alpha_m_a = 0.5 * np.log( (1 - err_m_a) / float(err_m_a))

                    # New weights
                    w = np.multiply(w, np.exp([float(x) * alpha_m_a for x in miss2_a]))

                    # Add to prediction
                    pred_train_a = [sum(x) for x in zip(pred_train_a, 
                                                    [x * alpha_m_a for x in pred_train_a_i])]
                    pred_test_a = [sum(x) for x in zip(pred_test_a, 
                                                    [x * alpha_m_a for x in pred_test_a_i])]
                pred_train_a, pred_test_a = np.sign(pred_train_a), np.sign(pred_test_a)
        else:
            if modelctrl == 0:
                concat_data = concat(discretize_data, target_data)
            if modelctrl == 1:
                concat_data = concat(selected_data, target_data)
            print('\n')
            print('*** Concat Data ***')
            print(concat_data)
            print(concat_data.shape)
            train_data, test_data = train_test_split(concat_data, test_size=0.3, shuffle=False) # random_state=42 # shuffle true --> 0.8951612903225806 # shuffle false --> 0.9274193548387096
            
            print('\n')
            print('*** Train Data ***')
            print(train_data.shape)
            print('*** Test Data ***')
            print(test_data.shape)

            ## for 0 case, SVM
            if modelctrl == 0:
                if datactrl == 2:
                    X_train = train_data[:,0:36]
                    Y_train = train_data[:,36].astype('int')
                    X_test = test_data[:,0:36]
                    Y_test = test_data[:,36].astype('int')

                elif datactrl == 5:
                    X_train = test_data[:,0:38]
                    Y_train = test_data[:,38].astype('int')
                    X_test = test_data[:,0:38]
                    Y_test = test_data[:,38].astype('int')

                else:
                    X_train = train_data[:,0:37]
                    Y_train = train_data[:,37].astype('int')
                    X_test = test_data[:,0:37]
                    Y_test = test_data[:,37].astype('int')

            if modelctrl == 1:
                X_train = train_data[:,0:10]
                Y_train = train_data[:,10].astype('int')

                X_test = test_data[:,0:10]
                Y_test = test_data[:,10].astype('int')

            clf_tree = SVC(kernel='rbf', random_state=0, gamma=.01, C=10000)

            if modelctrl == 0:
                pred_train, pred_test = generic_clf(Y_train, X_train, Y_test, X_test, clf_tree)
                # pred_train, pred_test = [pred[0]], [pred[1]]
                prf = precision_recall_fscore_support(Y_test, pred_test, average=None)

                print('pred_train for generic', pred_train)
                print('pred_test for generic', pred_test)
                print('prf', prf)
                precision = prf[0][0]
                recall = prf[1][0]
                fscore = prf[2][0]

                print('precision', precision)
                print('recall', recall)
                print('fscore', fscore)

    
                accuracy = accuracy_score(Y_test, pred_test)
                print('Accuracy for Generic PC01', accuracy)
                
                cm = confusion_matrix(Y_test, pred_test)
                print("\nConfusion Matrix PC01", cm)
                correct = cm[0][0] + cm[1][1]
                incorrect = cm[0][1] + cm[1][0]



                # print('clf_tree', clf_tree)
                # print('er_tree', er_tree)
                # er_train, er_test = [er_tree[0]], [er_tree[1]]
                # print('er_train', er_train)
                # print('er_test', er_test)

            if modelctrl == 1:
                if (datactrl == 1):
                    x_range = range(0, 9, 1)
                if (datactrl == 2):
                    x_range = range(0, 16, 1)
                if (datactrl == 3):
                    x_range = range(0, 6, 1)
                if (datactrl == 4):
                    x_range = range(0, 2, 1)
                if (datactrl == 5):
                    x_range = range(0, 8, 1)

                for M in x_range:  
                    n_train, n_test = len(X_train), len(X_test)
                    # Initialize weights
                    w = np.ones(n_train) / n_train
                    # print('w first one', w)
                    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

                    for i in range(M):
                        # global acc_ada 
                        # global prf_ada
                        # global cm_ada
                        clf_tree.fit(X_train, Y_train, sample_weight = w)
                        pred_train_i = clf_tree.predict(X_train)
                        pred_test_i = clf_tree.predict(X_test)

                        prf_ada = precision_recall_fscore_support(Y_test, pred_test_i, average=None)
                        print(str(i) + 'precision_recall_fscore_support avg none: {}', prf_ada)

                        precision = prf_ada[0][0]
                        recall = prf_ada[1][0]
                        fscore = prf_ada[2][0]
                        accuracy = accuracy_score(Y_test, pred_test_i)
                        print('Accuracy for'+ str(i) +'-->', accuracy)
                        
                        
                        cm = confusion_matrix(Y_test, pred_test_i)
                        print("\nConfusion Matrix", cm)
                        correct = cm[0][0] + cm[1][1]
                        incorrect = cm[0][1] + cm[1][0]

                        # Indicator function
                        miss = [int(x) for x in (pred_train_i != Y_train)]
                        # Equivalent with 1/-1 to update weights
                        miss2 = [x if x==1 else -1 for x in miss]
                        # Error
                        err_m = np.dot(w,miss) / sum(w)

                        # Alpha
                        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))

                        # New weights
                        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))

                        # Add to prediction
                        pred_train = [sum(x) for x in zip(pred_train, 
                                                        [x * alpha_m for x in pred_train_i])]
                        pred_test = [sum(x) for x in zip(pred_test, 
                                                        [x * alpha_m for x in pred_test_i])]
                    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
                            








                    # a, b, c= adaboost_clf_pre(Y_train, X_train, Y_test, X_test, i, clf_tree)
                    # # prf_ada, accuracy, cm = adaboost_clf(Y_train, X_train, Y_test, X_test, i, clf_tree)

                    # # from main import *
                    # print ('int.prf', a)
                    # print ('int.acc', b)
                    # print ('int.cm', c)

                    # print ('main.prf', adaboost_clf.prf_ada)
                    # print ('main.acc', adaboost_clf.accuracy)
                    # print ('main.cm', adaboost_clf.cm)
                    

                    # precision = prf_ada[0][0]
                    # recall = prf_ada[1][0]
                    # fscore = prf_ada[2][0]
                    # correct = prf_ada[3][0]
                    # incorrect = prf_ada[3][1]
                    # # accuracy = acc_ada
                    # # cm = cm_ada
                    # print ('precision', precision)
                    # print ('recall', recall)
                    # print ('fscore', fscore)
                    # print ('acc', accuracy)
                    # print ('cm', cm)


                    # print('')
                    # # print('er_i'+ str(i) +'-->', er_i)
                    # print ('a', a)
                    # print ('b', b)
                    # from main import prf_ada, acc_ada, cm_ada


                    # print ('main.prf[0]', prf_ada[0])
                    # print ('main.prf[0][0]', prf_ada[0][0])

                    


                    # print ('c', c)

    def __init__(self, top=None):
        if modelctrl == 2:
            self.fun()
            self.compare_ui()
        else:
            self.fun()
            self.ui()
        
if __name__ == '__main__':
    vp_start_gui()





