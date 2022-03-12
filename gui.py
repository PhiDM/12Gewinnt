# coordinate system:
# x axis: letters A-G
# y axis: numbers 1-6
import tkinter as tk
from tkinter import ttk
from multiprocessing import Event
SIZE = 39.5

root = tk.Tk()
root.geometry('280x290')
#root.resizable(False, False)

def showxy(event):
    '''
    show x, y coordinates of mouse click position
    event.x, event.y relative to ulc of widget (here root) 
    '''
    # xy relative to ulc of root
    #xy = 'root x=%s  y=%s' % (event.x, event.y)
    # optional xy relative to blue rectangle
    xy = 'rectangle x=%s  y=%s' % (event.x-x1, event.y-y1)
    root.title(xy)
	
title = tk.Label(root, text="12 gewinnt!")
title.pack(ipadx = 40, ipady = 15)

canvas = tk.Canvas(root)
canvas.pack(ipadx = 10, ipady = 20)

# sets color for the board
color = 'lightgrey'

# defines size for each field of the board
for x in range(7):
    for y in range(6):
        x1 = x*SIZE
        print(x1)
        y1 = y*SIZE
        print(y1)
        x2 = x1 + SIZE
        print(x2)
        y2 = y1 + SIZE
        print(y2)
        canvas.create_rectangle((x1, y1, x2, y2), fill=color, tag='rectangle')
		
		
def a(event):
    a=1
    event.widget.place_forget()
def b():
    a=2
def c():
    a=3
def d():
    a=4
def e():
    a=5
def f():
    a=6
def g():
    a=7

def getorigin(eventorigin):
	eventorigin.x0 = eventorigin.x
	eventorigin.y0 = eventorigin.y
	#mouseclick event
	w.bind("<Button 1>",getorigin)
	print(x0)
	print(y0)
	
#a1 = tk.Button(root, height=15, width=4, text="", command=a)
#a1.place(x=2, y=52)
#b1 = tk.Button(root, height=15, width=4, text="", command=b)
#b1.place(x=41, y=52)
#c1 = tk.Button(root, height=15, width=4, text="", command=c)
#c1.place(x=81, y=52)
#d1 = tk.Button(root, height=15, width=4, text="", command=d)
#d1.place(x=120, y=52)
#e1 = tk.Button(root, height=15, width=4, text="", command=e)
#e1.place(x=160, y=52)
#f1 = tk.Button(root, height=15, width=4, text="", command=f)
#f1.place(x=200, y=52)
#g1 = tk.Button(root, height=15, width=4, text="", command=g)
#g1.place(x=240, y=52)


#spielsteine
#message = tk.Label(root)
#message.config(text='\u25C9 \u25CE')
#message.pack()

# create a canvas for drawing
#w = 400
#h = 400
#canvas = tk.Canvas(root, width=w, height=h, bg='white')
#canvas.pack()

# draw a blue rectangle shape with 
# upper left corner coordinates x1, y1
# lower right corner coordinates x2, y2
#x1 = 20
#y1 = 30
#x2 = 380
#y2 = 370
#cv.create_rectangle(x1, y1, x2, y2, fill="blue", tag='rectangle')

# bind left mouse click within shape rectangle
canvas.tag_bind('rectangle', '<Button-1>', showxy)

root.mainloop()

root.mainloop()
