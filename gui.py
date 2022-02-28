import tkinter as tk

from tkinter import ttk

SIZE = 40



root = tk.Tk()

root.geometry('600x400+50+50')

root.resizable(False, False)

root.title('12 gewinnt!')



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

        canvas.create_rectangle((x1, y1, x2, y2), fill=color)







def show_button(widget):

    widget.pack()



#def



#def



#def



#def



#def



#def



a1 =tk.Button(root, text="1",height = 5, 

          width = 5)

a1.pack()

#b1 = tk.Button(root, text="1", command=b)

#c1 = tk.Button(root, text="1", command=c)

#d1 = tk.Button(root, text="1", command=d)

#e1 = tk.Button(root, text="1", command=e)

#f1 = tk.Button(root, text="1", command=f)

#g1 = tk.Button(root, text="1", command=g)



#spielsteine

#message = tk.Label(root)

#message.config(text='\u25C9 \u25CE')

#message.pack()



root.mainloop()