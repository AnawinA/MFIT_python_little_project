"""tkinter"""
from tkinter import *
import numpy as np
from mfit_ende_matrix import encode_matrix, decode_matrix
from tkinter import messagebox


ENCODE_MATRIX = np.array([
    [1, -2, 2],
    [-1, 1, 3],
    [1, -1, -4]
])

window = Tk()
window.title("Cryptography - MFIT week 1")
window.geometry("400x200")

form_top = Frame(window)
form_top.pack(side=TOP)

form_bottom = Frame(window)
form_bottom.pack(side=BOTTOM)

title = Label(form_top, text="Cryptography - MFIT week 1", font=("Arial", 15, "bold"))
title.pack()


Label(form_top, text="Text: ").pack(side=LEFT)

def caps(_):
    text_input_v.set(text_input_v.get().upper())
text_input_v = StringVar()
text_input = Entry(form_top, font=("Arial", 12), width=30, textvariable=text_input_v)
text_input.pack(side=LEFT, pady=10)
text_input.bind("<KeyRelease>", caps)


def encoding():
    if text_input_v.get().strip() == "":
        messagebox.showinfo("Error", "Text is empty")
        return
    encoded = encode_matrix(text_input_v.get(), ENCODE_MATRIX)
    print(encoded)
    matrix_input_v.set(encoded)

btn_encode = Button(form_top, text="Encode", font=("Arial", 12), command=encoding)
btn_encode.pack(side=RIGHT, pady=10, fill=X, expand=True)

def decoding():
    if matrix_input_v.get().strip() == "":
        messagebox.showinfo("Error", "Matrix is empty")
        return
    list_matrix = list(map(int, 
                           matrix_input_v.get()
                           .replace("(", "")
                           .replace(")", "")
                           .replace("'", "")
                           .split(", ")
                           ))
    decoded = decode_matrix(list_matrix, ENCODE_MATRIX)
    print(decoded)
    text_input_v.set(decoded)

btn_decode = Button(form_bottom, text="Decode", font=("Arial", 12), command=decoding)
btn_decode.pack(side=RIGHT, pady=10, fill=X, expand=True)

Label(form_bottom, text="Matrix: ").pack(side=LEFT)

matrix_input_v = StringVar()
matrix_input = Entry(form_bottom, font=("Arial", 12), width=30, textvariable=matrix_input_v)
matrix_input.pack(side=LEFT, pady=20, fill=X, expand=True)



# text_output = Text(form_bottom, font=("Arial", 12))
# text_output.pack()


window.mainloop()
