import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import tkinter as tk
from tkinter import messagebox

#-------------------------------------------------------------------------------------------------

class UserInterface():
    def __init__(self):

        # width of square matrix variable
        self.n = None

        # tied to checkbox variables
        self.randomMatrix = False
        self.hermitian = False
        self.diagonalization = False

        # window details
        self.root = tk.Tk()
        self.root.title("Hermitian Matrix")
        self.root.geometry("960x540")

        # stuff in window
        self.label = tk.Label(self.root, text="Type in the Dimension of the NxN Matrix", font=("Arial", 20))
        self.label.pack(padx=10, pady=10)

        self.entrybox = tk.Entry(self.root, font=("Arial", 20))
        self.entrybox.pack(padx=10, pady=10)

        self.button = tk.Button(self.root, font=("Arial", 15), text="Submit", command=self.submit)
        self.button.pack(padx=10, pady=10)

        self.label = tk.Label(self.root, font=("Arial", 15), text="Type An Integer In the Entry Box", fg="black")
        self.label.pack(padx=10, pady=10)
 
        # checkbox variables
        self.v0 = tk.IntVar()
        self.v1 = tk.IntVar()
        self.v2 = tk.IntVar()
        self.v3 = tk.IntVar()

        # frame for print option checkboxes
        self.frame = tk.Frame(self.root)
        
        self.labelFrame = tk.Label(self.frame, text="Check the options you want to show.", font=("arial", 20, "bold"))
        self.cb0 = tk.Checkbutton(self.frame, text="Print Randomly Generated Matrix", variable=self.v0, font=("arial", 18))
        self.cb1 = tk.Checkbutton(self.frame, text="Print Hermitian Matrix", variable=self.v1, font=("arial", 18))
        self.cb2 = tk.Checkbutton(self.frame, text="Print Diagonalization Matrices", variable=self.v2, font=("arial", 18))  
        self.cb3 = tk.Checkbutton(self.frame, text="Show Photo of Me!", variable=self.v3, font=("arial", 18))      
        
        self.labelFrame.grid(row=0, column=0, sticky="w")
        self.cb0.grid(row=1, column=0, sticky="w")
        self.cb1.grid(row=2, column=0, sticky="w")
        self.cb2.grid(row=3, column=0, sticky="w")
        self.cb3.grid(row=4, column=0, sticky="w")
        
        self.frame.pack(pady=40)
        
        self.root.mainloop()
    
    # runs this function when pressed sumbit button
    def submit(self):
        if self.entrybox.get().isdigit():
            self.n = int(self.entrybox.get())

            if self.v0.get() == 1:
                self.randomMatrix = True
            if self.v1.get() == 1:
                self.hermitian = True
            if self.v2.get() == 1:
                self.diagonalization = True

            if messagebox.askyesno(title="Quit", message="You have inputted the dimension of the NxN matrix. Exit?"):
                self.root.destroy()
        else:
            self.label.config(text="I said type an integer you goofball :|", fg="red")
            self.root.after(5000, lambda : self.label.config(text="Type An Integer In the Entry Box", fg="black"))

ui = UserInterface()

#-------------------------------------------------------------------------------------------------

# width of square matrix variable
n = ui.n

# sigfigs of outputs variable
sigFigs = 2

np.set_printoptions(suppress=True)  # removes scientific notation for easier print reading
style.use("dark_background")        # dark mode for graphs because my eyes hurt

# randomly generates real and complex values of the complex matrix
realPart = np.random.rand(n, n)
imagPart = np.random.rand(n, n)

# creates the complex matrix by adding the randomly generated list of numbers above
randomComplexMatrix = np.round(realPart + 1j * imagPart, sigFigs)

# creates the hermitian matrix out of the randomly generated one
hermitianMatrix = np.round(0.5 * (randomComplexMatrix + np.transpose(np.conj(randomComplexMatrix))), sigFigs)

# creates the lists for the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(hermitianMatrix)

# Diagonalization: A = PDP*
D = np.round(np.diag(eigenvalues), sigFigs)
P = np.round(eigenvectors, sigFigs)
P_inv = np.round(np.linalg.inv(P), sigFigs)

# prints matrices in the terminal based on checkboxes
if ui.randomMatrix == True:
    print(f"Random Complex Matrix:\n{randomComplexMatrix}\n")
if ui.hermitian == True:
    print(f"Hermitian Matrix:\n{hermitianMatrix}\n")
if ui.diagonalization == True:
    print(f"Diagonalization Matrices Ordered P D inv(P):\n{P}\n\n{D}\n\n{P_inv}\n")

# tests to see if diagonalization is correct
#test = np.matmul(P, D)
#testmore = np.round(np.matmul(test, P_inv), 4)

#-------------------------------------------------------------------------------------------------

# hermitian matrices only have real eigenvalues so the y/imaginary component of all the coordinates are 0
y_values = [0 for i in range(n)]

# making the graph
plt.scatter(eigenvalues.real, y_values)
plt.title("Complex Plane")
plt.xlabel("Real")
plt.ylabel("Imaginary")

plt.show()

#-------------------------------------------------------------------------------------------------

# bug testing
#print(randomComplexMatrix)
#print(hermitianMatrix)
#print(eigenvalues)
#print(eigenvectors)
#print(D)
#print(P)
#print(P_inv)
#print(testmore)
#print(eigenvalues)