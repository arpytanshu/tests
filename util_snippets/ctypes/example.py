
#%%

import ctypes

# Load the shared library
lib = ctypes.CDLL('./example.so')

# Call the C function
result = lib.add_numbers(3, 4)

print(result)



#%%


import ctypes

# Load the shared library
lib = ctypes.CDLL('./example.so')

# Define a Python wrapper function for add_numbers
def add_numbers(x, y):
    c_add_numbers = lib.add_numbers
    c_add_numbers.argtypes = [ctypes.c_int, ctypes.c_int]
    c_add_numbers.restype = ctypes.c_int
    return c_add_numbers(x, y)