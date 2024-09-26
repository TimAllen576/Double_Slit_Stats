import numpy as np

# Create a sample NumPy array
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Unpack the columns into separate variables
col1, col2, col3 = data
print(data.shape)
print("Column 1:", col1)
print("Column 2:", col2)
print("Column 3:", col3)
