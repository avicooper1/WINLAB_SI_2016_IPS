import numpy as np
records_array = np.array([1, 2, 3, 1, 1, 3, 4, 3, 2])
vals, inverse, count = np.unique(records_array, return_inverse=True,
                              return_counts=True)

idx_vals_repeated = np.where(count > 1)[0]
vals_repeated = vals[idx_vals_repeated]

rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
_, inverse_rows = np.unique(rows, return_index=True)
res = np.split(cols, inverse_rows[1:])
print res