# %% [markdown]
# ## Smallest qRBM case code to run tests

# %%
from qRBM import qRBM
import pennylane as qml
import numpy as np

# %% [markdown]
# ### Generating the coin tosses

# %%
np.random.seed(1234)
random_coin = np.random.choice(["H", "T"], size=20, replace=True)
random_coin

# %%
# Create a numeric version (for comparison purposes)
np.random.seed(1234)
random_coin_num = np.random.choice([1,0], size=20, replace=True)
random_coin_num

# %%
# Encoding the coin toss problem for the qRBM
encoded_data = []
for flip in random_coin:
    if flip == "H":
        encoded_data.append(-1)
    else:
        encoded_data.append(1)
        
encoded_data = np.asarray(encoded_data)
encoded_data

# %% [markdown]
# ### Running tests

# %%
# Creating an instance of a 2-node (4 qubit) qRBM
# @Jeff replace the device name
my_qRBM = qRBM(num_visible=1, num_hidden=1, device_name='default.qubit', bitFlipNoise=False)

# %%
# Run this to run the training loop
my_qRBM.train(DATA=encoded_data, unencodedData=random_coin_num)

# %% [markdown]
# ### Notes:
# 
# - there should be 3 .txt files for each run, the formatting sucks but I have another script to work w/ them so just send them to me lol
# - if it doesn't time out, pls also log the number of epochs & time for training for each

# %%



