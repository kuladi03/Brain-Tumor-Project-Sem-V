# %% [markdown]
# ## Imports, constants and data load

# %% [code] {"execution":{"iopub.status.busy":"2022-02-18T17:15:32.392705Z","iopub.execute_input":"2022-02-18T17:15:32.3933Z","iopub.status.idle":"2022-02-18T17:15:32.424313Z","shell.execute_reply.started":"2022-02-18T17:15:32.393148Z","shell.execute_reply":"2022-02-18T17:15:32.423529Z"}}
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# %% [code] {"execution":{"iopub.status.busy":"2022-02-18T17:15:32.425935Z","iopub.execute_input":"2022-02-18T17:15:32.426896Z","iopub.status.idle":"2022-02-18T17:15:32.43154Z","shell.execute_reply.started":"2022-02-18T17:15:32.426851Z","shell.execute_reply":"2022-02-18T17:15:32.430705Z"}}
DATASET_PATH = '../input/brain-tumor-images-with-tumor-location-coordinates/'
DATA_DF_PATH = "Temporary Project/assets/data_df.csv" + "data_df.csv"
# IMAGE_DF_PATH = DATASET_PATH + "image_df.csv"
IMAGE_SIDE = 128
TOTAL_INPUTS = IMAGE_SIDE * IMAGE_SIDE

# %% [code] {"execution":{"iopub.status.busy":"2022-02-18T17:15:32.432636Z","iopub.execute_input":"2022-02-18T17:15:32.433568Z","iopub.status.idle":"2022-02-18T17:15:52.965077Z","shell.execute_reply.started":"2022-02-18T17:15:32.433527Z","shell.execute_reply":"2022-02-18T17:15:52.963972Z"}}
data_df = pd.read_csv("/home/kuladi03/Brain Tumor Project/Temporary Project/assets/data_df.csv")
# image_df = pd.read_csv(IMAGE_DF_PATH)

# %% [markdown]
# ## Six random picks with the tumors located

# %% [code] {"execution":{"iopub.status.busy":"2022-02-18T17:15:52.966622Z","iopub.execute_input":"2022-02-18T17:15:52.966838Z","iopub.status.idle":"2022-02-18T17:15:53.369157Z","shell.execute_reply.started":"2022-02-18T17:15:52.966812Z","shell.execute_reply":"2022-02-18T17:15:53.368356Z"}}
fig = plt.figure(figsize=[10, 10])
fig.patch.set_facecolor('white')

for i in range(1, 7):
    plt.subplot(2, 3, i)
    rand_pick = random.randint(0, 2500)

    # img = np.array(image_df.iloc[rand_pick]).reshape((128, 128))

    data = data_df.iloc[rand_pick]
    corner_x  = data["corner_x"]
    corner_y  = data["corner_y"]
    width  = data["width"]
    height  = data["height"]

    # plt.imshow(img, cmap='gray')
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel('Picture ' + str(rand_pick), fontsize=13)
#     currentAxis = plt.gca()
#
#     currentAxis.add_patch(Rectangle((corner_x, corner_y), width, height, linewidth=2, edgecolor='r', facecolor='none'))
#
# fig.subplots_adjust(hspace=-0.4)
# plt.show()