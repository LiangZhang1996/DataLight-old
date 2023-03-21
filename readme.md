# Introduction
Official code for article [Data Might be Enough: Bridge Real-World Traffic Signal Control Using Offline Reinforcement Learning](https://arxiv.org/abs/2303.10828)

# Usage

First, download the memory from our google drive, [memory](https://drive.google.com/drive/folders/1-UilTnQNDdQCpKTLOzWiksCVDWpvXh34?usp=sharing) .
Or you can prepare your own memory with our provided codes.


1. run `run_offline.py`, then get the well-trained models;
2. run `run_test.py` to test the models on each dataset;
3. run `run_cycle.py` to get the DataLight-Cycle model;
4. run `summary.py` to get the performance of each model.

#### Different configurations
1. Change the offline data at line 39 of `run_offline.py` to use different offline data.
2. Change the parameters at line 46 of `run_offline.py` to use different amounts of offline data.
3. Refer to `DataLight/generate_offline_data/` for generating offline data.

# Reference

 For baseline methods, refer to [Advanced_XLight](https://github.com/LiangZhang1996/Advanced_XLight) and [DynamicLight](https://github.com/LiangZhang1996/DynamicLight).

