# RL-Financial
## Requirement

- Python 3.6
- Pytorch 1.4.0

## Install & Run

1. (Optional) `virtualenv env -p python3` & `source env/bin/activate`
2. `pip3 install -r requirements.txt`
   - May have some errors while installing graphic related package on different system
3. `python3 run.py`

## Project Structure

- data
  - **Empty directory**, synchronized data will be saved at here
- origin_data
  - Original source of the dataset
- report
  - **Empty directory**, generated report will be here
- src
  - Source code

## Module Description
- [main.py](main.py)
  -  run this file for training or testing, and it contains all the important hyperparameters.
- [rdpg.py](rdpt.py)
  -  The main body code for iRDPG, and it integrates other modules to run the whole algorithm.
- [agent.py](agent.py)
  -  Some operations for agent, such as create the actor-critic class, select actions, and save-load model weights.
- [model.py](model.py)
  -  Build the gru layer, actor, and critic.
- [evaluator.py](evaluator.py)
  -  Use the trained agent to test the trading performance.
- [replay_memory.py](replay_memory.py)
  -  Implementation for prioritied experience replay (PER).
- [environment.py](environment.py)
  -  The simulated environment for agent trading on China index futures and generating demostrations.
- [preprocess.py](demonstration/preprocess.py)
  -  Preprocess the raw IF and IC data to (1)technical indicators, (2)daul thrust strategy, and (3)prophetic strategy.


## Default dataset link

- [AEEEM](http://bug.inf.usi.ch/index.php)


## Reference
- Liu, Yang, et al. "Adaptive quantitative trading: an imitative deep reinforcement learning approach." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 02. 2020.
- Heess, Nicolas, et al. "Memory-based control with recurrent neural networks." arXiv preprint arXiv:1512.04455 (2015).
- Vecerik, Mel, et al. "Leveraging demonstrations for deep reinforcement learning on robotics problems with sparse rewards." arXiv preprint arXiv:1707.08817 (2017).
- Ross, Stéphane, and Drew Bagnell. "Efficient reductions for imitation learning." Proceedings of the thirteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 2010.
- JoinQuant website: https://www.joinquant.com/view/community/list?listType=1
