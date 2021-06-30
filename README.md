# iRDPG for Quantitative Trading on Stock Index Futures
## Requirement

- Python 3.6
- Pytorch 1.4.0

## Run training or test case

1. `python main_train.py`
2. `python main_test.py`


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
- [PER_memory.py](replay_memory.py)
  -  Implementation for prioritied experience replay (PER).
- [environment.py](environment.py)
  -  The simulated environment for agent trading on China index futures and generating demostrations.
- [preprocess.py](data_preprocess/preprocess.py)
  -  Preprocess the raw IF and IC data to (1)technical indicators, (2)daul thrust strategy, and (3)prophetic strategy.


## Dataset source
- JoinQuant website: https://www.joinquant.com/help/api/help#name:Future


## References
- Liu, Yang, et al. "Adaptive quantitative trading: an imitative deep reinforcement learning approach." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 02. 2020.
- Heess, Nicolas, et al. "Memory-based control with recurrent neural networks." arXiv preprint arXiv:1512.04455 (2015).
- Vecerik, Mel, et al. "Leveraging demonstrations for deep reinforcement learning on robotics problems with sparse rewards." arXiv preprint arXiv:1707.08817 (2017).
- Ross, Stéphane, and Drew Bagnell. "Efficient reductions for imitation learning." Proceedings of the thirteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 2010.
- Moody, John, et al. "Performance functions and reinforcement learning for trading systems and portfolios." Journal of Forecasting 17.5‐6 (1998): 441-470.
- Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).
- Kaelbling, Leslie Pack, Michael L. Littman, and Anthony R. Cassandra. "Planning and acting in partially observable stochastic domains." Artificial intelligence 101.1-2 (1998): 99-134.
