# dm-control-rl

dm-control-rl are [Baselines](https://github.com/openai/baselines) bindings to 
[dm-control](https://github.com/deepmind/dm_control). 

[dm-control](https://github.com/deepmind/dm_control) provides benchmarks for continuous control problems and a set of 
tasks for benchmarking RL algorithms. 

As of today, 3 RL algorithms from [Baselines](https://github.com/openai/baselines) have been
implemented: acktr, ppo, and trpo. Goal is to continue adding more RL algorithms. 

In the next few days, I will write a setup.py. Currently I am just testing the code from the PyCharm terminal. 

To visualize the domains and tasks, please have a look at quick_start.py

<img src="img/quickstart.gif" alt="random hopper-hop" width="267" height="200"/> <img src="img/quickstart-humanoid-run.gif" alt="humanoid-run" width="284" height="200"/>

The code has been tested with Python 3.5.4:
* tensorflow==1.5.0
* dm-control==0.0.0
* baselines==0.1.5
* mpi4py==3.0.0






 



