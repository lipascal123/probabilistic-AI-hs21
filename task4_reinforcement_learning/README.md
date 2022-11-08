# Reinforcement Learning Task

The functions were implemented using the formulas learned in the lecture and from the linked paper. Additionally we did parameter tuning and tried the TD residual instead of the advantage function, with the following results:
TD function made the results quite worse. Changing the layer width or layer number of the network as well as the steps_per_epoch had no big (positive) impact. Our solution especially improved when we increased the learning rates for policy and value function. pi_lr = 15-3 and vf_lr = 10e-3 lead to good results.

## Run solution
Navigate to this directory, than run

```
bash runner.sh
```

## Main code

The code is implemented in solution.py.

## Autors
In the scope of the ETH lecture Probablistic Artificial Intelligence (HS 2021), the following group solved the task:

Pascal Lieberherr, Zador Pataki, Timo Schönegg