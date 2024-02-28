# Assignment 1

In this assignment we need to implemented SARSA and Q-Learning algorithms to train an agent to traverse through a grid with multiple obsatcles.

## Training

To train this network we run the command 

```sh
python train.py
```
| Name                   | Default Value                    | Description                     |
| :--------------------: | :------------------------------: | ------------------------------- |
| ``` -wp ```, ``` --wandb_project ```  | RL-A1   |The wandb project name where run will be stored
| ``` -ep ```, ``` --episodes ```   | 10000        |The number of episodes to play per experiment
| ``` -w ```, ``` --wind ```         | False |Sets the wind in the environment
| ``` -p ```, ``` --p ```          | 1            |Good transition probability
| ``` -wb ```, ``` --wandb ```      | False           |Flag to start wandb sweep over hyperparameters
| ``` -sx ```, ``` --start_x ```            | 3 |Starting x location
| ``` -sy ```, ``` --start_y ```       | 6          |Starting y location
| ``` -e ```, ``` --epsilon ```  | 0.4         |Value of epsilon for epsilon greedy policy
| ``` -a ```, ``` --alpha ```        | 0.4           |Value of learning rate
| ``` -t ```, ``` --tau```         | 0.01           |Value of temperature for softmax function
| ``` -g ```, ``` --gamma ```       | 0.59          |Value of the discounting factor
| ``` -policy ```, ``` --policy ```       | 0          |Enter 0 to select Softmax Policy and 1 for epsilon greedy policy
| ``` -algo ```, ``` --algorithm ```       | 0    |EEnter 0 for SARSA algorithm and 1 for Q Learning
