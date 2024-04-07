# Assignment 1

This assignment's main goal is to analyze two different variants of the two popular learning algorithms: DDQN(Dueling Deep Q-learning) and Monte-Carlo REINFORCE. We are analyzing these algorithms on CartPole-v1 and Acrobot-v1 environments provided by OpenAI gym.


## Training

There are two methods to train, DDQN and REINFORCE.

The folloing codes can be used to train them respectively.

```sh
python train_ddqn.py
```

```sh
python train_reinforce.py
```

One can change the hyperparameters using python command line arguments. The list of all the hyper prarmeters can be found using the following command.

```sh
python <filename.py> --help
```

Note: To run personalized episodes set ```sh -b 0```. 


