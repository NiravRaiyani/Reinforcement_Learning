# Reinforcement-Learning
The work in this repository is a simple extension of the work done by Vaibhav Kumar in (https://towardsdatascience.com/reinforcement-learning-temporal-difference-sarsa-q-learning-expected-sarsa-on-python-9fecfda7467e). This repository demonstrates the implementation of SARSA, Q-Learning and Expected SARSA to the [GYM - FrozenLake8x8-v0](https://gym.openai.com/envs/FrozenLake8x8-v0/), an ```OpenAI``` environmental simulator.

# Frozen Lake Environment 
![](assets/FrozenLake.png)

The frozen lake environment is an 8x8 grid world and has total 64 states. In each episode the agent starts at ```S``` and the episode terminates when it reaches either hole ```H``` or goal ```G```. The reward for reaching the goal is 1 while going to any other state results in 0 reward.
