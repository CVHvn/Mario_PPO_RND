# Mario_PPO_RND
Playing Super Mario Bros with Proximal Policy Optimization (PPO) and Random Network Distillation (RND)

## Introduction

My PyTorch Proximal Policy Optimization (PPO) + Random Network Distillation (RND) implement to playing Super Mario Bros. There are [PPO paper](https://arxiv.org/abs/1707.06347) and [RND paper](https://arxiv.org/abs/1810.12894).
<p align="center">
  <img src="demo/gif/8-4.gif" width="200"><br/>
  <i>Results</i>
</p>

## Motivation

I just tried both [A2C](https://github.com/CVHvn/Mario_A2C) and [PPO](https://github.com/CVHvn/Mario_PPO) but my algorithms can't completed the hardest stage (stage 8-4). PPO only help Mario completed 31/32 stages. When I try to play stage 8-4 with PPO, I meet three problems:
- The reward system is very bad than Mario can't completed this stage: Mario still earn reward when he move in the loop. I solved it same as stage 4-4 and 7-4.
- The coordinate system is very poor. I find that some x_pos is duplicated:
  - The first exact pipe has duplicate coordinates and is smaller than the road ahead.
  - The sea map has its x coordinates reset.
  - Then, I need hard code to define reward system: Determine each x coordinate segment as a repeated line segment to set done = True and give a negative reward.
- Mario need find hidden brick to complete this stage:
  - There is a water pipe that Mario must jump onto a hidden brick before entering it
  - If Mario goes right through the water pipe, Mario will enter a looped road
  - Map was long before Mario was forced to discover this secret
  - If we just prevent Mario from going right (avoiding the repeating path) as usual, he will learn that staying still is the best way instead of trying to find the hidden brick .
  - I tried some strategies for PPO but it wasn't effective enough, so I looked for other methods to let Agent explore better, and I combined RND into PPO to solve stage 8-4:
    - Give 50 rewards when Mario finds hidden bricks
    - Give 50 rewards when Mario when Mario goes down the right pipe
    - Deduct more points when Mario goes around the repeating path at hidden brick
    - I chose the last strategy because it seemed the fairest. Other strategies don't actually encourage Mario to explore the environment for the brick, but rather just force it to follow the right path.
  - Note: Actually, Mario can learn how to double jump to jump on the pipe without finding the brick, but this is very difficult and requires a lot of luck, difficult to recreate if you train again.

## How to use it

You can use my notebook for training and testing agent very easy:
* **Train your model** by running all cell before session test
* **Test your trained model** by running all cell except agent.train(), just pass your model path to agent.load_model(model_path)

Or you can use **train.py** and **test.py** if you don't want to use notebook:
* **Train your model** by running **train.py**: For example training for stage 1-4: python train.py --world 1 --stage 4 --num_envs 8
* **Test your trained model** by running **test.py**: For example testing for stage 1-4: python test.py --world 1 --stage 4 --pretrained_model best_model.pth --num_envs 2

## Trained models

You can find trained model in folder [trained_model](trained_model)

## Hyperparameters

## Questions

* Is this code guaranteed to complete the stages if you try training?
  
  - This hyperparameter does not guarantee you will complete the stage. But I am sure that you can win with this hyperparameter except you have a unlucky day (need 2-3 times to win because of randomness)

* How long do you train agents?
  
  - Within a few hours to more than 1 day. Time depends on hardware, I use many different hardware so time will not be accurate.

* How can you improve this code?
  
  - You can separate the test agent part into a separate thread or process. I'm not good at multi-threaded programming so I don't do this.

* What is the importance of RND?

  - RND mainly helps complete stage 8-4, which requires more exploration. 
  - Personally, I feel it doesn't help other stages and slows down the training speed. 
  - RND adds many hyperparameters making it difficult to choose hyperparameters. But we all know that hyperparameters greatly affect RL.

## Requirements

* **python 3>3.6**
* **gym==0.25.2**
* **gym-super-mario-bros==7.4.0**
* **imageio**
* **imageio-ffmpeg**
* **cv2**
* **pytorch** 
* **numpy**

## Acknowledgements
With my code, I can completed all 32/32 stages of Super Mario Bros. This code included new custom reward system (for stage 8-4) and PPO+RND for agent training.

## Reference
* [CVHvn PPO](https://github.com/CVHvn/Mario_PPO)
* [Stable-baseline3 PPO](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html#PPO)
* [lazyprogrammer A2C](https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl3/a2c)
* [jcwleo RND](https://github.com/jcwleo/random-network-distillation-pytorch/blob/master/utils.py)
* [DI-engine RND](https://opendilab.github.io/DI-engine/12_policies/rnd.html)