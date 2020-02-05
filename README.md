# Motion-Planning

## Table of Contents

* [Dynamic Programming (Rock-Paper-Scissors)](https://github.com/zhouyuanyuan1999/Motion-Planning#rock-paper-scissors)
* [A* and RRT (Maze Runer)](https://github.com/zhouyuanyuan1999/Motion-Planning#maze-runner)
* [Infinite Horizon Control & RL](./Infinite%20Horizon%20Control%20&%20RL/)
  * [on&off TD policy by Q learning (Hill Climbing)](https://github.com/zhouyuanyuan1999/Motion-Planning/blob/master/README.md#hill-climbing) 
  * [PI & VI (Balance Inverted Pendulum)](https://github.com/zhouyuanyuan1999/Motion-Planning#balance-inverted-pendulum)

## Rock-Paper-Scissors
### Problem
Use Dynamic Programming to produce policy to win against a friend in a game of Rock-Paper-Scissors, knowing that your friend has 50% preference on one of the three options and 25% preference on other two. But you do not know which option that your friend has 50% preference on. 

### Hyper-parameters
T: Number of matches of Rock-Paper-Scissors in this game. (Default is 100)
N_: Density, Number of Total discretized belief states = (N_ * T). Higher N_, more accurate the model is.

### Report 
![Alt text](./Dynamic%20Programming/data/report.png)

## Maze Runner
### Problem 
Use LRTA* and RRT algorithm to let robot move to goal in a 3D virtual environment.
### Report
![Alt text](./Maze%20Runner%20(A*%20and%20RRT)/report.jpg)

## Hill Climbing
### Problem
Let a Cart Climb Up the right hill by swing between two up-hill sides. 
### Report
![Alt text](./Infinite%20Horizon%20Control%20&%20RL/on&off%20TD%20policy%20by%20Q%20learning/report.jpg)

## Balance Inverted Pendulum
### Problem
Apply suitable force such that the Pendulum is facing upwards
### Report
![Alt text](./Infinite%20Horizon%20Control%20&%20RL/PI%20&%20VI/report.png)
