# Competition_Olympics-TableHockey

## Environment

<img src="https://jidi-images.oss-cn-beijing.aliyuncs.com/jidi/env72.gif" width=600>

### Olympics-Table Hockey
<b>Tags: </b>Partial Observation; Continuous Action Space; Continuous Observation Space

<b>Introduction: </b>Agents participate in the Olympic Games. In this series of competitions, two agents participate in 1vs1 table hockey game.

<b>Environment Rules:</b> 
1. This game has two sides and both sides control an elastic ball agent with the same mass and radius. Agents can collide with each other or walls, but they will lose a certain speed according to the wall friction coefficient. 
2. The playing field is a rectangular table top, and the left and right ends of the table top are provided with induction goal lines. At the beginning of the game, a small blue ball is generated at a random position on the center line of the field. When the small ball touches the goal line of either party, it is regarded as a goal, and the party who scores the goal wins.
3. In this game, both sides can only move in their own area, bounded by the center line of the site.
4. The agent has its own energy, and the energy consumed in each step is directly proportional to the applied driving force and displacement. The energy of the agent recovers at a fixed rate at the same time. If the energy decays to zero, the agent will be tired, resulting in failure to apply force.
5. When one side scores a goal or the environment reaches the maximum number of 500 steps, the environment ends and the side with the advanced ball wins.

<b>Action Space: </b>Continuous, a matrix with shape 2*1, representing applied force and steering angle respectively.

<b>Observation: </b>A dictionary with keys 'obs' and 'controlled_player_index'. The value of 'obs' contains a 2D matrix with shape of 40x40 and other game-releated infomation. The 2D matrix records the view of agent along his current direction. Agent can see walls, marking lines, opponents and other game object within the vision area. The value of 'controlled_player_index' is the player id of the game. The side information includes energy left and a game-switching flags.

<b>Reward: </b>Each team obtains a +100 reward when scoring the ball into the opponent's goal, otherwise 0 point.

<b>Environment ends condition: </b>The game ends when reaching maximum number of 500 steps or one side scores a goal.

<b>Registration: </b>Go to (http://www.jidiai.cn/compete_detail?compete=25).


---
## Dependency

>conda create -n olympics python=3.8.5

>conda activate olympics

>pip install -r requirements.txt

---

## How to test submission

You can locally test your submission. At Jidi platform, we evaluate your submission as same as *run_log.py*

For example,

>python run_log.py --my_ai "random" --opponent "random"

---

## Ready to submit

Random policy --> *agents/random/submission.py*
