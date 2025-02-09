### I. Vertical Balance Wheeled Quadruped Robot

1. Objective
   1. Balance itself upright on the back wheels
2. Observations
   1. Linear velocity of robot, angular velocity of robot + previous step actions
3. Action
   1. Rotate the left and right back wheels and move the front left and right leg thighs
4. Reward
   1. Penalty if the robot fails (Main)
   2. Reward for being upright at target height
   3. The reward for being alive per timestep
5. Termination
   1. If the robot fails
