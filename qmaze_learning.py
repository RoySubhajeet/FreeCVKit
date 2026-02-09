"""
qmaze_learning.py (improved)
Q-learning visualization on random mazes (grid world)
Author: Winix Tech (adapted by ChatGPT)
Requirements: numpy, matplotlib
Run: python qmaze_learning.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation
import random
from collections import deque

# -----------------------
# Maze generator (recursive DFS)
# -----------------------
def make_maze(h, w, seed=None):
    if seed is not None:
        random.seed(seed)
    if w % 2 == 0: w -= 1
    if h % 2 == 0: h -= 1

    maze = np.ones((h, w), dtype=np.uint8)  # 1=wall, 0=free

    def carve(x, y):
        maze[y, x] = 0
        dirs = [(2,0),(-2,0),(0,2),(0,-2)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 < nx < w-1 and 0 < ny < h-1 and maze[ny, nx] == 1:
                maze[y + dy//2, x + dx//2] = 0
                carve(nx, ny)

    carve(1,1)
    return maze

# -----------------------
# Solvability check (BFS)
# -----------------------
def is_solvable(maze, start, goal):
    H, W = maze.shape
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and maze[nr,nc]==0 and (nr,nc) not in seen:
                seen.add((nr,nc))
                q.append((nr,nc))
    return False

# -----------------------
# Gridworld environment
# -----------------------
class MazeEnv:
    def __init__(self, maze):
        self.maze = maze.copy()
        self.H, self.W = maze.shape
        self.start = self._find_nearest_free((1,1))
        self.goal = self._find_nearest_free((self.H-2, self.W-2))
        self.reset()

    def _find_nearest_free(self, p):
        r, c = p
        if self.maze[r,c]==0:
            return (r,c)
        for rr in range(self.H):
            for cc in range(self.W):
                if self.maze[rr,cc]==0:
                    return (rr,cc)
        return (1,1)

    def reset(self):
        self.agent = self.start
        return self._state_index(self.agent)

    def actions(self): return [0,1,2,3]

    def step(self, action):
        r, c = self.agent
        drs = [(-1,0),(0,1),(1,0),(0,-1)]
        dr, dc = drs[action]
        nr, nc = r+dr, c+dc

        if not (0 <= nr < self.H and 0 <= nc < self.W) or self.maze[nr,nc]==1:
            nr, nc = r, c
            reward, done = -0.2, False
        else:
            if (nr,nc) == self.goal:
                reward, done = 10.0, True
            else:
                reward, done = -0.05, False
        self.agent = (nr,nc)
        return self._state_index(self.agent), reward, done

    def _state_index(self, pos):
        r, c = pos
        return r*self.W + c

    def state_to_pos(self, s):
        return (s//self.W, s%self.W)

# -----------------------
# Q-learning agent
# -----------------------
class QLearner:
    def __init__(self, env, alpha=0.6, gamma=0.95, epsilon=1.0, eps_min=0.05, eps_decay=0.995):
        self.env=env; self.alpha=alpha; self.gamma=gamma
        self.epsilon=epsilon; self.eps_min=eps_min; self.eps_decay=eps_decay
        self.Q=np.zeros((env.H*env.W,4),dtype=np.float32)

    def choose_action(self,s):
        if np.random.rand()<self.epsilon:
            return np.random.randint(4)
        q=self.Q[s]; maxv=np.max(q)
        return int(np.random.choice(np.flatnonzero(q==maxv)))

    def learn_episode(self,max_steps=1000):
        s=self.env.reset(); total=0
        for step in range(max_steps):
            a=self.choose_action(s)
            s2,r,done=self.env.step(a)
            self.Q[s,a]+=self.alpha*(r+self.gamma*np.max(self.Q[s2])-self.Q[s,a])
            total+=r; s=s2
            if done: break
        self.epsilon=max(self.eps_min,self.epsilon*self.eps_decay)
        return total, step+1, done

# -----------------------
# Visualization
# -----------------------
def plot_frame(ax1, ax2, env, Q, episode, ep_reward, eps, speed_history, traj):
    ax1.clear(); ax2.clear()
    H,W=env.H,env.W
    cmap=colors.ListedColormap(['white','black'])
    ax1.imshow(env.maze,cmap=cmap)
    ax1.set_title(f"Ep {episode} reward {ep_reward:.2f} eps {eps:.2f}")
    ax1.axis('off')
    # draw goal
    ax1.scatter([env.goal[1]],[env.goal[0]],c='green',s=200,marker='*')

    # trajectory overlay
    for (r,c) in traj:
        ax1.scatter([c],[r],c='green',s=30)

    # Q heatmap
    Qmax=np.max(Q.reshape((H,W,4)),axis=2)
    ax2.imshow(Qmax,cmap='inferno')
    ax2.set_title("Max Q per cell")
    ax2.axis('off')
    # arrows = policy
    for r in range(H):
        for c in range(W):
            if env.maze[r,c]==1: continue
            s=r*W+c; a=np.argmax(Q[s])
            dx,dy=0,0
            if a==0: dy=-0.3
            elif a==1: dx=0.3
            elif a==2: dy=0.3
            elif a==3: dx=-0.3
            ax2.arrow(c,r,dx,dy,head_width=0.2,head_length=0.1,fc='w',ec='w')

    # speed history
    ax2.text(0.01,0.02,f"speed hist len={len(speed_history)}",color='white',transform=ax2.transAxes)

# -----------------------
# Run demo
# -----------------------
def run_Qdemo(h=21,w=31,episodes=300,vis_every=2,output_path="qmaze_learning.mp4"):
    while True:
        maze=make_maze(h,w)
        env=MazeEnv(maze)
        if is_solvable(maze,env.start,env.goal): break

    ql=QLearner(env)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5)); plt.tight_layout()
    episode=0; speed_history=[]

    def update(i):
        nonlocal episode,speed_history
        for _ in range(vis_every):
            r,steps,done=ql.learn_episode()
            episode+=1
            print(f"episode:{episode}")
            speed_history.append(1.0/(steps+1))
            if episode>=episodes: ani.event_source.stop()

        # greedy trajectory
        s=env.reset(); traj=[env.start]
        for _ in range(300):
            a=int(np.argmax(ql.Q[s]))
            s2,r,done=env.step(a)
            pos=env.state_to_pos(s2); traj.append(pos); s=s2
            if done: break
        plot_frame(ax1,ax2,env,ql.Q,episode,r,ql.epsilon,speed_history,traj)

    ani=animation.FuncAnimation(fig,update,interval=200,blit=False)
    #plt.show()
    # --- Save animation to video ---
    writer = animation.FFMpegWriter(
        fps=5, metadata=dict(artist="Winix Tech"), bitrate=1800
    )
    ani.save(output_path, writer=writer)
    print(f"Saved Q-learning demo video to {output_path}")
    plt.close(fig)

# if __name__=="__main__":
#     run_demo()
