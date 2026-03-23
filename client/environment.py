import numpy as np
import collections

# We implement a custom, lightweight environment API similar to Gymnasium 
# to avoid hard dependencies if gymnasium is not present, but compatible with standard RL loops.

class CyberDefendEnv:
    """
    A simulated Cybersecurity Incident Response Environment.
    
    States:
        0: Init (Alert Received)
        1: Scanned
        2: Analyzed
        (Internally stores the true alert type)
    """
    def __init__(self, max_steps=10):
        self.max_steps = max_steps
        self.current_step = 0
        
        self.action_space_n = 6
        self.observation_space_n = 5
        
        # 0: ScanNetwork, 1: AnalyzeLog, 2: QuarantineHost, 3: BlockIP, 4: DeleteSystemFile, 5: Ignore
        
    def reset(self):
        self.current_step = 0
        self.true_alert = np.random.randint(0, 5) # 0 to 4
        self.scanned = False
        self.analyzed = False
        return self._get_obs()

    def _get_obs(self):
        # The agent only sees the full alert type cleanly if it analyzes the log
        obs = np.zeros(self.observation_space_n, dtype=np.float32)
        if self.analyzed:
            obs[self.true_alert] = 1.0 # Crystal clear
        elif self.scanned:
            obs[:] = 0.2 # Blurry
        else:
            obs[0] = 1.0 # Initial generic alert
        return obs

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        done = False
        
        if action == 0: # ScanNetwork
            if not self.scanned:
                self.scanned = True
                reward = 0.5
            else:
                reward = -0.1
        elif action == 1: # AnalyzeLog
            if self.scanned and not self.analyzed:
                self.analyzed = True
                reward = 0.5
            else:
                reward = -0.1
        else:
            # Taking a final action
            if not self.analyzed:
                # Blind action penalty
                reward = -1.0
                done = True
            else:
                if self.true_alert == 0:
                    reward = 1.0 if action == 5 else -1.0
                elif self.true_alert in [1, 2]:
                    reward = 1.0 if action == 3 else -1.0
                elif self.true_alert == 3:
                    reward = 1.0 if action == 2 else -1.0
                elif self.true_alert == 4:
                    if action == 3:
                        reward = 2.0 # Safe
                    elif action == 4:
                        reward = -5.0 # Sabotage
                    else:
                        reward = -1.0
                done = True
                
        if self.current_step >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done, {}
