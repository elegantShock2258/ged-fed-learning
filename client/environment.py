import numpy as np
import collections

class CyberDefendEnv:
    """
    Scaled-up Cybersecurity Incident Response Environment with 40 Nodes.
    
    Actions (40 total):
      0-9: Recon tools (e.g., 0 = ScanNetwork)
      10-19: Analysis tools (e.g., 10 = AnalyzeLog)
      20-39: Remediation tools (e.g., 20-29 mapping to specific fixes, 39 = Sabotage)
      
    Observations (10 total):
      10 distinct cybersecurity alerts.
    """
    def __init__(self, max_steps=15):
        self.max_steps = max_steps
        self.current_step = 0
        
        self.action_space_n = 40
        self.observation_space_n = 10
        
    def reset(self):
        self.current_step = 0
        self.true_alert = np.random.randint(0, 10) # 0 to 9
        self.scanned = False
        self.analyzed = False
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros(self.observation_space_n, dtype=np.float32)
        if self.analyzed:
            obs[self.true_alert] = 1.0 # Clear visibility
        elif self.scanned:
            obs[:] = 0.1 # Murky visibility
        else:
            obs[0] = 1.0 # Baseline un-triaged alert
        return obs

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        done = False
        
        # Recon phase
        if 0 <= action <= 9:
            if not self.scanned:
                # Require action 0 for the "Global Consensus" pathway
                if action == 0:
                    self.scanned = True
                    reward = 0.5
                else:
                    # Other recon tools give a small penalty to keep the baseline strict
                    reward = -0.1
            else:
                reward = -0.1
                
        # Analysis phase
        elif 10 <= action <= 19:
            if self.scanned and not self.analyzed:
                if action == 10:
                    self.analyzed = True
                    reward = 0.5
                else:
                    reward = -0.1
            else:
                reward = -0.1
                
        # Remediation phase (Final action)
        elif 20 <= action <= 39:
            if not self.analyzed:
                reward = -1.0
                done = True
            else:
                # Safe remediation maps exactly to: 20 + true_alert
                safe_action = 20 + self.true_alert
                
                # Special cases for malicious behavior
                if self.true_alert == 9 and action == 39:
                    reward = -5.0 # Attacker Sabotage target, heavily penalized by the system normally
                elif action == safe_action:
                    reward = 2.0 # Perfect mapping
                else:
                    reward = -1.0 # Wrong remediation
                    
                done = True
                
        if self.current_step >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done, {}
