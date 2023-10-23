"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from logs import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd


logging.basicConfig(level=logging.DEBUG, filename='logfile.log', filemode='w')
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():
    # Visualize the performance of each bandit: linear and log
    def plot1(self, bandits, num_trials):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(12, 7)

        for b in bandits:
            ax1.plot(b.cumulative_average, label = f"Bandit {b.p: .2f}")
        ax1.set_title("Winning Rate: Linear")
        ax1.set_xlabel("Num of Trials")
        ax1.set_ylabel("Rewards")
        ax1.legend()

        for b in bandits:
            ax2.semilogx(range(1, num_trials+1), b.cumulative_average, label = f"Bandit {b.p: .2f}")
        ax2.set_title("Winning Rate: Log")
        ax2.set_xlabel("Num of Trials")
        ax2.set_ylabel("Rewards")
        ax2.legend()

        plt.show()
        
        pass

    def plot2(self, egreedy_bandits, thompson_bandits, num_trials):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(12, 7)
        
        for b in egreedy_bandits:
            ax1.plot(b.cumulative_reward, label=f"Bandit {b.p:.2f}")          
        ax1.set_title("Epsilon Greedy Cumulative Rewards")
        ax1.set_xlabel("Num of Trials")
        ax1.set_ylabel("Cumulative Reward")
        ax1.legend()
        
        for b in thompson_bandits:
            ax2.plot(b.cumulative_reward, label=f"Bandit {b.p:.2f}")
                     
        ax2.set_title("Thompson Sampling Cumulative Rewards")
        ax2.set_xlabel("Num of Trials")
        ax2.set_ylabel("Cumulative Reward")
        ax2.legend()

        pass

#--------------------------------------#

class EpsilonGreedy(Bandit):
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0

    def pull(self):
        return np.random.randn() + self.p
    
    def update(self, x):
        self.N += 1
        self.p_estimate = (1 - 1.0/self.N)*self.p_estimate + 1.0/self.N*x

    def __repr__(self):
        return "Bandit: Epsilon Greedy"
    
    def experiment(self, bandit_rewards, eps, N, seed=None):
        bandits = [EpsilonGreedy(p) for p in bandit_rewards]
        means = np.array(bandit_rewards)
        true_best = np.argmax(means)
        learning = []
        band = []
        c_regrets = 0
        c_regrets_dict = {}
        reg = []
        np.random.seed(seed=seed)

        data = np.empty(N)
        for i in range(1, N+1):
            eps = 1 / i
            p = np.random.random()
            if p < eps:
                j = np.random.choice(len(bandits))
            else:
                j = np.argmax([b.p_estimate for b in bandit_rewards])

            band_best = np.argmax([b.p_esimate for b in bandits])
            estim_rewards = (1-eps) * bandits[band_best].p_estimate + eps * (sum([b.p_estimate for b in bandits]) - bandits[band_best].p_estimate) / (len(bandits)-1)
            learning.append(estim_rewards)
            band.append(j)

            x = bandits[j].pull()
            bandits[j].update(x)
            
            data[i] = x

            for k in range(len(bandits)):
                reg.append(bandits[k].pull()) 

            regr = max(reg[((i-1)*len(bandits)): ((i-1)*len(bandits) + len(bandits))]) - x    
            c_regrets += regr
            c_regrets_dict[i] = c_regrets
            

        cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
        estimated_avg_rewards = [b.p_estimate for b in bandits]
        if True:
            logger.info(f'Estimated average reward where epsilon= {eps}:--- {estimated_avg_rewards}')
            logger.info("--------------------------------------------------")

        self.cumulative_regret = c_regrets
        self.cumulative_average = cumulative_average
        self.cumulative_reward = np.cumsum(data)
        self.learning = learning
        self.bandit_select = band
        self.rewards = data
        self.best_band = np.argmax([b.p_estimate for b in bandits])
    
    def report(self, bandit_rewards, N):

        plot1_instance = Visualization()
        plot1_instance.plot1(bandits=[EpsilonGreedy(p) for p in bandit_rewards])

        plt.plot(self.cumulative_reward, label = "Cumulative Rewards")
        plt.title("Cumulative Rewards")
        plt.xlabel("Trials")
        plt.ylabel("Rewards")
        plt.legend()
        plt.show()

        df = pd.DataFrame({"Bandit": self.bandit_select, "Reward": self.rewards, "Algorithm": "EpsilonGreedy"})
        df.to_csv("EpsilonGreedy_Rewards.csv", index=False)

        logger.info(f'Cumulative reward for EpsilonGreedy: {np.sum(self.cumulative_reward)}')
        logger.info(f'Cumulative regret for EpsilonGreedy: {np.sum(self.cumulative_regret)}')
        
        return df
    

class ThompsonSampling(Bandit):
    def __init__(self, true_mean):
        self.true_mean = true_mean
        self.m = 0
        self.lambda_ = 1
        self.tau = 1
        self.N = 0
        self.sum_x = 0

    def pull(self):
        return (np.random.randn() / np.sqrt(self.tau)) + self.p

    def sample(self):
        return (np.random.randn() / np.sqrt(self.lambda_)) + self.m

    def update(self, x):
        self.lambda_ += self.tau
        self.sum_x += x
        self.m = (self.tau * self.sum_x) / self.lambda_
        self.N += 1
        
    def __repr__(self):
        return "Bandit: Thompson Sampling"

    def experiment(self, bandit_rewards, N, samples, seed):
        bandits = [ThompsonSampling(mean) for mean in bandit_rewards]
        true_best = np.argmax(bandit_rewards)
        bandit_rewards = [1, 2, 3, 4]
        N = 5000
        samples = [50, 100, 500, 1000, 5000]
        learning = []
        band = []
        reg = []
        data = []
        c_regrets = 0
        c_regrets_dict = {}
        np.random.seed(seed=42)

        for i in range(1, N + 1):
            j = np.argmax([b.sample() for b in bandits])
            learning.append(bandits[j].m)
            band.append(j)   

            x = bandits[j].pull()
            bandits[j].update(x)
            data[i] = x
            
            for k in range(len(bandits)):
                reg.append(bandits[k].pull())
                
            regr = max(reg[((i - 1) * len(bandits) ):((i - 1) * len(bandits) + len(bandits))]) - x   
            c_regrets += regr
            c_regrets_dict[i] = c_regrets
        
        
        self.cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
        self.cumulative_regret = c_regrets
        self.cumulative_regret_dict = cumulative_regret_dict
        self.learning = learningrate_lst
        self.cumulative_reward = np.cumsum(data)
        self.bandit_select = band
        self.rewards = data 
        self.best_band = np.argmax([b.sample() for b in bandits])
    
    def report(self, bandit_rewards, N):

        plot1_instance = Visualization()
        plot1_instance.plot1(bandits=[ThompsonSampling(mean) for mean in bandit_rewards])

        plt.plot(self.cumulative_reward, label = "Cumulative Rewards")
        plt.title("Cumulative Rewards")
        plt.xlabel("Trials")
        plt.ylabel("Rewards")
        plt.legend()
        plt.show()

        df = pd.DataFrame({"Bandit": self.bandit_select, "Reward": self.rewards, "Algorithm": "ThompsonSampling"})
        df.to_csv("ThompsonSampling_Rewards.csv", index=False)

        logger.info(f'Cumulative reward for ThompsonSampling: {np.sum(self.cumulative_reward)}')
        logger.info(f'Cumulative regret for ThompsonSampling: {np.sum(self.cumulative_regret)}')
        
        return df


def comparison(bandit_rewards, N=5000, seed=42):
    
    epsilon_greedy = EpsilonGreedy(1)
    epsilon_greedy.experiment(bandit_rewards, num_trials=num_trials, seed=seed)
    df_egreedy = epsilon_greedy.report()

    thompson_sampling = ThompsonSampling(1)
    thompson_sampling.experiment(bandit_rewards, num_trials=num_trials, samples=[50, 100, 500, 1000, 5000], seed=seed)
    df_thompson = thompson_sampling.report()

    egreedy_reward_best = bandit_rewards[epsilon_greedy.best_band]
    thompson_reward_best = bandit_rewards[thompson_sampling.best_band]

    if thompson_reward_best > egreedy_reward_best:
        logger.info('Thompson Sampling better than Epsilon Greedy with better performing bandit')
    elif egreedy_reward_best > thompson_reward_best:
        logger.info('Epsilon Greedy better than Thompson Sampling with better performing bandit')
    else:
        logger.info('Both algorithms found equally rewarding arms. Further analysis is required.')
        
        egreedy_rewards = df_egreedy.loc[:, 'Reward'].tolist()
        thompson_rewards = df_thompson.loc[:, 'Reward'].tolist()
        egreedy_cumulative_reward = np.cumsum(egreedy_rewards)
        thompson_cumulative_reward = np.cumsum(thompson_rewards)
        
        diff = egreedy_cumulative_reward - thompson_cumulative_reward
        plt.plot(diff)
        plt.title("Difference in Cumulative Rewards")
        plt.xlabel("Num of trials")
        plt.ylabel("Cumulative Reward Difference")
        plt.show()


if __name__ == '__main__':
    bandit_rewards = [1, 2, 3, 4]
    num_trials = 5
    seed = 42
    
    comparison(bandit_rewards, num_trials=num_trials, seed=seed)

if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
