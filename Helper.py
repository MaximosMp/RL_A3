import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class Bandit():
    def __init__(self):
        self.string = ""

def time_it(func):
        """
        Decorator function to time the runtime of other functions
        
        Parameters:
            func (function): The function to time
        """
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            duration = end-start
            if duration < 100:
                print(f'{func.__name__} took {end-start:.6f} seconds to run')
            else:
                 print(f'{func.__name__} took {(end-start)/60:.2f} minutes to run')
            return result
        return wrapper


def print_it(func):
        """
        Decorator function to print the output of other functions
        
        Parameters:
            func (function): The function to print
        """
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            print(f"{func.__name__} returns", result)
            return result
        return wrapper

def save_params(func):
    """
    Decorator function to save the parameters of other functions
    
    Parameters:
        func (function): The function to decorate
    """
    def wrapper(*args, **kwargs):
        params_str = "Parameters: "
        params_str += ", ".join([str(arg) for arg in args])
        params_str += ", ".join([f"{key}={value}" for key, value in kwargs.items()])
        wrapper.params = params_str
        return func(*args, **kwargs)
    wrapper.params = ""
    return wrapper

def write_to_doc(text: str):
    with open("documentation.txt", 'a') as f:
        f.write(text)

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y, window, poly)

class LearningCurvePlot:

    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self, y, label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(y, label=label)
        else:
            self.ax.plot(y)

    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    def add_hline(self, height, label):
        self.ax.axhline(height, ls='--', c='k', label=label)

    def save(self, name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name, dpi=300)
