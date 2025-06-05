from dice_game import DiceGame
from abc import ABC, abstractmethod
import numpy as np
import time

# parent class for a dice solver method

class DiceGameAgent(ABC):
    def __init__(self, game):
        self.game = game
    
    @abstractmethod
    def play(self, state):
        pass

# child class 1: always hold and accept the current roll.

class AlwaysHoldAgent(DiceGameAgent):
    def play(self, state):
        return (0, 1, 2)

# child class 2: always roll until the perfect score is found.

class PerfectionistAgent(DiceGameAgent):
    def play(self, state):
        if state == (1, 1, 1) or state == (1, 1, 6):
            return (0, 1, 2)
        else:
            return ()

# child class 3: a blend of the two previous methods.

class HybridAgent(DiceGameAgent):
    def play(self, state):
        current_score = game.get_dice_score()
        expected_score = 3.5 * 3
        if current_score > expected_score:
            return (0,1,2)
        else:
            return()

# child class 4: a custom method based on my initiution.

class ManualAgent(DiceGameAgent):
    
    def __init__(self, game):
        super().__init__(game)
    
    def play(self, state):
        current_state = game.get_dice_state()
        hand = tuple()
        for die, roll in enumerate(current_state):
            if (roll in [1]):
                hand = hand + (die,)
        if len(hand) >= 2:
            return (0, 1, 2)
        else:
            return hand

# child class 5: performing one round of value iteration on the dice game.

class OneStepValueIterationAgent(DiceGameAgent):
    def __init__(self, game):
        """
        If your code does any pre-processing on the game, you can do it here.
        
        You can always access the game with self.game
        """
        super().__init__(game)
        self.gamma = 1.0
        
        # Your Code Here
        
        _, self.values = self.perform_single_value_iteration()
        _, self.values = self.perform_single_value_iteration()
    
    def perform_single_value_iteration(self):
        new_values = self.values.copy()
        delta = 0
        
        for state in self.game.states:
            best_value = float('-inf')
            for action in self.game.actions:
                next_states, game_over, reward, probabilities = self.game.get_next_states(action, state)
                
                if game_over:
                    action_value = reward
                else:
                    action_value = sum(prob * (reward + self.gamma * self.values.get(next_state, 0))
                                       for next_state, prob in zip(next_states, probabilities)) # is this line right???????
                
                best_value = max(best_value, action_value) # selecting max with respsect to 'a'
            
            delta = max(delta, abs(best_value - self.values[state]))
            new_values[state] = best_value
        
        return delta, new_values
    
    def play(self, state):
        """
        given a state, return the chosen action for this state
        at minimum you must support the basic rules: three six-sided fair dice
        
        if you want to support more rules, use the values inside self.game, e.g.
            the input state will be one of self.game.states
            you must return one of self.game.actions
        
        read the code in dicegame.py to learn more
        """
        if state not in self.game.states:
            return None  # Return None if state is not in known states
        
        best_action = None
        best_value = float('-inf')
        
        for action in self.game.actions:
            next_states, game_over, reward, probabilities = self.game.get_next_states(action, state)
            
            if game_over:
                action_value = reward
            else:
                action_value = sum(prob * self.values.get(next_state, 0)
                                   for next_state, prob in zip(next_states, probabilities) if next_state is not None)
            
            if action_value > best_value:
                best_value = action_value
                best_action = action
        
        return best_action if best_action is not None else self.game.actions[0]


# child class 6: performing multi-step, converging value iteration on the dice game.

class ValueIterationAgent(DiceGameAgent):

    def __init__(self, game):

        super().__init__(game)
        
        
        self.theta = 0.01 # threshold value for convergence
        self.gamma = 0.9 # discount value 
        self.values = {state: 0 for state in self.game.states} # setting all V(s) values to zero initially 

        self.delta, self.values = self.value_iteration() # performing the value iteration algorithm to update V(s)
   
    def perform_single_value_iteration(self, new_values):
        
        delta = 0

        for state in self.game.states: # looping through each s in s

            best_action_return = float('-inf') 

            for action in self.game.actions: # looping through each possible action for a given state 

                next_states, game_over, reward, probabilities = self.game.get_next_states(action, state)
                
                if game_over:
                    action_expected_return = reward # terminal state reached 
                else:
                    action_expected_return = sum(prob * (reward + self.gamma * self.values.get(next_state, 0)) # Bellman optimailty equation 
                                       for next_state, prob in zip(next_states, probabilities)) 
                
                best_action_return = max(best_action_return, action_expected_return) # selecting the action with the greatest expected return 
            
            delta = max(delta, abs(best_action_return - self.values[state])) # calculating the change in expected return for a given state 
            new_values[state] = best_action_return
    
        return delta, new_values
    
    def value_iteration(self, theta=0.001):

        delta = 0
        new_values = self.values.copy()
        
        while delta >= theta: # performing value iteration until the expected values have sufficiently converged
            delta, new_values = self.perform_single_value_iteration(new_values)

        return delta, new_values # returning the final delta and 'optimal' expected returns
        
    def play(self, state):
       
        # optimal policy extraction

        if state not in self.game.states:
            return None  # simple error handling 
        
        best_action = None
        best_value = float('-inf')
        
        for action in self.game.actions: # looping through all actions available for the inputted state 

            next_states, game_over, reward, probabilities = self.game.get_next_states(action, state)
            
            if game_over:
                action_value = reward # terminal state rearched
            else:

                action_value = sum(prob * self.values.get(next_state, 0)
                                   for next_state, prob in zip(next_states, probabilities) if next_state is not None) # calculating the expected return with the converged V(s) values
            
            if action_value > best_value: # deducing which action produces the highest expected return 
                best_value = action_value
                best_action = action
        
        return best_action if best_action is not None else self.game.actions[0] # returning the action that produces the highest expected return  
        

# a function for simulation the game with a given method agent

def play_game_with_agent(agent, game, verbose=False):
    state = game.reset()
    
    if(verbose): print(f"Testing agent: \n\t{type(agent).__name__}")
    if(verbose): print(f"Starting dice: \n\t{state}\n")
    
    game_over = False
    actions = 0
    while not game_over:
        action = agent.play(state)
        actions += 1
        
        if(verbose): print(f"Action {actions}: \t{action}")
        _, state, game_over = game.roll(action)
        if(verbose and not game_over): print(f"Dice: \t\t{state}")

    if(verbose): print(f"\nFinal dice: {state}, score: {game.score}")
        
  return game.score

# here you can compare models by testing them agaisnt each other

if __name__ == "__main__":
    # random seed makes the results deterministic
    np.random.seed(1)
    
    game = DiceGame()
    
    agent1 = AlwaysHoldAgent(game)
    play_game_with_agent(agent1, game, verbose=True)
    
    print("\n")
    
    agent2 = ValueIterationAgent(game)
    play_game_with_agent(agent2, game, verbose=True)
