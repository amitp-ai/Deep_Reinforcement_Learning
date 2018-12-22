################################# MCTS #######################################

# Version 1 (mcts.ai)

from math import *
import random

# This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
# The function UCT(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a 
# state.GetRandomMove() or state.DoRandomRollout() function.
# 
# Example GameState classes for Nim, OXO and Othello are included to give some idea of how you
# can write your own GameState use UCT in your 2-player game. Change the game to be played in 
# the UCTPlayGame() function at the bottom of the code.
# 
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
# 
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
# 
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai


class OXOState(object):
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement UCT in any 2-player complete information deterministic 
        zero-sum game, although they can be enhanced and made quicker, for example by using a 
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1 and 2.
    """
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
    """
    def __init__(self):
        self.playerJustMoved = 2 # At the root pretend the player just moved is p2 - p1 has the first move
        self.board = [0,0,0,0,0,0,0,0,0] # 0 = empty, 1 = player 1, 2 = player 2
        
    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OXOState()
        st.playerJustMoved = self.playerJustMoved
        st.board = self.board[:]
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        assert move >= 0 and move <= 8 and move == int(move) and self.board[move] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[move] = self.playerJustMoved
        
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return [i for i in range(9) if self.board[i] == 0] #empty spots are initialized to 0 (which is printed as '.' by design, see __repr__ function)
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        for (x,y,z) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
            if self.board[x] == self.board[y] == self.board[z]:
                if self.board[x] == playerjm:
                    return 1.0
                elif self.board[x] == 3-playerjm:
                    return 0.0
        if self.GetMoves() == []: return 0.5 # draw
        return None #if it comes here then the game is still undecided

    def __repr__(self):
        s= ""
        for i in range(9): 
            s += ".XO"[self.board[i]]
            if i % 3 == 2: s += "\n"
        return s


class Node(object):
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = [] #all the explored children
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves() # future child nodes
        self.playerJustMoved = state.playerJustMoved # the only part of the state that the Node needs later
        
    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + sqrt(2*log(self.visits)/c.visits))[-1]
        return s
    
    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move = m, parent = self, state = s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n
    
    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
             s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
             s += str(c) + "\n"
        return s

def UCT(rootstate, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""
        #Note: node.childNodes include all the explored children. Unexplored children are not in it.

    if rootstate.GetResult(1) is not None: #it should never come here as the UCT function will not be called if it is
        raise ValueError("Game Has Ended!")


    rootnode = Node(state = rootstate)
    #node is just a pointer to rootnode. So as node is changed, rootnode is changed too. (before node is assigned to a different object)
    #this is actually how the search tree is built during each iteration.

    for i in range(itermax):
        node = rootnode #node is just a pointer to rootnode. So as node is changed, rootnode is changed too. (before node is assigned to a different object)
        state = rootstate.Clone() #state is updated in place when executing the state.domove() method below
        
        #print('at beginning', node, '\t', rootnode)
        # Select
        while node.untriedMoves == [] and node.childNodes != [] and state.GetResult(1) is None: # node is fully expanded and non-terminal and game has not yet ended
            node = node.UCTSelectChild() #only updates node and not rootnode
            state.DoMove(node.move) #this updates the variable state in place when executing the state.domove() method
        #print('after select', node, '\t', rootnode)

        # Expand
        if node.untriedMoves != [] and node.childNodes == [] and state.GetResult(1) is None: # if we can expand (i.e. state/node is non-terminal) and game has not yet ended
            m = random.choice(node.untriedMoves)  #randomly pick an unexplored child
            state.DoMove(m) #this updates the variable state in place when executing the move
            node = node.AddChild(m,state) # add child and descend tree. node.addchild() updates node in place. It changes rootnode only if 'select' module was not executed
            #node.addchild() also returns a different node which is assigned to variable 'node.' There after node is now not poining to rootnode. They are different.
        #print('Node children: \n'.format(node.ChildrenToString()))
        #print('after expand', node, '\t', rootnode)

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        # Rollout starts from the selected/expanded node (as state is updated after state.DoMove() in select and expand sections)
        while state.GetMoves() != [] and state.GetResult(1) is None: # while state is non-terminal and game has not yet ended
            m = random.choice(state.GetMoves())
            state.DoMove(m) #this updates the variable state in place when executing the move
        #rollout does not change node or rootnode
        #print('after rollout', node, '\t', rootnode)

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode
        #print('after backprop', node, '\t', rootnode)
        #after backpropagation, node points to None

    # Output some information about the tree - can be omitted
    if (verbose): print(rootnode.TreeToString(0))
    else: print(rootnode.ChildrenToString())

    return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited



def UCTPlayGame():
    """ Play a sample game between two UCT players where each player gets a different number 
        of UCT iterations (= simulations = tree nodes).
    """
    state = OXOState() # uncomment to play OXO
    while (state.GetMoves() != [] and state.GetResult(1) is None): #doesn't matter for which player 1 or 2
        print(str(state))
        if state.playerJustMoved == 1:
            print('Next Player is 2')
            m = UCT(rootstate = state, itermax = 1000, verbose = False) # play with values for itermax and verbose = True
            #m = random.choice(state.GetMoves())
        else:
            print('Next Player is 1')
            m = UCT(rootstate = state, itermax = 100, verbose = False)
            #m = random.choice(state.GetMoves())
        print("Best Move: " + str(m) + "\n")
        state.DoMove(m)
    if state.GetResult(state.playerJustMoved) == 1.0:
        print("Player " + str(state.playerJustMoved) + " wins!")
    elif state.GetResult(state.playerJustMoved) == 0.0:
        print("Player " + str(3 - state.playerJustMoved) + " wins!")
    else: print("Nobody wins!")



if __name__ == "__main__":
    """ Play a single game to the end using UCT for both players. 
    """
    UCTPlayGame()


# Version 2
'''
# Pseudo Code
def monte_carlo_tree_search(root):
    while resources_left(time, computational power):
        leaf = traverse(root) # leaf = unvisited node 
        simulation_result = rollout(leaf)
        backpropagate(leaf, simulation_result)
    return best_child(root)

def traverse(node):
    while fully_expanded(node):
        node = best_uct(node)
    return pick_univisted(node.children) or node # in case no children are present / node is terminal 

def rollout(node):
    while non_terminal(node):
        node = rollout_policy(node)
    return result(node) 

def rollout_policy(node):
    return pick_random(node.children)

def backpropagate(node, result):
   if is_root(node) return 
   node.stats = update_stats(node, result) 
   backpropagate(node.parent)

def best_child(node):
    pick child with highest number of visits
'''

'''
import numpy as np
from collections import defaultdict
from games.tictactoe import *
from games.common import TwoPlayersGameState

class MonteCarloTreeSearchNode:

    def __init__(self, state: TwoPlayersGameState, parent = None):
        self.state = state
        self.parent = parent
        self.children = []

    @property
    def untried_actions(self):
        raise NotImplemented()

    @property
    def q(self):
        raise NotImplemented()

    @property
    def n(self):
        raise NotImplemented()

    def expand(self):
        raise NotImplemented()

    def is_terminal_node(self):
        raise NotImplemented()

    def rollout(self):
        raise NotImplemented()

    def backpropagate(self, reward):
        raise NotImplemented()


    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param = 1.4):
        choices_weights = [
            (c.q / (c.n)) + c_param * np.sqrt((2 * np.log(self.n) / (c.n)))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):        
        return possible_moves[np.random.randint(len(possible_moves))]

class TwoPlayersGameMonteCarloTreeSearchNode(MonteCarloTreeSearchNode):

    def __init__(self, state: TwoPlayersGameState, parent):
        super(TwoPlayersGameMonteCarloTreeSearchNode, self).__init__(state, parent)
        self._number_of_visits = 0.
        self._results = defaultdict(int)

    @property
    def untried_actions(self):
        if not hasattr(self, '_untried_actions'):
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    @property
    def q(self):
        wins = self._results[self.parent.state.next_to_move]
        loses = self._results[-1 * self.parent.state.next_to_move]
        return wins - loses

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = TwoPlayersGameMonteCarloTreeSearchNode(next_state, parent = self)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

class MonteCarloTreeSearch:

    def __init__(self, node: MonteCarloTreeSearchNode):
        self.root = node


    def best_action(self, simulations_number):
        for _ in range(0, simulations_number):            
            v = self.tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        # exploitation only
        return self.root.best_child(c_param = 0.)


    def tree_policy(self):
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node


'''
