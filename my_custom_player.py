
from sample_players import DataPlayer
#import numpy as np
from collections import defaultdict
import random, time
from isolation import isolation
import sys, math

class MCTSNode():
    """
    Monte Carlo Tree Search node class
    """
    
    def __init__(self, state: isolation, action=None, parent=None):
        '''
        @param state: Game state included in this node
        @param parent: Parent node for current node  
        '''
        
        self.state=state
        self.parent=parent
        self.children=[]
        self.action=action  # The action that led to this state from parent. Default is None (as in root node)
        #self._number_of_visits = 0.
        self.q=0
        self.n=0
        self._results = defaultdict(int)
        self.untried_actions=state.actions()
        
        #print("MCTSNode state "+str(self.state))
        #print("MCTSNode parent "+str(self.parent))
        #print("MCTSNode children "+str(self.children))
        #print("MCTSNode number of visits "+str(self._number_of_visits))
        #print("MCTSNode results "+str(self._results))
        
    def is_fully_expanded(self):
        #print("is_fully_expanded "+str(len(self.untried_actions) == 0))
        return len(self.untried_actions) == 0
    
    def best_child(self, c_param = 1.4):
        #print("best_child "+str(self.children[np.argmax([(c.q / (c.n)) + c_param * np.sqrt((2 * np.log(self.n) / (c.n)))                            for c in self.children])]))
        #choices_weights = [(c.q / (c.n)) + c_param * np.sqrt((2 * np.log(self.n) / (c.n))) for c in self.children]
        choices_weights = [(c.q / (c.n)) + c_param * math.sqrt(2*math.log(self.n) / (c.n)) \
                           for c in self.children]
        #return self.children[np.argmax(choices_weights)]
        return self.children[choices_weights.index(max(choices_weights))]
    
    def max_q_child(self):
        choices_weights = [c.q for c in self.children]
        return self.children[choices_weights.index(max(choices_weights))]
    
    def rollout_policy(self,possible_moves):
        #print("rollout_policy "+str(random.choice(possible_moves)))
        return random.choice(possible_moves)
    '''
    @property
    def untried_actions(self):
        if not hasattr(self, '_untried_actions'):
            self._untried_actions = self.state.actions()
        print("untried_actions: "+str(self._untried_actions))
        return self._untried_actions    
        
    @property
    def q(self):
        wins = self._results[self.state.player()]
        loses = self._results[1-self.state.player()]
        #print("q wins "+str(wins)+", losses "+str(loses)+", value "+str(wins-loses))
        return wins - loses    
    
    @property
    def n(self):
        #print("n "+str(self._number_of_visits))
        return self._number_of_visits
    '''
    
    def expand(self):
        #print("expand ")
        res_action=random.choice(self.untried_actions)
        #res_action=minimax(self.state, 3)
        self.untried_actions.remove(res_action)
        next_state=self.state.result(res_action)
        child_node = MCTSNode(next_state, action=res_action, parent = self)
        self.children.append(child_node)
        #print(str(child_node))
        return child_node
    
    def is_terminal_node(self):
        #print("is_terminal_node "+str(self.state.terminal_test()))
        return self.state.terminal_test()
    
    def rollout(self,player_id):
        #print("rollout")
        current_rollout_state = self.state
        while not current_rollout_state.terminal_test():
            current_rollout_state = current_rollout_state.result(self.rollout_policy(current_rollout_state.actions()))
            #current_rollout_state = self.state.result(minimax(current_rollout_state, depth=3))
        #print("rollout utility "+str(current_rollout_state.utility(player_id)))
        #if current_rollout_state.utility(player_id)==float('inf'):
        if current_rollout_state.utility(1-current_rollout_state.player())==float('inf'):   # Utility with respect to the player of the parent state
            return 1.
        #elif current_rollout_state.utility(player_id)==float('-inf'):
        elif current_rollout_state.utility(1-current_rollout_state.player())==float('-inf'):    # Utility with respect to the player of the parent state
            return -1.
        else:
            return 0
        
    def backpropagate(self, result):
        #print("backpropagate visits "+str(self._number_of_visits+1)+", results["+str(result)+"] "+str(self._results[result]+1))
        self.n += 1.
        self.q += result
        result=-result
        if self.parent:
            self.parent.backpropagate(result)
    
    
            
class MCTSSearch():
    '''
    Perform Monte Carlo Tree Search
    '''
    
    def __init__(self, node: MCTSNode):
        self.root = node
        #if not self.root.parent:
        self.player_id=self.root.state.player()
        self.node_no=1
        #print("Start MCTSearch, root: "+str(self.root)+", player: "+str(self.player_id))
        
    def best_action(self, simulations_time):
        #count=1
        start=time.time()
        allowed_time=math.ceil(simulations_time*0.75)
        #for _ in range(100):
        while (time.time()-start)*1000<=allowed_time:
            #print("best_action loop "+str(count))
            #count+=1
            v = self.tree_policy()
            reward = v.rollout(self.player_id)
            v.backpropagate(reward)
        #res=self.root.best_child(c_param = 0.).action if self.root.children else None
        #res_node=self.root.max_q_child() if self.root.children else None
        res_node=self.root.best_child(c_param = 0.5)
        res_act=res_node.action
        #print("best_action node "+str(res_node)+"player: "+str(self.player_id)+", q:"+str(res_node.q)+", visits: "+str(res_node.n))
        return res_act
        #return self.root.best_child(c_param = 1.4)
    
    def tree_policy(self):
        #print("tree_policy with root "+str(self.root))
        current_node = self.root
        #print("tree_policy terminal node "+str(current_node.is_terminal_node()))
        #cnt=1.
        while not current_node.is_terminal_node():
            #print("tree_policy current node "+str(current_node))
            if not current_node.is_fully_expanded():
                new_node=current_node.expand()
                self.node_no+=1
                return new_node
            else:
                current_node = current_node.best_child(c_param=0.5)
                #cnt+=1
        return current_node
    
    

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        #import random
        #self.queue.put(random.choice(state.actions()))
        
        depth=3
        node_no=[]
        try:
            if state.terminal_test() or state.ply_count < 2:
                self.queue.put(random.choice(state.actions()))
            else:
                '''
                for d in range(1,depth+1):
                    res = alpha_beta_search(state, d)
                    #res = minimax(state, d)
                    node_no.append(res[1])
                self.queue.put(res[0])
                print("Node No: "+str(node_no))
                '''
                mcts=MCTSSearch(MCTSNode(state))
                res=mcts.best_action(450)
                #print("node_no: "+str(mcts.node_no))
                if res:
                    self.queue.put(res)
                elif state.actions():
                    self.queue.put(random.choice(state.actions()))
                else:
                    self.queue.put(None)
                #'''
        except:
            print("Unexpected error:"+str(sys.exc_info()[0]))


def alpha_beta_search(state, depth):
    """ Return the move along a branch of the game tree that
    has the best possible value.  A move is a pair of coordinates
    in (column, row) order corresponding to a legal move for
    the searching player.
    
    You can ignore the special case of calling this function
    from a terminal state.
    """
    
    player_id = state.player()    

    alpha = float("-inf")
    beta = float("inf")
    best_score = float("-inf")
    best_move = None
    node_no=0
    
    def min_value(state, alpha, beta, depth,node_no):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        #print("min_value node_no:"+str(node_no))
        node_no+=1

        if state.terminal_test(): return (state.utility(player_id),node_no)
        #print("test node_no: "+str(node_no)+", depth: "+str(depth))
        if depth <= 0: return (score(state, player_id),node_no)
        
        
        v = float("inf")
        for a in state.actions():
            res=max_value(state.result(a), alpha, beta, depth-1,node_no)
            v = min(v, res[0])
            if v <= alpha:
                return (v,res[1])
            beta = min(beta, v)
        return (v,res[1])
    
    def max_value(state, alpha, beta, depth,node_no):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        #print("max_value node_no:"+str(node_no))
        node_no+=1
        if state.terminal_test(): return (state.utility(player_id),node_no)
        if depth <= 0: return (score(state, player_id),node_no)
        
        v = float("-inf")
        for a in state.actions():
            res=min_value(state.result(a), alpha, beta, depth-1,node_no)
            v = max(v, res[0])
            if v >= beta:
                return (v,res[1])
            alpha = max(alpha, v)
        return (v,res[1])

    for a in state.actions():
        #print("action: "+str(a))
        v = min_value(state.result(a), alpha, beta, depth-1,node_no)
        #print("action: "+str(a)+", value: "+str(v[0])+", node_res: "+str(v[1]))
        alpha = max(alpha, v[0])
        if v[0] >= best_score:
            best_score = v[0]
            best_move = a
    return (best_move,v[1])

# AI minimax, from this project itself, in sample_players.py
def minimax(state, depth):

    player_id = state.player()
    
    def min_value(state, depth):
        if state.terminal_test(): return state.utility(player_id)
        if depth <= 0: return score(state, player_id)
        value = float("inf")
        for action in state.actions():
            value = min(value, max_value(state.result(action), depth - 1))
        return value

    def max_value(state, depth):
        if state.terminal_test(): return state.utility(player_id)
        if depth <= 0: return score(state, player_id)
        value = float("-inf")
        for action in state.actions():
            value = max(value, min_value(state.result(action), depth - 1))
        return value

    return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))

def score(state, player_id):
    own_loc = state.locs[player_id]
    opp_loc = state.locs[1 - player_id]
    own_liberties = state.liberties(own_loc)
    opp_liberties = state.liberties(opp_loc)
    #print("test score, state: "+str(len(own_liberties) - len(opp_liberties)))
    return len(own_liberties) - len(opp_liberties)
