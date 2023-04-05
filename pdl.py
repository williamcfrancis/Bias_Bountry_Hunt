#!/usr/bin/python3

#imports section
###
import numpy as np
from sklearn.metrics import mean_squared_error as loss_fn
import os
import dill as pkl
import shutil
import copy
import sys
sys.setrecursionlimit(10000)

class node:
    '''
    Substructure of the UDT class. Stores instructions for predicting on an instance within a level set at a given point in time for the UDT structure.

    Inputs:
        - self: self
        - right_child: next node in UDT sequence (node)
        - node_name: node name (str)
    '''
    def __init__(self, index, right_child=None,left_child = None, node_name='node'):
        # index
        self.index = index

        #node relation
        self.right_child = right_child
        self.left_child = left_child
        self.node_name = node_name

        # node data
        self.data = None

class addGroupNode:
    def __init__(self, index, right_child = None, left_child = None):
        #node location relation
        self.right_child = right_child
        self.left_child = left_child

        self.node_name = 'addGroupNode'

        #group function
        self.index = index

class updateNode:
    def __init__(self, index, right_child = None, left_child = None):
        #node location relation
        self.right_child = right_child
        self.left_child = left_child
        #update indicator
        self.node_name = 'updateNode'
        
        #group to update
        self.index = index

class repairNode:
    def __init__(self, index, pointer, right_child = None, left_child = None):
        #node location relation
        self.right_child = right_child
        self.left_child = left_child

        #update indicator
        self.node_name = 'repairNode'
        
        #group to update
        self.index = index

        # node to point to 
        self.pointer = pointer

        # node data
        self.data = None

class PointerDecisionList:
    '''
    The UDT structure is a list of nodes which each contain conditional statements for predicting on instances within a level set.

    Inputs:
        - self: self
        - T: 
    '''
    def __init__(self, initial_model, x_train, y_train, x_val, y_val, alpha, min_group_size):
        
        #set initialization
        self.head_node = node(lambda x: np.ones(x), initial_model, node_name = 'head node')
        self.tail_node = self.head_node
        self.current_node = self.head_node
        self.node_list = [self.head_node]
        self.min_group_size = min_group_size
        self.head_node_name = 'head node'
        self.initial_model = initial_model
        self.alpha = alpha
        self.updates = 0

        #store data in model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        #initial predictions
        self.train_predictions = initial_model.predict(x_train)
        self.val_predictions = initial_model.predict(x_val)

        self.model_error_train = loss_fn(self.y_train, self.train_predictions)
        self.model_error_val = loss_fn(self.y_val, self.val_predictions)

        self.train_list = [self.model_error_train]
        self.val_list = [self.model_error_val]

        self.group_indices_train = {}
        self.group_indices_val = {}

        self.best_group_predictions_train = {}
        self.best_group_predictions_val = {}

        self.best_group_errors_val = np.array([])
        
        self.best_node_location = {}

        self.group_functions = []
        self.hypothesis_functions = []

    def append_node(self, new_node):
        self.tail_node.right_child = new_node
        new_node.left_child = self.tail_node
        self.tail_node = new_node
        self.node_list.append(new_node)

    def add_group(self, group, hypothesis):
        indices_train = group(self.x_train).astype('bool')
        indices_val = group(self.x_val).astype('bool')
        self.group_indices_train[self.updates] = np.array(indices_train)
        self.group_indices_val[self.updates] = np.array(indices_val)
        self.group_functions.append(group)
        self.hypothesis_functions.append(hypothesis)
        self.append_node(addGroupNode(self.updates))

    def compute_group_errors(self, y_train, y_val):
        #comput group MSE
        self.group_errors_train = []
        self.group_errors_val = []

        for index in range(len(self.group_indices_train)):
            group_indices = self.group_indices_train[index]
            group_mse_train = loss_fn(self.train_predictions[group_indices],y_train[group_indices])
            self.group_errors_train.append(group_mse_train)

        for index in range(len(self.group_indices_val)):
            group_indices = self.group_indices_val[index]
            group_mse_val = loss_fn(self.val_predictions[group_indices],y_val[group_indices])
            self.group_errors_val.append(group_mse_val)
        
        self.group_errors_train = np.array(self.group_errors_train)
        self.group_errors_val = np.array(self.group_errors_val)

    def update_group_predictions(self):

        #add group best predictions node
        update_array = (self.group_errors_val < self.best_group_errors_val)
        for index in range(len(update_array)):
            if update_array[index] == True:
                #not actually best train preds but for naming conventions sake
                self.best_group_predictions_train[index] = self.train_predictions[self.group_indices_train[index]]
                self.best_group_predictions_val[index] = self.val_predictions[self.group_indices_val[index]]
                self.best_node_location[index] = len(self.node_list) - 1
        
        for index in range(len(update_array)):
            if update_array[index] == True:
                self.append_node(updateNode(index))
        #update best group errors
        np.putmask(self.best_group_errors_val, self.best_group_errors_val > self.group_errors_val, self.group_errors_val)
   
    def check_group_violations(self):
        #check for group error violations
        violations_array = ((self.group_errors_val - self.best_group_errors_val) > self.alpha)
        for index in range(len(violations_array)):
            if violations_array[index] == True:
                self.append_node(repairNode(index, self.best_node_location[index]))
                replace_indices = self.group_indices_train[index]
                self.train_predictions[replace_indices] = self.best_group_predictions_train[index]
                replace_indices = self.group_indices_val[index]
                self.val_predictions[replace_indices] = self.best_group_predictions_val[index]
                return True
        return False

    def repair_groups(self, y_train, y_val):
        #check if there are groups to protect
        if len(self.group_indices_val) != 0:
            violated_group = True
            while violated_group == True:
                self.compute_group_errors(y_train, y_val)
                
                self.update_group_predictions()

                violated_group = self.check_group_violations()
        return

    def update(self, group, hypothesis, x_train, y_train, x_val, y_val):
        
        indices_val = group(x_val).astype('bool')

        if indices_val.sum() <= self.min_group_size:
            return False

        model_error_val = loss_fn(y_val[indices_val], self.val_predictions[indices_val])
        
        hypothesis_preds_val = hypothesis(x_val[indices_val])

        hypothesis_error_val = loss_fn(y_val[indices_val], hypothesis_preds_val)
        if ( ( (indices_val.sum()/len(indices_val)) *(model_error_val - hypothesis_error_val) ) > self.alpha):

            #metrics for paper, lightweight version would not require the following
            indices_train = group(x_train).astype('bool')
            hypothesis_preds_train = hypothesis(x_train[indices_train])
            self.add_group(group, hypothesis)
            self.append_node(node(self.updates, right_child=None, node_name='node'))
            self.best_node_location[self.updates] = len(self.node_list) - 1

            self.train_predictions[indices_train] = hypothesis_preds_train
            self.val_predictions[indices_val] = hypothesis_preds_val
            self.best_group_errors_val = np.append(self.best_group_errors_val, loss_fn(self.y_val[indices_val], self.val_predictions[indices_val]))
            self.best_group_predictions_train[self.updates] = self.train_predictions[indices_train]
            self.best_group_predictions_val[self.updates] = self.val_predictions[indices_val]
            self.repair_groups(y_train, y_val)

            self.train_list.append(loss_fn(self.y_train, self.train_predictions))
            self.val_list.append(loss_fn(self.y_val, self.val_predictions))

            self.updates += 1
            print("Update Accepted!")
            return True
        
        print("Update Rejected!")
        return False


    def predict(self, X):
        
        predictions = np.array([None]*len(X), dtype = float)

        current_indices_allowable = np.ones(len(X)).astype('bool')

        group_indices_X = {}

        self.current_node = self.tail_node
        
        data_empty = np.zeros(len(X)).astype('bool')

        # get to the first non group/update node. 
        while self.current_node.node_name in ['addGroupNode', 'updateNode']:
            self.current_node = self.current_node.left_child
            continue

        # while self.current_node != self.head_node and (has_prediction == 0).any():
        while self.current_node != self.head_node:
            # check if it is a node we don't care about
            if self.current_node.node_name in ['addGroupNode', 'updateNode']:
                self.current_node = self.current_node.left_child
                continue
            
            # get node group indices
            try: 
                group_indices = group_indices_X[self.current_node.index]
            except:
                group = self.group_functions[self.current_node.index]
                group_indices = group(X).astype('bool')

            if type(self.current_node.data) == type(None):
                data_bool = data_empty
            else:
                data_bool = self.current_node.data.astype('bool')

            current_indices_allowable = current_indices_allowable | data_bool

            if self.current_node.node_name == 'repairNode':
                forward_indices = current_indices_allowable & group_indices 
                if type(self.node_list[self.current_node.pointer].data) == type(None):
                    self.node_list[self.current_node.pointer].data = forward_indices
                else:
                    self.node_list[self.current_node.pointer].data = (self.node_list[self.current_node.pointer].data | forward_indices)
                current_indices_allowable[forward_indices] = False
            else:
                update_indices = group_indices & current_indices_allowable
                try:
                    predictions[update_indices] = self.hypothesis_functions[self.current_node.index](X[update_indices])
                    current_indices_allowable[update_indices] = False
                except:
                    pass
                

            self.current_node.data = None

            self.current_node = self.current_node.left_child

        if (current_indices_allowable).any():
            predictions[current_indices_allowable] = self.initial_model.predict(X[current_indices_allowable])
        
        return np.array(predictions, dtype = float)

    def track(self, X, y):
        
        error_list = []

        self.current_node = self.head_node
        predictions = self.initial_model.predict(X)     

        best_group_predictions = np.empty((0,len(predictions)), float)
        group_list = np.empty((0,len(predictions)), bool)

        if self.current_node.right_child != None:
            self.current_node = self.current_node.right_child
        else:
            return predictions

        #traverse down UDT
        while self.current_node != None:
            print(f'Node: {self.current_node.node_name}')
            print(f'Error: {loss_fn(y, predictions)}')
            #group prediction updates
            if 'updateNode' == self.current_node.node_name:
                best_group_predictions[self.current_node.index] = copy.deepcopy(predictions)
                self.current_node = self.current_node.right_child
                continue

            if 'addGroupNode' == self.current_node.node_name:
                group_list = np.vstack([group_list, np.array(self.group_functions[self.current_node.index](X))])
                self.current_node = self.current_node.right_child
                continue

            #group prediction repairs
            if 'repairNode' == self.current_node.node_name:
                group_indices = group_list[self.current_node.index]
                predictions[group_indices] = best_group_predictions[self.current_node.index][group_indices]
                self.current_node = self.current_node.right_child
                error_list[-1] = (loss_fn(y, predictions))
                continue
        
            index_updates = self.group_functions[self.current_node.index](X)
            
            if index_updates.sum() == 0:
                self.current_node = self.current_node.right_child
                continue

            node_predictions = self.hypothesis_functions[self.current_node.index](X[index_updates])

            np.put(predictions, np.where(index_updates == 1), node_predictions)

            best_group_predictions = np.vstack([best_group_predictions, copy.deepcopy(predictions)])

            self.current_node = self.current_node.right_child

            error_list.append(loss_fn(y, predictions))

        return error_list

    def save_model(self, directory_name = 'PDL'):
        if os.path.exists(directory_name):
            shutil.rmtree(directory_name)
        os.mkdir(directory_name)
        os.mkdir(f'{directory_name}/groups')
        os.mkdir(f'{directory_name}/hypotheses')

        for index, group in enumerate(self.group_functions): 
            pkl.dump(group, open(f'{directory_name}/groups/g{index}.pkl', 'wb'))

        for index, hypothesis in enumerate(self.hypothesis_functions):
            pkl.dump(hypothesis, open(f'{directory_name}/hypotheses/h{index}.pkl', 'wb'))

        delattr(self, "group_functions")
        delattr(self, "hypothesis_functions")

        pkl.dump(self, open(f'{directory_name}/model.pkl', 'wb'))

        self.reload_functions(directory_name)

        return
    
    def reload_functions(self, directory_name = 'PDL'):
        self.group_functions = []
        self.hypothesis_functions = []

        for index in range(self.updates):
            self.group_functions.append(pkl.load(open(f'{directory_name}/groups/g{index}.pkl', 'rb')))
            self.hypothesis_functions.append(pkl.load(open(f'{directory_name}/hypotheses/h{index}.pkl', 'rb')))

        return 
