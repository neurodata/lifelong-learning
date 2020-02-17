#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.ensemble.forest import _generate_sample_indices

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

class LifelongForest:
    """
    Lifelong Forest class.
    """
    def __init__(self, acorn=None):
        """
        Two major things the Forest Class needs access to:
            1) the realized random forest model (self.models_ is a list of forests, 1 for each task)
            2) old data (to update posteriors when a new task is introduced)
        """
        self.models_ = []
        self.X_ = []
        self.y_ = []
        self.hash_remap_y_fwd = []
        self.hash_remap_y_rev = []
        self.n_tasks = 0
        self.n_classes = None
        
        if acorn is not None:
            np.random.seed(acorn)
    
    def new_forest(self, X, y, n_estimators=200, max_samples=0.32,
                        bootstrap=True, max_depth=30, min_samples_leaf=1,
                        acorn=None):
        """
        Input
        X: an array-like object of features; X.shape == (n_samples, n_features)
        y: an array-like object of class labels; len(y) == n_samples
        n_estimators: int; number of trees to construct (default = 200)
        max_samples: float in (0, 1]: number of samples to consider when 
            constructing a new tree (default = 0.32)
        bootstrap: bool; If True then the samples are sampled with replacement
        max_depth: int; maximum depth of a tree
        min_samples_leaf: int; minimum number of samples in a leaf node
        
        Return
        model: a BaggingClassifier fit to X, y
        """
        
        if X.ndim == 1:
            raise ValueError('1d data will cause headaches down the road')
            
        if acorn is not None:
            np.random.seed(acorn)
            
        self.X_.append(X)

        #Remap the Y class as otherwise class labels are restricted to 0...n
        #There are TWO sets of hashes - with a forward and reverse for each.
        #The following two are the LOCAL model hashes (keeps track of unique classes and remappings ONLY for this model). 
        hash_fwd = {}
        hash_rev = {}
        #Update LOCAL remapping of classes with Labels used for this model
        uniqueset = np.unique(y)
        for cntr in range(0, len(uniqueset)):
            hash_fwd[uniqueset[cntr]] = cntr
            hash_rev[cntr] = uniqueset[cntr]
        #Append the lookup tables just as we do the models
        self.hash_remap_y_fwd.append(hash_fwd)
        self.hash_remap_y_rev.append(hash_rev)
        self.y_.append(y)
            
        n = X.shape[0]
        K = len(np.unique(y))
        
        if self.n_classes is None:
            self.n_classes = K
        
        max_features = int(np.ceil(np.sqrt(X.shape[1])))

        model=BaggingClassifier(DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                         max_features = max_features),
                                  n_estimators=n_estimators,
                                  max_samples=max_samples,
                                  bootstrap=bootstrap)

        model.fit(X, y)
        self.models_.append(model)
        self.n_tasks += 1
        #Remap the Y class as otherwise class labels are restricted to 0...n
        #There are TWO sets of hashes - with a forward and reverse for each.
        #The following two are the GLOBAL model hashes for the LLF (keeps track of ALL UNIQUE classes across ALL models). 
        self.hash_classes_fwd = {}
        self.hash_classes_rev = {}
        cntr = 0

        lblsAll = []
        for classforest in self.y_:
            lblsAll = np.hstack((lblsAll, np.unique(classforest)))
        #Get unique set of labels across ALL CLASSES we've observed thus far and create two new hashes (forward and reverse)
        #This will allow us to encode and decode across the global space with consistency
        uniquelbls = np.unique(lblsAll)
        for classlabel in range(0, len(uniquelbls)):
            self.hash_classes_fwd[uniquelbls[classlabel]] = cntr
            self.hash_classes_rev[cntr] = uniquelbls[classlabel]
            cntr += 1
        
        self.n_classes = len(self.hash_classes_fwd)
        
        return model
    
    
    def _get_leaves(self, estimator):
        """
        Internal function to get leaf node ids of estimator.
        
        Input
        estimator: a fit DecisionTreeClassifier
        
        Return
        leaf_ids: numpy array; an array of leaf node ids
        
        Usage
        _estimate_posteriors(..)
        """
        
        # adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold

        leaf_ids = []
        stack = [(0, -1)] 
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                leaf_ids.append(node_id)

        return np.array(leaf_ids)
    
    
    def _finite_sample_correction(self, class_probs, row_sums):
        """
        An internal function for finite sample correction of posterior estimation.
        
        Input
        class_probs: numpy array; array of posteriors to correct
        row_sums: numpy array; array of partition counts
        
        Output
        class_probs: numpy array; finite sample corrected posteriors
        
        Usage
        _estimate_posteriors(..)
        
        """
    
        where_0 = np.argwhere(class_probs == 0)
        for elem in where_0:
            class_probs[elem[0], elem[1]] = 1 / (2 * row_sums[elem[0], None])
        where_1 = np.argwhere(class_probs == 1)
        for elem in where_1:
            class_probs[elem[0], elem[1]] = 1 - 1 / (2 * row_sums[elem[0], None])
    
        return class_probs
    
    
    def _estimate_posteriors(self, test, representation=0, decider=0, subsample=1, acorn=None):
        """
        An internal function to estimate the posteriors.
        
        Input
        task_number: int; indicates which model in self.model_ to use
        test: array-like; test observation
        in_task: bool; True if test is an in-task observation(s)
        subsample: float in (0, 1]; proportion of out-of-task samples to use to
            estimate posteriors
            
        Return
        probs: numpy array; probs[i, k] is the probability of observation i
            being class k
            
        Usage
        predict(..)
        """
        
        if acorn is not None:
            acorn = np.random.seed(acorn)
            
        if representation==decider:
            in_task=True
        else:
            in_task=False
            
        train = self.X_[decider]
        y = self.y_[decider]
            
        model = self.models_[representation]

        n, d = train.shape
        
        if test.ndim > 1:
            m, d_ = test.shape
        else:
            m = len(test)
            d_ = 1
        #TODO!!!
        class_counts = np.zeros((m, model.n_classes_))
        for tree in model:
            #  NOTE: For in_task, this call to _generate_unsampled_indices doesn't work on latest versions of sklearn.  Defaulting to same sampling for now
            #  get out of bag indicies
            #if in_task:
                #18 prob_indices / n 50 subsample 50 
                #prob_indices = _generate_unsampled_indices(tree.random_state, n, int(n))
                # in_bag_idx = _generate_sample_indices(tree.random_state, n) # this is not behaving as i expected
            #else:
            prob_indices = np.random.choice(range(n), size=int(subsample*n), replace=False)

            leaf_nodes = self._get_leaves(tree)
            unique_leaf_nodes = np.unique(leaf_nodes)

            # get all node counts
            node_counts = tree.tree_.n_node_samples
            # get probs for eval samples
            posterior_class_counts = np.zeros((len(unique_leaf_nodes), model.n_classes_))
            
            for prob_index in prob_indices:
                temp_node = tree.apply(train[prob_index].reshape(1, -1)).item()
                #Here, we want to hash to the local 0...n remapping
                posterior_class_counts[np.where(unique_leaf_nodes == temp_node)[0][0], self.hash_remap_y_fwd[decider][y[prob_index]]] += 1

            # total number of points in a node
            row_sums = posterior_class_counts.sum(axis=1)

            # no divide by zero
            row_sums[row_sums == 0] = 1

            # posteriors
            class_probs = (posterior_class_counts / row_sums[:, None])
            # posteriors with finite sampling correction

            #Only keep the columns for classes observed in this training for THIS model - otherwise finite_sample_correction will break
            class_probs = self._finite_sample_correction(class_probs, row_sums)
            

            # posteriors as a list
            class_probs.tolist()

            partition_counts = np.asarray([node_counts[np.where(unique_leaf_nodes == x)[0][0]] for x in tree.apply(test)])
            # get probability for out of bag samples
            eval_class_probs = [class_probs[np.where(unique_leaf_nodes == x)[0][0]] for x in tree.apply(test)]
            eval_class_probs = np.array(eval_class_probs)
            # find total elements for out of bag samples
            elems = np.multiply(eval_class_probs, partition_counts[:, np.newaxis])
            # store counts for each x (repeat fhis for each tree)
            class_counts += elems
        # calculate p(y|X = x) for all x's
        
        probs = class_counts / class_counts.sum(axis=1, keepdims=True)
        
        #With finite sample correction now already run, we can remap back to our original class space
        #Create as many rows as there are test data... and then an index for each class to store probabilities.
        probsRemapped = np.zeros((len(test), self.n_classes))
        for x in range(0, len(probs)):
            for y in range(0, len(probs[x])):
                #For each class, get the original class that was assigned - at this point, we're back to the original classes...
                origClass = self.hash_remap_y_rev[decider][y]
                #BUT for LLF, we need to calculate a NEW hashed offset 0...n for the classes in the GLOBAL classpace (because argmax is used in predict)
                #this is reverse mapped in the predict function so the code calling this class doesn't have to keep track of the class mappings
                offsetClass = self.hash_classes_fwd[origClass]
                probsRemapped[x][offsetClass] = probs[x][y]
        return probsRemapped


    def predict(self, test, representation=0, decider='all', subsample=1, acorn=None):
        """
        Predicts the class labels for each sample in test.
        
        Input
        test: array-like; either a 1d array of length n_features
            or a 2d array of shape (m, n_features) 
        task_number: int; task number 
        """
        
        sum_posteriors = np.zeros((test.shape[0], self.n_classes))
        
        if representation is 'all':
            for i in range(self.n_tasks):
                #TODO!!!! if the estimate posterior has already been called with the current decider, then used the cached data!!!
                sum_posteriors += self._estimate_posteriors(test,
                                                            i,
                                                            decider,
                                                            subsample,
                                                            acorn)
             
        else:
            sum_posteriors += self._estimate_posteriors(test,
                                                        representation,
                                                        decider,
                                                        subsample,
                                                        acorn)
        
        ret = np.argmax(sum_posteriors, axis=1)
        ans = np.zeros(len(ret))
        #Finally, translate all the argmaxes back to their original class!  estimate_posteriors returns ENCODED classes, so this reverse hash decodes them before returning
        for x in range(0, len(ret)):
            ans[x] = int(self.hash_classes_rev[ret[x]])
        return ans


def generate_2d_rotation(theta=0, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
    
    R = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    
    return R

def generate_3d_rotation(angle_params=[0,0,0], acorn=None):
    if acorn is not None:
            np.random.seed(acorn)

    a,b,c = angle_params[0], angle_params[1], angle_params[2]
    
    R1 = np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a), np.cos(a), 0],
        [0, 0, 1]
    ])
    
    R2 = np.array([
        [np.cos(b), 0, np.sin(b)],
        [0, 1, 0],
        [-np.sin(b), 0, np.cos(b)]
    ])
    
    R3 = np.array([
        [np.cos(c), -np.sin(c), 0],
        [np.sin(c), np.cos(c), 0],
        [0, 0, 1]
    ])
    
    R = R1 @ R2 @ R3 # R3 @ R2 @ R1
    
#     q = np.random.normal(0,1,4)
#     q = q / np.sqrt(np.sum(q**2))
    
#     R1 = np.array([np.cos()])

#     R = np.array([
#         [1 -2*(q[2]**2+q[3]**2), 2*(q[1]*q[2]-q[3]*q[0]), 2*(q[1]*q[3]+q[2]*q[0])],
#         [2*(q[1]*q[2]+q[3]*q[0]), 1-2*(q[1]**2+q[3]**2), 2*(q[2]*q[3]-q[1]*q[0])],
#         [2*(q[1]*q[3] - q[2]*q[0]), 2*(q[2]*q[3]+q[1]*q[0]), 1-2*(q[1]**2+q[2]**2)]
#     ])
    return R

def generate_parity(n, d=2, angle_params=None, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
        
    X = np.random.uniform(-2, 2, size=(10*n, d))
    Y = (np.sum(X > 0, axis=1) % 2 == 0).astype(int)
    
    if d == 2:
        if angle_params is None:
            angle_params = np.random.uniform(0, 2*np.pi)
        R = generate_2d_rotation(angle_params)
        X = X @ R
        inds = (abs(X[:, 0]) < 1) + (abs(X[:, 1]) < 1)
        Y = Y[(abs(X[:, 0]) < 1) * (abs(X[:, 1]) < 1)][:n]
        X = X[(abs(X[:, 0]) < 1) * (abs(X[:, 1]) < 1)][:n]
    fout = open('C:\\LifelongLearning\\lllinputdata' + str(angle_params) + '.csv', 'w')
    for cntr in range(0, len(X)):
        fout.write(str(X[cntr][0]) + ',' + str(X[cntr][1]) + ',' + str(Y[cntr]) + '\n')
    fout.close()
        
    if d==3:
        if angle_params is None:
            pass
        R = generate_3d_rotation(angle_params[0], angle_params[1], angle_params[2])
        X = X @ R
        Y = Y[(abs(X[:, 0]) < 1) * (abs(X[:, 1]) < 1) * (abs(X[:, 2]) < 1)][:n]
        X = X[(abs(X[:, 0]) < 1) * (abs(X[:, 1]) < 1) * (abs(X[:, 2]) < 1)][:n]
        
            
    return X, Y.astype(int)