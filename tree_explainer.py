# -*- coding: utf-8 -*-
import numpy as np
import sklearn

from sklearn.ensemble.forest import ForestClassifier, ForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, _tree
from distutils.version import LooseVersion
if LooseVersion(sklearn.__version__) < LooseVersion("0.17"):
    raise Exception("treeinterpreter requires scikit-learn 0.17 or later")


def _get_tree_paths(tree, node_id, depth=0):
    """
    Returns all paths through the tree as list of node_ids
    """
    if node_id == _tree.TREE_LEAF:
        raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    if left_child != _tree.TREE_LEAF:
        left_paths = _get_tree_paths(tree, left_child, depth=depth + 1)
        right_paths = _get_tree_paths(tree, right_child, depth=depth + 1)

        for path in left_paths:
            path.append(node_id)
        for path in right_paths:
            path.append(node_id)
        paths = left_paths + right_paths
    else:
        paths = [[node_id]]
    return paths


def _predict_tree(model, X, joint_contribution=False):
    """
    For a given DecisionTreeRegressor, DecisionTreeClassifier,
    ExtraTreeRegressor, or ExtraTreeClassifier,
    returns a triple of [prediction, bias and feature_contributions], such
    that prediction ≈ bias + feature_contributions.
    Also returns the relevant thresholds...
    """
    leaves = model.apply(X)
    paths = _get_tree_paths(model.tree_, 0)
    thresh_list = list(model.tree_.threshold)  # RW ADDED

    for path in paths:
        path.reverse()

    leaf_to_path = {}
    #map leaves to paths
    for path in paths:
        leaf_to_path[path[-1]] = path         
    
    # remove the single-dimensional inner arrays
    values = model.tree_.value.squeeze()
    # reshape if squeezed into a single float
    if len(values.shape) == 0:
        values = np.array([values])
    if isinstance(model, DecisionTreeRegressor):
        biases = np.full(X.shape[0], values[paths[0][0]])
        line_shape = X.shape[1]
    elif isinstance(model, DecisionTreeClassifier):
        # scikit stores category counts, we turn them into probabilities
        normalizer = values.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        values /= normalizer

        biases = np.tile(values[paths[0][0]], (X.shape[0], 1))
        line_shape = (X.shape[1], model.n_classes_)
    direct_prediction = values[leaves]
    
    
    #make into python list, accessing values will be faster
    values_list = list(values)
    feature_index = list(model.tree_.feature)
    
    contributions = []
    thresholds = []   # RW ADDED
    if joint_contribution:
        for row, leaf in enumerate(leaves):
            path = leaf_to_path[leaf]
            thresholds.append([None] * X.shape[1])   # RW ADDED
            
            path_features = set()
            contributions.append({})
            for i in range(len(path) - 1):
                path_features.add(feature_index[path[i]])
                contrib = values_list[path[i+1]] - \
                         values_list[path[i]]
                #path_features.sort()
                contributions[row][tuple(sorted(path_features))] = \
                    contributions[row].get(tuple(sorted(path_features)), 0) + contrib
                if contrib[1] > 0:   # RW ADDED
                    thresholds[row][feature_index[path[i]]] = thresh_list[path[i]]   
                    
        return direct_prediction, biases, contributions, thresholds  # RW ADDED
        
    else:

        for row, leaf in enumerate(leaves):
            thresholds.append([None] * X.shape[1])   # RW ADDED

            for path in paths:
                if leaf == path[-1]:
                    break
            
            contribs = np.zeros(line_shape)
            for i in range(len(path) - 1):
                
                contrib = values_list[path[i+1]] - \
                         values_list[path[i]]
                contribs[feature_index[path[i]]] += contrib
                if contrib[1] > 0:   # RW ADDED
                    thresholds[row][feature_index[path[i]]] = thresh_list[path[i]]   
                    
            contributions.append(contribs)
    
        return direct_prediction, biases, np.array(contributions), thresholds
                                                                  # RW ADDED


def _predict_forest(model, X, joint_contribution=False):
    """
    For a given RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, or ExtraTreesClassifier returns a triple of
    [prediction, bias and feature_contributions], such that prediction ≈ bias +
    feature_contributions.
    Also returns thresholds...
    """
    biases = []
    contributions = []
    predictions = []
    thresholds = []  # RW ADDED
    
    if joint_contribution:
        
        for tree in model.estimators_:
            pred, bias, contribution, th = _predict_tree(tree, X, joint_contribution=joint_contribution)

            biases.append(bias)
            contributions.append(contribution)
            predictions.append(pred)
            thresholds.append(th)
        
        total_contributions = []
        
        for i in range(len(X)):
            contr = {}
            for j, dct in enumerate(contributions):
                for k in set(dct[i]).union(set(contr.keys())):
                    contr[k] = (contr.get(k, 0)*j + dct[i].get(k,0) ) / (j+1)

            total_contributions.append(contr)    
            
        for i, item in enumerate(contribution):
            total_contributions[i]
            sm = sum([v for v in contribution[i].values()])
                

        
        return (np.mean(predictions, axis=0), np.mean(biases, axis=0),
            total_contributions)
    else:
        for tree in model.estimators_:
            pred, bias, contribution, th = _predict_tree(tree, X)  # RW ADDED

            biases.append(bias)
            contributions.append(contribution)
            predictions.append(pred)
            thresholds.append(th)  # RW ADDED
        
        return (
            np.mean(predictions, axis=0), 
            np.mean(biases, axis=0),
            np.mean(contributions, axis=0), 
            [[  # RW ADDED
                list(set(thresholds[t][s][f] for t in range(len(thresholds)) 
                         if thresholds[t][s][f] is not None)) 
                for f in range(X.shape[1])
            ] for s in range(X.shape[0])] 
        )


def predict(model, X, joint_contribution=False):
    """ Returns a triple (prediction, bias, feature_contributions), such
    that prediction ≈ bias + feature_contributions.
    Parameters
    ----------
    model : DecisionTreeRegressor, DecisionTreeClassifier,
        ExtraTreeRegressor, ExtraTreeClassifier,
        RandomForestRegressor, RandomForestClassifier,
        ExtraTreesRegressor, ExtraTreesClassifier
    Scikit-learn model on which the prediction should be decomposed.

    X : array-like, shape = (n_samples, n_features)
    Test samples.
    
    joint_contribution : boolean
    Specifies if contributions are given individually from each feature,
    or jointly over them

    Returns
    -------
    decomposed prediction : quadruple of
    * prediction, shape = (n_samples) for regression and (n_samples, n_classes)
        for classification
    * bias, shape = (n_samples) for regression and (n_samples, n_classes) for
        classification
    * contributions, If joint_contribution is False then returns and  array of 
        shape = (n_samples, n_features) for regression or
        shape = (n_samples, n_features, n_classes) for classification, denoting
        contribution from each feature.
        If joint_contribution is True, then shape is array of size n_samples,
        where each array element is a dict from a tuple of feature indices to
        to a value denoting the contribution from that feature tuple.
    * thresholds...
    """
    # Only single out response variable supported,
    if model.n_outputs_ > 1:
        raise ValueError("Multilabel classification trees not supported")

    if (isinstance(model, DecisionTreeClassifier) or
        isinstance(model, DecisionTreeRegressor)):
        return _predict_tree(model, X, joint_contribution=joint_contribution)
    elif (isinstance(model, ForestClassifier) or
          isinstance(model, ForestRegressor)):
        return _predict_forest(model, X, joint_contribution=joint_contribution)
    else:
        raise ValueError("Wrong model type. Base learner needs to be a "
                         "DecisionTreeClassifier or DecisionTreeRegressor.")

        
        
def predict_explain(rf, X, num_reasons = 2):  # RW ADDED (ENTIRE METHOD)
    '''
    Produce scores and explanations for an entire data frame.
        * `rf` is a RandomForestClassifier,
        * `X` is the features data frame,
        * `num_reasons` (default 2) is the number of 
          reasons/explanations to be produced for each row.
    '''    
    # Prepare the structure to be returned    
    pred_ex = X[[]]
    
    # Get scores and feature contributions from a tree interpreter
    pred, _, contrib, thresh = predict(rf, X)
    pred = pred[:,1]    
    pred_ex['SCORE'] = pred
    
    # Reformat the contributions: the final result is a list of the 
    # top `num_reasons` contributors for each data point and score
    contrib = [[c[1] for c in l] for l in contrib]
    contrib = [[
        tup for tup in
        sorted(enumerate(c), key = lambda tup: -tup[1])[:num_reasons]
        if tup[1] > 0
    ] for c in contrib]

    # Find the reasons/explanations
    for n in range(num_reasons):
        reason = []
        for i, c in enumerate(contrib):
            if len(c) > n:
                line_thresh = thresh[i][c[n][0]]
                name = X.columns[c[n][0]] # The feature's name
                val = X.iloc[i, c[n][0]]  # The feature's value in this row
                
                # Get the lower and upper thresholds that contributed to the
                # score of the current row
                low = max([t for t in line_thresh if t < val], default = None)
                high = min([t for t in line_thresh if t > val], default = None)
                
                # Formulate the reason/explanation as a human-readable string
                if high is None and low is None: reason.append('%s' % name)
                elif high is None: reason.append('%s > %.2f' % (name, low))
                elif low is None: reason.append('%s <= %.2f' % (name, high))
                else: reason.append('%.2f < %s <= %.2f' % (low, name, high))
            else:
                reason.append('')
                
        pred_ex['REASON%d' % (n+1)] = reason
        
    return pred_ex
