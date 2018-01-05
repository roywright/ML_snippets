import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from math import atan2
from scipy.spatial import ConvexHull
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import confusion_matrix as cm

def permutation_importances(clf, Xtest, ytest):
    '''
    Use testing data to calculate permutation importances for 
    all features of a trained classifier
        - `clf` must be a trained sklearn classifier with `predict_proba` method
        - `Xtest` must be a pandas DataFrame of features, where
        - `ytest` contains the targets (1's and 0's)
    '''
    auc = roc_auc_score(ytest, clf.predict_proba(Xtest)[:,1])
    pimp = []
    for column in Xtest.columns:
        Xtemp = Xtest.copy()
        Xtemp[column] = Xtemp[column].sample(frac = 1.0).values
        pimp.append(auc - roc_auc_score(ytest, clf.predict_proba(Xtemp)[:,1]))
    return pimp

def _std_results_(clf, Xtest, ytest):
    '''Helper function for the seven plotter functions that follow.'''
    return pd.DataFrame({
        'truth' : ytest,
        'pred'  : clf.predict_proba(Xtest)[:,1]
    }).sort_values('pred', ascending = False)

def plot_roc(clf, Xtest, ytest):
    '''Plot the ROC curve and report AUC.'''
    pred = clf.predict_proba(Xtest)[:,1]
    fpr, tpr, _ = roc_curve(ytest, pred)
    plt.plot([0, 1], [0, 1], linestyle='--', c = 'black', lw = .5)
    plt.plot(fpr, tpr, c='red', lw = 3)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(
        'ROC Curve (AUC %.4f)' 
        % roc_auc_score(ytest, pred)
    )
    plt.show()

def plot_prc(
    clf, Xtest, ytest, 
    precision = None, recall = None, threshold = None
):
    '''
    Plot the precision-recall curve.
    If `precision` or `recall` is specified, the best point on the curve
    satisfying that precision or recall will be shown.
    '''
    pr, rc, ts = precision_recall_curve(ytest, clf.predict_proba(Xtest)[:,1])
    pr, rc, ts = list(pr), list(rc), list(ts)
    i = None
    if precision is not None:
        i = ts.index(min(t for t, p in zip(ts, pr) if p > precision))
    elif recall is not None:
        i = ts.index(max(t for t, r in zip(ts, rc) if r > recall))
    elif threshold is not None:
        i = ts.index(max(t for t in ts if t < threshold))
    plt.plot(pr, rc, c='red', lw = 3)
    if i is not None:
        plt.plot([pr[i]], [rc[i]], marker = 'o', color = 'black')
        plt.text(
            pr[i], rc[i], '(%.2f, %.2f)  ' % (pr[i], rc[i]),
            fontdict = {'ha':'right', 'va':'center'}
        )
    plt.grid()
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title('Precision & Recall')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.show()

def plot_gain(clf, Xtest, ytest):
    '''Plot the cumulative gain curve.'''
    results = _std_results_(clf, Xtest, ytest)
    results['rand'] = results.sample(frac = 1.0).truth.values
    results['wiz'] = results.sort_values(
        'truth', ascending = False
    ).truth.values
    x = np.linspace(0, 1, 21)
    y_t = results.truth.sum()
    y_r = [results.head(int(len(results)*p)).rand.sum() / y_t for p in x]
    y_m = [results.head(int(len(results)*p)).truth.sum() / y_t for p in x]
    y_w = [results.head(int(len(results)*p)).wiz.sum() / y_t for p in x]
    plt.plot(x, y_r, x, y_m, x, y_w, lw = 3)
    plt.xlabel('% from top')
    plt.ylabel('% of all positives')
    vals = plt.gca().get_xticks()
    plt.gca().set_xticklabels(['{:3.0f}%'.format(x*100) for x in vals])
    vals = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])
    plt.title('Cumulative Gain')
    plt.legend(['Random', 'Model', 'Perfect'])
    plt.show()

def plot_ks(clf, Xtest, ytest):
    '''Plot the Kolmogorov-Smirnov chart.
    Returns the KS statistic.'''
    x = np.linspace(0, 1, 21)    
    results = _std_results_(clf, Xtest, ytest)
    y_t = results.truth.sum()
    y_m = [results.head(int(len(results)*p)).truth.sum() / y_t for p in x]
    y_f = (results.truth == 0).sum()
    y_n = [
        (results.head(int(len(results)*p)).truth == 0).sum() / y_f 
        for p in x
    ]
    KS = [y_m[i] - y_n[i] for i in range(len(x))]
    KSi = KS.index(max(KS))
    plt.plot(x, y_m, 'b', x, y_n, 'r', lw = 3, zorder = 10)
    plt.plot(
        [x[KSi],x[KSi]], [y_n[KSi], y_m[KSi]], 
        c = 'gray', lw = 5
    )
    plt.text(
        x[KSi] + .02, (y_n[KSi] + y_m[KSi])/2, 
        'KS: %.2f' % max(KS), 
        zorder = 9000,
        fontdict = {'ha': 'left', 'va': 'center', 'rotation': 90}
    )
    plt.xlabel('% from top')
    vals = plt.gca().get_xticks()
    plt.gca().set_xticklabels(['{:3.0f}%'.format(x*100) for x in vals])
    vals = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])
    plt.title('Kolmogorov-Smirnov')
    plt.legend(['% of all positives', '% of all negatives'])
    plt.show()
    return max(KS)

def plot_lift(clf, Xtest, ytest):
    '''Plot the (local) lift chart.'''
    x = np.linspace(0, 1, 21)
    results = _std_results_(clf, Xtest, ytest)
    results['rand'] = results.sample(frac = 1.0).truth.values
    y_r = [1 for p in x[1:]]
    dx = x[1] - x[0]
    y_l = [
        results.iloc[int(len(results)*(p-dx)):int(len(results)*p)].truth.mean() 
        / results.rand.mean() 
        for p in x[1:]
    ]
    plt.plot(x[1:], y_l, lw = 3)
    plt.plot(x[1:], y_r, lw = 3)
    # Find and plot the crossing point
    Qi = min([i for i in range(len(x)-1) if y_l[i] < y_r[i]]) - 1
    if y_l[Qi] > y_l[Qi+1]:
        Qx = x[Qi+2] 
        Qx -= (y_l[Qi+1] - 1) * (x[Qi+2] - x[Qi+1])/(y_l[Qi+1] - y_l[Qi])
    else:
        Qx = x[Qi+1]
    plt.scatter([Qx], [1], c = 'black', zorder = 9001)
    plt.text(
        Qx, .95, 
        '{:3.0f}%'.format((1-Qx)*100),
        fontdict = {'ha': 'right', 'va': 'top'}
    )
    # Finish up
    vals = plt.gca().get_xticks()
    plt.gca().set_xticklabels(['{:3.0f}%'.format((1-x)*100) for x in vals])
    plt.xlabel('Percentile')
    plt.ylabel('Lift at quantile')
    plt.title('Lift Chart')
    plt.legend(['Model', 'Random'])
    plt.show()

def plot_cumlift(clf, Xtest, ytest, show_spf = False):
    '''Plot the cumulative lift chart.
    Returns an approximate lower convex envelope of the graph.'''
    x = np.linspace(0, 1, 21)
    results = _std_results_(clf, Xtest, ytest)
    y_l = [
        results.head(int(len(results)*p)).truth.mean() / results.truth.mean()
        for p in x[1:]
    ]
    hull = ConvexHull([[i,j] for i,j in zip(x[1:], y_l)])
    ihull = np.roll(hull.vertices, -list(hull.vertices).index(0))
    ihull = ihull[:list(ihull).index(max(ihull))+1]
    xhull = [x[1:][i] for i in ihull]
    yhull = [y_l[i] for i in ihull]
    spf = UnivariateSpline(xhull, yhull, k = 1, s = 0, ext = 'const')
    xplot = np.linspace(0, 1, 1000)
    plt.plot(
        x[1:], y_l, 
        linewidth = 1 if show_spf else 3, 
        marker = 'o' if show_spf else None, 
        c = 'blue' if show_spf else None
    )
    plt.xlabel('% from top')
    plt.ylabel('Cumulative lift')
    vals = plt.gca().get_xticks()
    plt.gca().set_xticklabels(['{:3.0f}%'.format(x*100) for x in vals])
    plt.title('Cumulative Lift Chart')
    if show_spf:
        plt.plot(xplot, spf(xplot), lw = 2, c='red', zorder = 42)
        plt.legend(['Actual lift', 'Lower envelope'])
    plt.show()
    return spf

def indiv_plot(clf, Xtest, ytest, width = 12, height = 9, fillfactor = 2500):
    '''
    Visualize classifier performance on an individual level.
    Optional arguments:
        `width` and `height` determine the visual's shape, 
        `fillfactor` (optional) controls size of markers
    '''
    dims = width, height
    plt.rcParams['figure.figsize'] = dims
    plt.rcParams['font.size'] = sum(dims) // 2
    results_xy = _std_results_(clf, Xtest, ytest).reset_index()[['truth']]
    rowlen = int((dims[0] / dims[1] * len(results_xy)) ** .5)
    results_xy['x'] = results_xy.index % rowlen + 1
    results_xy['y'] = results_xy.index // rowlen + 1
    collen = results_xy.y.max()
    results_xy['y'] = collen - results_xy.y + 1
    results_xy.plot(
        'x','y',
        c = ['blue' if t == 1 else 'red' for t in results_xy.truth], 
        kind = 'scatter',
        marker = 's',
        s = dims[0]*dims[1]*2500 / (rowlen*collen)
    )
    for i in range(12):
        _, j, k = results_xy.iloc[i].values
        plt.text(
            j, k, str(i+1) if i < 9 else '...', 
            fontdict = {
                'color':'white', 'weight':'bold', 
                'ha':'center', 'va':'center'
            }
        )
    plt.text(
        results_xy.iloc[-1].x + .5, 1, '← lowest-ranked', 
        fontdict = {
            'color':'black', 'weight':'bold', 
            'ha':'left', 'va':'center'
        }
    )
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
class HypothesisTest(object):
    """
    Represents a hypothesis test.
    Adapted from Allen Downey's work
    e.g. https://github.com/AllenDowney/ThinkStats2/blob/master/code
    """

    def __init__(self, data):
        """Initializes.

        data: data in whatever form is relevant
        """
        self.data = data
        self.MakeModel()
        self.actual = self.TestStatistic(data)
        self.test_stats = None

    def PValue(self, iters=1000):
        """Computes the distribution of the test statistic and p-value.

        iters: number of iterations

        returns: float p-value
        """
        self.test_stats = np.array([self.TestStatistic(self.RunModel()) 
                                       for _ in range(iters)])

        count = sum(self.test_stats >= self.actual)
        return count / iters

    def MaxTestStat(self):
        """Returns the largest test statistic seen during simulations.
        """
        return max(self.test_stats)

    def PlotHist(self, label=None):
        """Draws a Cdf with vertical lines at the observed test stat.
        """
        ys, xs, patches = plt.hist(self.test_stats)
        plt.vlines(self.actual, 0, max(ys), linewidth=3, color='black')
        plt.xlabel('test statistic')
        plt.ylabel('count')
        plt.show()

    def TestStatistic(self, data):
        """Computes the test statistic.

        data: data in whatever form is relevant        
        """
        raise UnimplementedMethodException()

    def MakeModel(self):
        """Build a model of the null hypothesis.
        """
        pass

    def RunModel(self):
        """Run the model of the null hypothesis.

        returns: simulated data
        """
        raise UnimplementedMethodException()

class DiffAUCsPermute(HypothesisTest):
    """Tests a difference in AUCs by permutation."""

    def TestStatistic(self, data):
        """Computes the test statistic.

        data: two ranked binary target lists        
        """
        group1, group2 = data
        n1, n2 = len(group1), len(group2)
        pred1 = [i/n1 for i in range(n1, 0, -1)]        
        pred2 = [i/n2 for i in range(n2, 0, -1)]        
        test_stat = abs(
            roc_auc_score(group1, pred1) 
            - roc_auc_score(group2, pred2)
        )
        return test_stat

    def MakeModel(self):
        """Build a model of the null hypothesis.
        """
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def RunModel(self):
        """Run the model of the null hypothesis.

        returns: simulated data
        """
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data
    
def TestAUCs(clf1, clf2, Xtest, ytest, iters = 1000):
    '''Run a two-AUC significance test for two given classifiers.'''
    isort = clf1.predict_proba(Xtest)[:,1].argsort()[::-1]
    y1 = ytest[isort]
    isort = clf2.predict_proba(Xtest)[:,1].argsort()[::-1]
    y2 = ytest[isort]
    ht = DiffAUCsPermute([y1, y2])
    p_value = ht.PValue(iters = iters)
    print('Diff. in AUCs =', ht.actual)
    print('      P-value =', p_value)
    ht.PlotHist()    

def cm_labeled(clf, Xtest, ytest, threshold = 0.5):
    '''Show a nicely-labeled version of the confusion matrix.'''
    return pd.DataFrame(
        cm(ytest, clf.predict_proba(Xtest)[:,1] >= threshold, labels = [1,0]), 
        columns = ['Predicted positive', 'Predicted negative'],
        index   = ['Actually positive',  'Actually negative']
    )
