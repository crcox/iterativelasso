Iterative Lasso
===============
Iterative Lasso is a procedure that implements Lasso[1] in a loop. Each
time Lasso is run, some set of features are identified as informative
and are assigned non-zero weights. Exactly as with a standard regression
model, these weights can be multiplied with the value at each
corresponding feature to generate predictions. If the set of features
selected by Lasso support generalizable predictions about the task of
interest, as determined through cross validation, then those features
are flagged as informative, **and removed from the dataset**. Now, Lasso
is run again on the data, with those previously identified features
excluded. This process is repeated until the solutions obtained by Lasso
no longer support generalizable predictions. The features selected at
each iteration are compiled into a single set. If one is interested in
how well a model can perform if trained on this compilated set of
features, one can use Ridge Regression. See our poster presented at the
22nd meeting of the Cognitive Neurscience Society for a little more
detail and an application[2].

[1] http://statweb.stanford.edu/~tibs/lasso.html

[2] https://github.com/crcox/CNS2015/raw/master/IterativeLassoPoster.pdf

Why use Iterative Lasso?
------------------------
Lasso is great in that one can begin with a dataset in which there are
many more features than observations and obtain a sparse model that
utilizes only a small subset of the total feature set. Lasso strives to
build the best possible model with the fewest features. However, this
quality will often lead Lasso to identify only a small number of the
features that are truly informative. This seems to be especially
problematic when performing binomial classification with logistic Lasso.

Iterative Lasso addresses this problem directly and intuitively. If
Lasso has not recovered all of the informative features, perhaps running
Lasso again will identify some more of them.

Thus, Iterative Lasso is an attempt to improve the hit-rate of Lasso by
repeating the analysis until generalization performance drops to chance.

Why NOT to use Iterative Lasso.
-------------------------------
The goal of Iterative Lasso is to identify more features, **not** to
build a more accurate model. While it is not impossible that a model fit
to the compiled feature set will yield better performance than any of
the individual Lasso iterations, this is by no means expected.

Disclaimer
----------
There are currently no theory or proofs that characterize the Iterative
Lasso procedure or can formally relate it to other methods. It is an
intuitive solution to a practical issue with Lasso that works well in
practice, at least in the context of cognitive neuroscience when applied
to fMRI data.

Install
=======
Clone this repository somewhere on your machine, and add it to your
Matlab path. You will need to do the same for several dependencies.

Dependencies
------------
- glmnet (http://web.stanford.edu/~hastie/glmnet_matlab/)
- (optional) condorutils (https://github.com/crcox/condortools)

Usage
=====
In it's most simple form, Iterative Lasso can be run by providing a data
matrix, `X`, a vector specifying the dependent variable, `Y`, and a
logical matrix that defines how to split up the data for cross
validation. In this matrix, each column corresponds to a cross
validation run, and true values indicate test items. There should be a
row for every observation. So, for instance:

```
1 0 0
1 0 0
0 1 0
0 1 0
0 0 1
0 0 1
```
Specifies that there are 6 observations total, and we will perform
3-fold cross validation. On each cross validation run, 2 observations
will be withheld for testing, as indicated by the true values.

Consider the function definition:

```matlab
function [finalModel,iterModels,finalTune,iterTune] = iterativelasso(X,Y,CVBLOCKS,varargin)
```

There are four outputs:

1. *finalModel* Contains the solution obtained by fitting ridge
   regression to the compiled feature set. Recall that iterative Lasso
   involves running Lasso several times, each time identifying different
   voxels. This finalModel structure contains all the information
   pertaining to a model fit to just that feature set.
2. *iterModels* Contains the solutions for each iteration.
3. *finalTune* Lasso and Ridge regression both involve specifying a free
   parameter that scales how severely to regularize the solution. This
   is found via a nested cross validation procedure. If you would like
   to see or inspect the models fit while tuning this parameter for the
   final model, those data are retured here.
4. *iterTune* This tuning procedure is done for every iteration, and
   those data are returned here.

In reality, you will get a solution for each cross validation run. While
it is natural to average performance over cross validation runs,
combining model solutions across cross validation runs (i.e., the actual
features selected) is not as natural. So all solutions are returned, and
you can do whatever seems sensible to you.

The `varargin` allows for *'key', 'value'* arguments to be passed to the
function. Currently, only two are recognized:

1. 'MaxIter' allows you to specify a number of iterations after which
   itertive Lasso should stop iterating, even if cross validation
   performance has not dropped to chance. This may be useful for testing
   and debugging, but in practice you will probably want to leave the
   default of `Inf`.
2. 'ExpDir' Currently, Iterative Lasso will write out a small `summary.
   json` file that contains information about the number of iterations
that were run and the number of cross validation folds performed. If you
want this file written to some directory other than the current working
directory, you can specify that directory here.

Thus concludes the basic usage of Iterative Lasso. In practice, you
will probably need to do more sophisticated things. I can make the
following recommendations:

Recommended Project Structure
-----------------------------
1. Wrap `iterativelasso()` in a `runIterLasso.m` function or script that
   handled preparing your data for analysis and saving the results to
disk, rather than modifying iterativelasso.m itself.
2. Write `runIterLasso.m` so that it interprets parameter file, that you
   can stick in a folder somewhere and point to. That folder is the
'ExpDir', or 'JobDir'. This is a nice setup in that is is both scalable
and readily generalizes to distributed computing contexts.
