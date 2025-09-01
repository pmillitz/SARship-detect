'''
File name: utilities.py

Authors: Du Huynh
Date created: December 6th 2024
Last modified: July 5th 2025

This Python file contains some generic utility functions for model saving
and loading, for displaying and for saving figures (as pdf files for inclusion
in research papers), etc.
'''

import numpy as np
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# -----------------------------------------------------------------------

def save_fig(fig_name, fig_extension="pdf", resolution=300, fig=None):
    '''
    This function would save the current figure in pdf format to the file
    named fig_name.pdf.
    
    Args:
        fig_name: Name for the output file (without extension)
        fig_extension: File extension (default: "pdf")
        resolution: DPI for the output (default: 300)
        fig: Optional figure object to save (if None, saves current figure)
    '''
    filename = f"{fig_name}.{fig_extension}"
    if fig is not None:
        fig.savefig(filename, format=fig_extension, dpi=resolution,
                   bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(filename, format=fig_extension, dpi=resolution,
                   bbox_inches='tight', pad_inches=0)
    print(f"Figure saved as: {filename}")

# -----------------------------------------------------------------------

def plot_learning_curves(history, attribs=["loss","val_loss"], attribs_extra=[],
                         style=['r--','b-.'], style_extra=['c-', 'k:'],
                         figsize=(4,3), title="", ylabel="Loss"):
    '''
    This function plots the learning curves from the training of a
    model.  The argument, history, must be the output from
    `model.fit(...)`. By default, the learning curves for training
    loss (attribute "loss") and validation loss (attribute "val_loss")
    would be plot.
    Input arguments:
        history - the history output from `model.fit(...)`
        attribs - optional argument (default=["loss","val_loss"]).
            This argument should be a list containing two attributes
            so that the learning curves of the two attributes would be
            superimposed in the same diagram. Obviously, the two
            attributes must be present in the `history` argument.
        attribs_extra - optional argument (default=[]). If this
            argument is specified, then an additional subplot would be
            shown. Similar to attribs, this argument, if used, should
            be a list containing two attributes. Obviously, the extra
            attributes specified here must be present in the `history`
            argument.
        style - optional argument (default=['r--','b-.']) for the
            colours and line styles of the learning curves for the
            attributes in the `attribs` argument.
        style_extra - optional argument (default=['c-', 'k:']) for the
            colours and line styles of the learning curves for the
            attributes in the `attribs_extra` argument. This argument
            should only be used with the `attribs_extra` argument.
        figsize - optional argument (default=(4,3)) for the total
            figure size of the single subplot (or double
            subplots). Note that this default figsize is suitable for
            single subplot (i.e., when `attribs_extra` = []).
        title - optional argument (default="") for the plot title.
        ylabel - optional argument (default="Loss") for the y-axis label.
    '''
    ncols = 1 if attribs_extra == [] else 2
    
    fig = plt.figure(figsize=figsize)
    if title:
        fig.suptitle(title, fontweight='medium')
    
    ax = fig.add_subplot(1, ncols, 1)
    pd.DataFrame(history.history)[attribs].plot(ax=ax, style=style)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid('on')
    
    if attribs_extra != []:
        ax = fig.add_subplot(1, ncols, 2)
        pd.DataFrame(history.history)[attribs_extra].plot(ax=ax,
                            style=style_extra)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.grid('on')
    
    plt.show()  # Display the figure
    return fig  # Return figure object for saving

# -----------------------------------------------------------------------

def plot_confusion_matrices(ytrain, ytrain_pred, ytest, ytest_pred,
                            labels=['class0', 'class1'],
                            suptitle="", figsize=(7,4),
                            normalize='all', **kwargs):
    '''
    This function shows two confusion matrices side-by-side. The
    default figure size is set for binary classification.
    Input arguments:
        ytrain, ytrain_pred, ytest, ytest_pred: all should be 1D
          arrays containing the class IDs.  They are arrays for the
          training labels, predicted labels for the training set, test
          labels, and predicted labels for the test set.
        labels: a list of strings for class 0, class 1, and so on.
        normalize: optional argument (default='all'). This is the same
            optional argument as the `sklearn.metrics.confusion_matrix`
            function. Use: 
              normlize=None for no normalization;
              normlize='true' for normalization on the rows using the
                 ground truth counts; 
              normlize='pred' for normalization on
                  the columns using the prediction counts.            
        figsize: a tuple containing the suitable width and height values
            of the whole figure.
        **kwargs: additional arguments that are recognized by the
            `sklearn.metrics.ConfusionMatrixDisplay.plot` function.
            Some useful keyword argument are:
                values_format='.1%' -- to show the values as percentage
                    with 1 decimal place.
    '''
    fig, axes = plt.subplots(1,2, figsize=figsize, sharey=True)
    for ytrue, ypred, axes_i, title in zip([ytrain,ytest],
                                           [ytrain_pred,ytest_pred], axes,
                                           ['Training set', 'Test set']):
        cm = confusion_matrix(ytrue, ypred, normalize=normalize)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=axes_i, colorbar=False, **kwargs)
        axes_i.set_xticklabels(labels, rotation=30, ha='right',
                               rotation_mode='anchor')
        axes_i.set_title(title, fontweight='bold')
        axes_i.set_xlabel('Predicted labels', fontsize=14)
        if axes_i == axes[0]:
            axes_i.set_ylabel('True labels', fontsize=14)
        else:
            # Do not plot a ylabel for the second subplot
            axes_i.set_ylabel('')
    plt.suptitle(suptitle, fontsize=16, fontweight='bold')

# -----------------------------------------------------------------------

def plot_confusion_matrix_as_heatmap(ytrue, ypred, ticks=[], ticklabels=[],
        colorbar=False, title="", figsize=(7,7), cmap="viridis",
        xlabel="Predicted", ylabel="Actual"):
    '''
    This function shows a confusion matrix as a heat map. This function
    is useful when the number of classes is large (i.e., for multiclass
    classification). Because of the size of the confusion matrix, numbers
    are usually not shown in the cells.
    Input arguments:
        ytrue, ypred: should be 1D arrays containing the class IDs
          (0, 1, 2, 3, ...). They are arrays for the training labels
          and predicted labels.
        ticks - (optional, default=[]) this should be a list or numpy array
          containing the tick mark positions. The default is: no tick
          marks will be shown.
        ticklabels - (optional, default=[]) this should be a list of
          character strings to be displayed at the tick mark positions.
          It should have the same length as the optional argument `ticks`.
        colorbar - (optional, default=False): a boolean to indicate whether
          the colour bar should be displayed. Default: no colour bar.
        title - (option, default="") a character string for the title
          of the plot. Default: no title.
        figsize - (optional, default=(7,7)) a 2-tuple containing the
          suitable width and height values of the whole figure.
          Obviously, this depends on the number of classes in the multiclass
          classification problem (i.e., the largest integer value in the
          ytrue and ypred array).
        cmap - (optional, default="viridis") the colour map to be used
          for the heat map. For the full list of recognised colour maps, see
          https://matplotlib.org/stable/gallery/color/colormap_reference.html
        xlabel - (optional, default="Predicted") the character string for
          the horizontal axis (or x-axis).
        ylabel - (optional, default="Actual") the character string for
          the vertical axis (or y-axis).
    '''
    fig = plt.figure(figsize=figsize)
    cm = confusion_matrix(ytrue, ypred, normalize=None)
    if colorbar == False:
        ax = sns.heatmap(cm, cmap=cmap, annot=False, square=True, cbar=False)
    else:
        ax = sns.heatmap(cm, cmap=cmap, annot=False, square=True,
            # default colour bar setting
            cbar_kws={"orientation":"vertical","label":"counts", "shrink":0.8})
    if len(ticks) > 0:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_yticklabels(ticklabels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

# -----------------------------------------------------------------------

def picklesave(dict, filename):
    '''
    Save the given object, `dict`, as a pickle file.
    Input arguments:
    dict - any Python variable to be saved to the specified pickle file.
    filename - the pickle file name (usually with .pkl as the extension).

    Example 1: save a variable to a pickle file
    picklesave(history, "myDNN-history.pkl")
    Example 1: save multiple variables
    picklesave((X_train, y_train, history), "myDNN-objects.pkl")

    SEE ALSO: pickleload
    '''
    import pickle
    f = open(filename, 'wb')
    pickle.dump(dict, f)
    f.close()

# -----------------------------------------------------------------------
    
def pickleload(filename):
    '''
    Load a previously saved object from disk (if the object was saved 
    as a pickle file).
    Input argument:
    file name - the name of the pickle file.
    Output:
    dict - the object read from the file.

    Example 1: load a variable previously saved in a pickle file
    history = pickleload("myDNN-history.pkl")
    Example 2: load variables previously saved in a pickle file (note that
               the order of extracting the variables needs to be consistent
               with the order that they were saved)
    (X_train, y_train, history) = pickleload("myDNN-objects.pkl")

    SEE ALSO: picklesave
    '''
    import pickle
    f = open(filename, 'rb')
    dict = pickle.load(f)
    f.close()
    return dict

# -----------------------------------------------------------------------

def calculate_metrics(ytrue, ypred):
    '''
    This function returns the accuracy, mean precision, and mean recall
    for the given ground truth array (ytrue) and the predicted array (ypred).
    It is suitable for multiclass classification problems. The function
    assumes that the two arrays contain integer values that denote the class
    IDs starting from 0.
    Input arguments:
    ytrue, ypred - The two input arrays must have the same length and must
        contain integer values only.
    Output arguments:
    accuracy - the accuracy of the predictions
    mPrecision, mRecall, mF1score - the mean precision, mean recall, and mean
        F1 score.
    precisions, recalls - each is a list of precisions and a list of recalls.
        Each class takes turn to be the target class while the other classes
        are bundled together to form a large non-target class. Precision and
        recall are then calculated for each target class to yield these two
        output arrays.
        
    NOTE: This function is equivalent to 
    sklearn.metrics.precision_recall_fscore_support(ytrue, ypred, averageg='macro')
    '''
    cm = confusion_matrix(ytrue, ypred, normalize=None)
    Nclasses = cm.shape[0]
    accuracy = np.diag(cm).sum() / np.sum(cm)
    precisions = [cm[i,i] / np.sum(cm[:,i]) for i in range(Nclasses)]
    mPrecision = np.mean(precisions)
    recalls = [cm[i,i] / np.sum(cm[i]) for i in range(Nclasses)]
    mRecall = np.mean(recalls)
    F1scores = [2*precisions[i]*recalls[i] / (precisions[i] + recalls[i]) \
                for i in range(Nclasses)]
    mF1score = np.mean(F1scores)
    return accuracy, mPrecision, mRecall, mF1score, precisions, recalls
        
# -----------------------------------------------------------------------
