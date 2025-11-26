from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np
# TODO 好使的部分  针对100个类别的混淆矩阵
def plot_confusion_matrix(true_data, pre_data, path,SNR,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure()#  dpi=150
    cm = confusion_matrix(true_data, pre_data)
    if title is not None:
        plt.title(title)
    tick_marks = np.arange(len(set(true_data)))# 设置课标 自适应的
    if len(set(true_data))<31:
        plt.xticks(tick_marks, fontsize=16, rotation=90)#
        plt.yticks(tick_marks, fontsize=16)
    elif len(set(true_data))<60&len(set(true_data))>31:
        # print("1111")
        plt.xticks(tick_marks, fontsize=13, rotation=90)#
        plt.yticks(tick_marks, fontsize=13)
    elif len(set(true_data))>90:
        # print("1111")
        plt.xticks(tick_marks, fontsize=5.5, rotation=90)#
        plt.yticks(tick_marks, fontsize=5.7)
    else:
        plt.xticks(tick_marks, fontsize=10, rotation=90)#
        plt.yticks(tick_marks, fontsize=10)

    if normalize:
        cm2 = np.array(cm,dtype=np.float32)
        num = np.sum(cm2, axis=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cm2[i][j] = 1.0 * cm2[i][j] / num[i]
                cm2[i][j] = round(float(cm2[i][j]), 3)
        plot = plt.imshow(cm2, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(plot)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        print('Confusion matrix, without normalization')
        plt.colorbar()
    plt.ylabel('True label', fontsize=20,labelpad=12.5)#
    plt.xlabel('Predicted label',fontsize=20,labelpad=12.5)# ,labelpad=12.5
    cm1 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    dig = np.diag(cm1)
    acc = dig.mean()
    acc = format(acc, '.4%')
    print("Mean accuracy:", acc)
    plt.show()
    if normalize:
        fig.savefig(path+'Confusion_matrix_'+ str(SNR) + 'dB_' + str(acc) + '.svg')
    else:
        fig.savefig(path + 'Confusion_matrix_without normalization_' + str(SNR) + 'dB_'+ str(acc) + '.svg')