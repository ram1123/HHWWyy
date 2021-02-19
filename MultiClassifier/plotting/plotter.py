import numpy as np
#import shap
import pandas
import os
import sklearn
import subprocess
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

class plotter(object):

    def __init__(self):
        self.separations_categories = []
        self.output_directory = ''
        #self.bin_edges_low_high = np.array([0.,0.0625,0.125,0.1875,0.25,0.3125,0.375,0.4375,0.5,0.5625,0.6125,0.6875,0.75,0.8125,0.875,0.9375,1.0])
        self.nbins = np.linspace(0.0,1.0,num=50)
        w, h = 4, 4
        self.yscores_train_categorised = [[0 for x in range(w)] for y in range(h)]
        self.yscores_test_categorised = [[0 for x in range(w)] for y in range(h)]
        self.yscores_train_non_categorised = [[0 for x in range(w)] for y in range(h)]
        self.yscores_test_non_categorised = [[0 for x in range(w)] for y in range(h)]
        self.plots_directory = ''
        pass

    def save_plots(self, dir='plots/', filename=''):
        self.check_dir(dir)
        filepath = os.path.join(dir,filename)
        self.fig.savefig(filepath)
        return self.fig

    def check_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def plot_training_progress_acc(self, histories, labels):
        self.fig, self.ax1 = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('Model Loss')
        plt.ylabel('Loss')
        #plt.tick_params(labelsize=15)
        #self.ax1.plot(x1,y1)
        #self.ax1.set_xticklabels(fontsize=15)
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        for i in range(len(histories)):
            history1 = histories[i]
            label_name_train = '%s train' % (labels[i])
            label_name_test = '%s test' % (labels[i])
            plt.plot(history1.history['loss'], label=label_name_train)
            plt.plot(history1.history['val_loss'], label=label_name_test)
            plt.legend(['train', 'test'], loc='upper right')

        #plt.yticks([2.05,2.10,2.15,2.20,2.25,2.30,2.35,2.40,2.45],labelsize=8)
        acc_title = 'plots/DNN_accuracy_wrt_epoch.png'
        plt.tight_layout()
        return

    def correlation_matrix(self, data, **kwds):

        #iloc[<row selection>,<column selection>]
        self.data = data.iloc[:, :-4]
        self.labels = self.data.corr(**kwds).columns.values
        self.fig, self.ax1 = plt.subplots(ncols=1, figsize=(20,20))
        #self.ax1.plot(x1,y1)
        #self.fig.tick_params(fontsize=15)
        opts = {"annot" : True, "ax" : self.ax1, "vmin" : 0, "vmax" : 1*100, "annot_kws" : {"size":8}, "cmap" : plt.get_cmap("Blues",20), 'fmt' : '0.2f',}
        self.ax1.set_title("Correlations")
        sns.heatmap(self.data.corr(method='spearman')*100, **opts)
        for ax in (self.ax1,):
            # Shift tick location to bin centre
            ax.set_xticks(np.arange(len(self.labels))+0.5, minor=False)
            ax.set_yticks(np.arange(len(self.labels))+0.5, minor=False)
            ax.set_xticklabels(self.labels, minor=False, ha='right', rotation=45)
            #ax.set_yticklabels(np.flipud(self.labels), minor=False, rotation=45)
            ax.set_yticklabels(self.labels, minor=False, rotation=45)

        plt.tight_layout()

        return

    def ROC(self, model, X_test, Y_test, X_train, Y_train):
        y_pred_keras_test = model.predict(X_test).ravel()
        fpr_keras_test, tpr_keras_test, thresholds_keras_test = roc_curve(Y_test, y_pred_keras_test)
        auc_keras_test = auc(fpr_keras_test, tpr_keras_test)

        y_pred_keras_train = model.predict(X_train).ravel()
        fpr_keras_train, tpr_keras_train, thresholds_keras_train = roc_curve(Y_train, y_pred_keras_train)
        auc_keras_train = auc(fpr_keras_train, tpr_keras_train) 
        
        self.fig, self.ax1 = plt.subplots(ncols=1, figsize=(10,10)) 
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras_test, tpr_keras_test, label='Test (area = {:.3f})'.format(auc_keras_test))
        plt.plot(fpr_keras_train, tpr_keras_train, label='Train (area = {:.3f})'.format(auc_keras_train))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.tight_layout()

        return

    def history_plot(self, history, label='accuracy'):
        self.fig, self.ax1 = plt.subplots(ncols=1, figsize=(10,10)) 
        plt.plot(history.history[label])
        plt.plot(history.history['val_'+label])
        plt.title('model '+label)
        plt.ylabel(label)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        plt.tight_layout()

        return

    def conf_matrix(self, y_true, y_predicted, EventWeights_, norm=' '):
        y_true = pandas.Series(y_true, name='truth')
        y_predicted = pandas.Series(y_predicted, name='prediction')
        EventWeights_ = pandas.Series(EventWeights_, name='eventweights')
        if norm == 'index':
            self.matrix = pandas.crosstab(y_true,y_predicted,EventWeights_,aggfunc=sum,normalize='index')
            vmax = 1
        elif norm == 'columns':
            self.matrix = pandas.crosstab(y_true,y_predicted,EventWeights_,aggfunc=sum,normalize='columns')
            vmax = 1
        else:
            self.matrix = pandas.crosstab(y_true,y_predicted,EventWeights_,aggfunc=sum)
            vmax = 150

        self.labelsx = self.matrix.columns
        self.labelsy = self.matrix.index
        self.fig, self.ax1 = plt.subplots(ncols=1, figsize=(10,10))
        plt.rcParams.update({'font.size': 22})
        self.ax1.set_title("Confusion Matrix", fontsize=18)
        opts = {"annot" : True, "ax" : self.ax1, "vmin" : 0, "vmax" : vmax, "annot_kws" : {"size":18}, "cmap" : plt.get_cmap("Reds",20), 'fmt' : '0.2f',}
        sns.set(font_scale=2.4)
        sns.heatmap(self.matrix, **opts)
        label_dict = {
            0 : 'HH',
            1 : 'yyjets',
            2 : 'GJets',
            3 : 'DY'
        }
        for ax in (self.ax1,):
            #Shift tick location to bin centre
            ax.set_xticks(np.arange(len(self.labelsx))+0.5, minor=False)
            ax.set_yticks(np.arange(len(self.labelsy))+0.5, minor=False)
            new_xlabel = []
            new_ylabel = []
            for xlabel in self.labelsx:
                new_xlabel.append(label_dict.get(xlabel))
            for ylabel in self.labelsy:
                new_ylabel.append(label_dict.get(ylabel))
            ax.set_xticklabels(new_xlabel, minor=False, ha='right', rotation=45, fontsize=18)
            ax.set_yticklabels(new_ylabel, minor=False, rotation=45, fontsize=18)
        plt.tight_layout()
        return


    def ROC_sklearn(self, original_encoded_train_Y, result_probs_train, original_encoded_test_Y, result_probs_test, encoded_signal, pltname='', train_weights=[], test_weights=[]):

        # ROC curve and AUC calculated using absolute number of examples, not weighted events.
        self.fig, self.ax1 = plt.subplots(ncols=1, figsize=(10,10))

        # Set value in list to 1 for signal and 0 for any background.
        SorB_class_train = []
        SorB_class_test = []
        output_probs_train = []
        output_probs_test = []
        training_event_weight = []
        testing_event_weight = []

        # Loop over all training events
        for i in range(0,len(original_encoded_train_Y)):
            # If training events truth value is target for the node assigned as signal by the variable encoded_signal, append a 1
            # else assign as background and append a 0.
            if original_encoded_train_Y[i] == encoded_signal:
                SorB_class_train.append(1)
            else:
                SorB_class_train.append(0)
            # For ith event, get the probability that this event is from the signal process
            output_probs_train.append(result_probs_train[i])
            training_event_weight.append(train_weights[i])

        # Loop over all testing events and repeat procedure as for training
        for i in range(0,len(original_encoded_test_Y)):
            if original_encoded_test_Y[i] == encoded_signal:
                SorB_class_test.append(1)
            else:
                SorB_class_test.append(0)
            output_probs_test.append(result_probs_test[i])
            testing_event_weight.append(test_weights[i])

        if len(original_encoded_test_Y) == 0:
            labels = ['SR applied']
        else:
            labels = ['TR train','TR test']

        #print('# training_event_weight: ',len(training_event_weight))
        #print('# testing_event_weight: ', len(testing_event_weight))
        if len(SorB_class_train) > 0:
            # Create ROC curve - scan across the node distribution and calculate the true and false
            # positive rate for given thresholds.
            fpr, tpr, thresholds = roc_curve(SorB_class_train, output_probs_train, sample_weight=None, pos_label=1)
            auc_train_node_score = roc_auc_score(SorB_class_train, output_probs_train)

            # Plot the roc curve for the model
            # Interpolate between points of fpr and tpr on graph to get curve
            plt.plot(fpr, tpr, marker='.', markersize=8, label='%s (area = %0.2f)' % (labels[0],auc_train_node_score))

        if len(SorB_class_test) > 0:
            fpr, tpr, thresholds = roc_curve(SorB_class_test, output_probs_test, sample_weight=None, pos_label=1)
            auc_test_node_score = roc_auc_score(SorB_class_test, output_probs_test)
            # Plot the roc curve for the model
            plt.plot(fpr, tpr, marker='.', markersize=8, label='%s (area = %0.2f)' % (labels[1],auc_test_node_score))

        plt.plot([0, 1], [0, 1], linestyle='--', markersize=8,)
        plt.rcParams.update({'font.size': 22})
        self.ax1.set_title(pltname, fontsize=18)
        plt.legend(loc="lower right")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.tight_layout()
        # save the plot
        save_name = pltname
        self.save_plots(dir=self.plots_directory, filename=save_name)
        return

    def GetSeparation(self, hist_sig, hist_bckg):


        minima = np.minimum(hist_sig, hist_bckg)
        intersection = np.true_divide(np.sum(minima), np.sum(hist_bckg))
        return intersection

        '''
        # compute "separation" defined as
        # <s2> = (1/2) Int_-oo..+oo { (S(x) - B(x))^2/(S(x) + B(x)) dx }
        separation = 0;
        # sanity check: signal and background histograms must have same number of bins and same limits
        if len(hist_sig) != len(hist_bckg):
            print 'Number of bins different for sig. and bckg'

        nBins = len(hist_sig)
        nS = np.sum(hist_sig)
        nB = np.sum(hist_bckg)

        if nS == 0:
            print 'WARNING: no signal'
        if nB == 0:
            print 'WARNING: no bckg'

        for i in range(1,nBins):
            # No need to norm as already done?
            sig_bin_norm = hist_sig[i]/nS
            #sig_bin_norm = hist_sig[i]
            bckg_bin_norm = hist_bckg[i]/nB
            #bckg_bin_norm = hist_bckg[i]
            # Separation:
            if(sig_bin_norm+bckg_bin_norm > 0):
                separation += 0.5 * ((sig_bin_norm - bckg_bin_norm) * (sig_bin_norm - bckg_bin_norm)) / (sig_bin_norm + bckg_bin_norm)
        #separation *= dX
        return separation'''


    def draw_category_overfitting_plot(self, y_scores_train, y_scores_test, plot_info):
        labels = plot_info[0]
        colours = plot_info[1]
        data_type = plot_info[2]
        plots_dir = plot_info[3]
        node_name = plot_info[4]
        plot_title = plot_info[5]
        name = filter(str.isalnum, str(data_type).split(".")[-1])
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        self.ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
        self.ax.set_facecolor('white')

        bin_edges_low_high = np.array([0.,0.0625,0.125,0.1875,0.25,0.3125,0.375,0.4375,0.5,0.5625,0.6125,0.675,0.7375,0.8,0.8625,0.9375,1.0])

        for index in range(0,len(y_scores_train)-1):
            y_train = y_scores_train[index]
            y_test = y_scores_test[index]
            label = labels[index]
            colour = colours[index]

            trainlabel = label + ' train'
            width = np.diff(bin_edges_low_high)
            histo_train_, bin_edges = np.histogram(y_train, bins=bin_edges_low_high)
            bincenters = 0.5*(bin_edges[1:]+bin_edges[:-1])
            dx_scale_train =(bin_edges[len(bin_edges)-1] - bin_edges[0]) / (len(bin_edges)-1)
            histo_train_ = histo_train_ / np.sum(histo_train_, dtype=np.float32) / dx_scale_train
            plt.bar(bincenters, histo_train_, width=width, color=colour, edgecolor=colour, alpha=0.5, label=trainlabel)

            if index == 0:#HH
                histo_train_HH = histo_train_ / np.sum(histo_train_, dtype=np.float32)
            if index == 1:#yyjets
                histo_train_yyjets = histo_train_ / np.sum(histo_train_, dtype=np.float32)
            if index == 2:#GJets
                histo_train_GJets = histo_train_ / np.sum(histo_train_, dtype=np.float32)
            if index == 3:#DY
                histo_train_DY = histo_train_ / np.sum(histo_train_, dtype=np.float32)

            testlabel = label + ' test'
            histo_test_, bin_edges = np.histogram(y_test, bins=bin_edges_low_high)
            dx_scale_test =(bin_edges[len(bin_edges)-1] - bin_edges[0]) / (len(bin_edges)-1)
            bincenters = 0.5*(bin_edges[1:]+bin_edges[:-1])

            if np.sum(histo_test_, dtype=np.float32) <= 0 :
                histo_test_ = histo_test_
                err = 0
                plt.errorbar(bincenters, histo_test_, yerr=err, fmt='o', c=colour, label=testlabel)
                if index == 0:
                    histo_test_HH = histo_test_
                if index == 1:
                    histo_test_yyjets = histo_test_
                if index == 2:
                    histo_test_GJets = histo_test_
                if index == 3:
                    histo_test_DY = histo_test_
            else:
                err = np.sqrt(histo_test_/np.sum(histo_test_, dtype=np.float32))
                histo_test_ = histo_test_ / np.sum(histo_test_, dtype=np.float32) / dx_scale_test
                plt.errorbar(bincenters, histo_test_, yerr=err, fmt='o', c=colour, label=testlabel)
                if index == 0:
                    histo_test_HH = histo_test_ / np.sum(histo_test_, dtype=np.float32)
                if index == 1:
                    histo_test_yyjets = histo_test_ / np.sum(histo_test_, dtype=np.float32)
                if index == 2:
                    histo_test_GJets = histo_test_ / np.sum(histo_test_, dtype=np.float32)
                if index == 3:
                    histo_test_DY = histo_test_ / np.sum(histo_test_, dtype=np.float32)

        if 'HH' in node_name:
            train_HHvyyjetsSep = "{0:.5g}".format(self.GetSeparation(histo_train_HH,histo_train_yyjets))
            train_HHvGJetsSep = "{0:.5g}".format(self.GetSeparation(histo_train_HH,histo_train_GJets))
            train_HHvDYSep = "{0:.5g}".format(self.GetSeparation(histo_train_HH,histo_train_DY))

            test_HHvyyjetsSep = "{0:.5g}".format(self.GetSeparation(histo_test_HH,histo_test_yyjets))
            test_HHvGJetsSep = "{0:.5g}".format(self.GetSeparation(histo_test_HH,histo_test_GJets))
            test_HHvDYSep = "{0:.5g}".format(self.GetSeparation(histo_test_HH,histo_test_DY))

            HH_v_yyjets_train_sep = 'HH vs yyjets train Sep.: %s' % ( train_HHvyyjetsSep )
            self.ax.annotate(HH_v_yyjets_train_sep,  xy=(0.7, 2.5), xytext=(0.7, 2.5), fontsize=9)
            HH_v_GJets_train_sep = 'HH vs GJets train Sep.: %s' % ( train_HHvGJetsSep )
            self.ax.annotate(HH_v_GJets_train_sep,  xy=(0.7, 2.25), xytext=(0.7, 2.25), fontsize=9)
            HH_v_DY_train_sep = 'HH vs DY train Sep.: %s' % ( train_HHvDYSep )
            self.ax.annotate(HH_v_DY_train_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)

            HH_v_yyjets_test_sep = 'HH vs yyjets test Sep.: %s' % ( test_HHvyyjetsSep )
            self.ax.annotate(HH_v_yyjets_test_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.5), fontsize=9)
            HH_v_GJets_test_sep = 'HH vs GJets test Sep.: %s' % ( test_HHvGJetsSep )
            self.ax.annotate(HH_v_GJets_test_sep,  xy=(0.7, 1.5), xytext=(0.7, 1.25), fontsize=9)
            HH_v_DY_test_sep = 'HH vs DY test Sep.: %s' % ( test_HHvDYSep )
            self.ax.annotate(HH_v_DY_test_sep,  xy=(0.7, 1.), xytext=(0.7, .75), fontsize=9)
            separations_forTable = r'''\textbackslash & %s & %s & %s ''' % (test_HHvyyjetsSep, test_HHvGJetsSep, HH_v_DY_test_sep)
        if 'yyjets' in node_name:
            train_yyjetsvHH = "{0:.5g}".format(self.GetSeparation(histo_train_yyjets,histo_train_HH))
            train_yyjetsvGJets = "{0:.5g}".format(self.GetSeparation(histo_train_yyjets,histo_train_GJets))
            train_yyjetsvDY = "{0:.5g}".format(self.GetSeparation(histo_train_yyjets,histo_train_DY))

            test_yyjetsvHH = "{0:.5g}".format(self.GetSeparation(histo_test_yyjets,histo_test_HH))
            test_yyjetsvGJets = "{0:.5g}".format(self.GetSeparation(histo_test_yyjets,histo_test_GJets))
            test_yyjetsvDY = "{0:.5g}".format(self.GetSeparation(histo_test_yyjets,histo_test_DY))

            yyjets_v_HH_train_sep = 'yyjets vs HH train Sep.: %s' % ( train_yyjetsvHH )
            self.ax.annotate(yyjets_v_HH_train_sep,  xy=(0.7, 2.5), xytext=(0.7, 2.5), fontsize=9)
            yyjets_v_GJets_train_sep = 'yyjets vs GJets train Sep.: %s' % ( train_yyjetsvGJets )
            self.ax.annotate(yyjets_v_GJets_train_sep,  xy=(0.7, 2.25), xytext=(0.7, 2.25), fontsize=9)
            yyjets_v_DY_train_sep = 'yyjets vs DY train Sep.: %s' % ( train_yyjetsvDY )
            self.ax.annotate(yyjets_v_DY_train_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)

            yyjets_v_HH_test_sep = 'yyjets vs HH test Sep.: %s' % ( test_yyjetsvHH )
            self.ax.annotate(yyjets_v_HH_test_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)
            yyjets_v_GJets_test_sep = 'yyjets vs GJets test Sep.: %s' % ( test_yyjetsvGJets )
            self.ax.annotate(yyjets_v_GJets_test_sep,  xy=(0.7, 1.5), xytext=(0.7, 1.5), fontsize=9)
            yyjets_v_DY_test_sep = 'yyjets vs DY test Sep.: %s' % ( test_yyjetsvDY )
            self.ax.annotate(yyjets_v_DY_test_sep,  xy=(0.7, 1.), xytext=(0.7, 1.), fontsize=9)

            separations_forTable = r'''%s & \textbackslash & %s & %s''' % (test_yyjetsvHH,test_yyjetsvGJets,yyjets_v_DY_test_sep)

        if 'GJets' in node_name:
            train_GJetsvHH = "{0:.5g}".format(self.GetSeparation(histo_train_GJets,histo_train_HH))
            train_GJetsvyyjets = "{0:.5g}".format(self.GetSeparation(histo_train_GJets,histo_train_yyjets))
            train_GJetsvDY = "{0:.5g}".format(self.GetSeparation(histo_train_GJets,histo_train_DY))

            test_GJetsvHH = "{0:.5g}".format(self.GetSeparation(histo_test_GJets,histo_test_HH))
            test_GJetsvyyjets = "{0:.5g}".format(self.GetSeparation(histo_test_GJets,histo_test_yyjets))
            test_GJetsvDY = "{0:.5g}".format(self.GetSeparation(histo_test_GJets,histo_test_DY))

            GJets_v_HH_train_sep = 'GJets vs HH train Sep.: %s' % ( train_GJetsvHH )
            self.ax.annotate(GJets_v_HH_train_sep,  xy=(0.7, 2.5), xytext=(0.7, 2.5), fontsize=9)
            GJets_v_yyjets_train_sep = 'GJets vs yyjets train Sep.: %s' % ( train_GJetsvyyjets )
            self.ax.annotate(GJets_v_yyjets_train_sep,  xy=(0.7, 2.25), xytext=(0.7, 2.25), fontsize=9)
            GJets_v_DY_train_sep = 'GJets vs GJets train Sep.: %s' % ( train_GJetsvDY )
            self.ax.annotate(GJets_v_DY_train_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)

            GJets_v_HH_test_sep = 'GJets vs HH test Sep.: %s' % ( test_GJetsvHH )
            self.ax.annotate(GJets_v_HH_test_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)
            GJets_v_yyjets_test_sep = 'GJets vs yyjets test Sep.: %s' % ( test_GJetsvyyjets )
            self.ax.annotate(GJets_v_yyjets_test_sep,  xy=(0.7, 1.5), xytext=(0.7, 1.5), fontsize=9)
            GJets_v_DY_test_sep = 'GJets vs DY test Sep.: %s' % ( test_GJetsvDY )
            self.ax.annotate(GJets_v_DY_test_sep,  xy=(0.7, 1.), xytext=(0.7, 1.), fontsize=9)

            separations_forTable = r'''%s & %s & \textbackslash & %s''' % (test_GJetsvHH,test_GJetsvyyjets,GJets_v_DY_test_sep)

        if 'DY' in node_name:
            train_DYvHH = "{0:.5g}".format(self.GetSeparation(histo_train_DY,histo_train_HH))
            train_DYvyyjets = "{0:.5g}".format(self.GetSeparation(histo_train_DY,histo_train_yyjets))
            train_DYvGJets = "{0:.5g}".format(self.GetSeparation(histo_train_DY,histo_train_GJets))

            test_DYvHH = "{0:.5g}".format(self.GetSeparation(histo_test_DY,histo_test_HH))
            test_DYvyyjets = "{0:.5g}".format(self.GetSeparation(histo_test_DY,histo_test_yyjets))
            test_DYvGJets = "{0:.5g}".format(self.GetSeparation(histo_test_DY,histo_test_GJets))

            DY_v_HH_train_sep = 'DY vs HH train Sep.: %s' % ( train_DYvHH )
            self.ax.annotate(DY_v_HH_train_sep,  xy=(0.7, 2.5), xytext=(0.7, 2.5), fontsize=9)
            DY_v_yyjets_train_sep = 'DY vs yyjets train Sep.: %s' % ( train_DYvyyjets )
            self.ax.annotate(DY_v_yyjets_train_sep,  xy=(0.7, 2.25), xytext=(0.7, 2.25), fontsize=9)
            DY_v_GJets_train_sep = 'DY vs GJets train Sep.: %s' % ( train_DYvGJets )
            self.ax.annotate(DY_v_GJets_train_sep,  xy=(0.7, 2.), xytext=(0.7, 2.), fontsize=9)

            DY_v_HH_test_sep = 'DY vs HH test Sep.: %s' % ( test_DYvHH )
            self.ax.annotate(DY_v_HH_test_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)
            DY_v_yyjets_test_sep = 'DY vs yyjets test Sep.: %s' % ( test_DYvyyjets )
            self.ax.annotate(DY_v_yyjets_test_sep,  xy=(0.7, 1.5), xytext=(0.7, 1.5), fontsize=9)
            DY_v_GJets_test_sep = 'DY vs GJets test Sep.: %s' % ( test_DYvGJets )
            self.ax.annotate(DY_v_GJets_test_sep,  xy=(0.7, 1.25), xytext=(0.7, 1.25), fontsize=9)

            separations_forTable = r'''%s & %s & %s & \textbackslash ''' % (test_DYvHH,test_DYvyyjets,test_DYvGJets)


        title_ = '%s %s node' % (plot_title,node_name)
        plt.title(title_)
        label_name = 'DNN Output Score'
        plt.xlabel(label_name)
        plt.ylabel('(1/N)dN/dX')

        leg = plt.legend(loc='best', frameon=False, fancybox=False, fontsize=9)
        leg.get_frame().set_edgecolor('w')
        frame = leg.get_frame()
        frame.set_facecolor('White')

        overfitting_plot_file_name = 'overfitting_plot_%s_%s.png' % (node_name,plot_title)
        print('Saving : %s%s' % (plots_dir, overfitting_plot_file_name))
        self.save_plots(dir=plots_dir, filename=overfitting_plot_file_name)

        return separations_forTable


    def draw_binary_overfitting_plot(self, y_scores_train, y_scores_test, plot_info, test_weights):
        colours = plot_info[0]
        data_type = plot_info[1]
        plots_dir = plot_info[2]
        plot_title = plot_info[3]
        name = filter(str.isalnum, str(data_type).split(".")[-1])
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        self.ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
        self.ax.set_facecolor('white')

        bin_edges_low_high = np.array([0.,0.0625,0.125,0.1875,0.25,0.3125,0.375,0.4375,0.5,0.5625,0.6125,0.675,0.7375,0.8,0.8625,0.9375,1.0])

        index=0
        for index in range(0,len(y_scores_train)):
            train_bin_errors = np.zeros(len(bin_edges_low_high)-1)
            test_bin_errors = np.zeros(len(bin_edges_low_high)-1)

            y_train = y_scores_train[index]
            y_test = y_scores_test[index]
            colour = colours[index]
            width = np.diff(bin_edges_low_high)
            if index==0:
                print('<plotter> Overfitting plot: Signal')
                label='signal'
            if index==1:
                label='bckg'
                print('<plotter> Overfitting plot: Background')

            # Setup training histograms
            histo_train_, bin_edges = np.histogram(y_train, bins=bin_edges_low_high)
            dx_scale_train =(bin_edges[len(bin_edges)-1] - bin_edges[0]) / (len(bin_edges)-1)
            bincenters = 0.5*(bin_edges[1:]+bin_edges[:-1])

            bin_errors_sumw2 = 0

            # Scale training histograms to: (hist / sum of histogram entries) / (range / number of bins)
            # so like scaling hist relative to its integral.
            # then scaling the result relative to its avergae bin width.
            histo_train_ = (histo_train_ / np.sum(histo_train_, dtype=np.float32)) / dx_scale_train
            plt.bar(bincenters, histo_train_, width=width, color=colour, edgecolor=colour, alpha=0.5, label=label+' training')

            if index == 0:
                histo_train_sig = histo_train_
            if index == 1:
                histo_train_bckg = histo_train_

            histo_test_, bin_edges = np.histogram(y_test, bins=bin_edges_low_high)
            dx_scale_test =(bin_edges[len(bin_edges)-1] - bin_edges[0]) / (len(bin_edges)-1)
            bincenters = 0.5*(bin_edges[1:]+bin_edges[:-1])

            # Protection against low stats in validation dataset.
            if np.sum(histo_test_, dtype=np.float32) <= 0 :
                histo_test_ = histo_test_
                err = 0
                plt.errorbar(bincenters, histo_test_, yerr=err, fmt='o', c=colour, label=label+' testing')
                if index == 0:
                    histo_test_sig = histo_test_
                if index == 1:
                    histo_test_bckg = histo_test_
            else:
                # Currently just taking the sqrt of the bin entry
                # Correct way:
                # Errors calculated using error propogation and the histograms intrinsic poissonian statistics
                # err(sum weights)^2 == variance on the sum of weights = sum of the variance of each weight =  sum{var(w_i)} [i=1,2,N]
                # Varianceof weight i is determined by the statistical fluctuation of the number of events considered.
                # var(w_i) = var(w_i * 1 event) = w_i^2 * var(1 event) = w_i^2
                # err(sum weights) = sqrt( sum{var(w_i)}[i=1,2,N] )
                #                  = sqrt( sum{w_i^2}[i=1,2,N] )
                bin_errors_sumw2 = 0
                for yval_index in range(0,len(y_scores_test[0])):
                    for bin_index in range(0,len(bin_edges_low_high)-1):
                        bin_low_edge = bin_edges_low_high[bin_index]
                        bin_high_edge = bin_edges_low_high[bin_index+1]
                        if y_scores_test[0][yval_index] > bin_low_edge and y_scores_test[0][yval_index] < bin_high_edge:
                            test_bin_errors[[bin_index]] += 1#test_weights[yval_index]**2
                # Take square root of sum of bins.
                test_bin_errors = (np.sqrt(test_bin_errors)/np.sum(histo_test_, dtype=np.float32)) / dx_scale_test
                #test_bin_errors = np.sqrt(( test_bin_errors/np.sum(histo_test_, dtype=np.float32) ) / dx_scale_test )
                histo_test_ = ( histo_test_ / np.sum(histo_test_, dtype=np.float32) ) / dx_scale_test
                #err = np.sqrt(histo_test_/np.sum(histo_test_, dtype=np.float32))
                #err = np.sqrt(histo_test_)

                plt.errorbar(bincenters, histo_test_, yerr=test_bin_errors, fmt='o', c=colour, label=label+' testing')
                if index == 0:
                    histo_test_sig = histo_test_
                if index == 1:
                    histo_test_bckg = histo_test_


        train_SvsBSep = "{0:.5g}".format(self.GetSeparation(histo_train_sig,histo_train_bckg))
        test_SvsBSep = "{0:.5g}".format(self.GetSeparation(histo_test_sig,histo_test_bckg))

        S_v_B_train_sep = 'SvsB train Sep.: %s' % ( train_SvsBSep )
        self.ax.annotate(S_v_B_train_sep,  xy=(0.7, 2.5), xytext=(0.7, 2.5), fontsize=9)
        S_v_B_test_sep = 'SvsB test Sep.: %s' % ( test_SvsBSep )
        self.ax.annotate(S_v_B_test_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.5), fontsize=9)

        separations_forTable = r'''%s & \textbackslash ''' % (S_v_B_test_sep)

        title_ = '%s output node' % (plot_title)
        plt.title(title_)
        label_name = 'Output Score'
        plt.xlabel(label_name)
        plt.ylabel('(1/N)dN/dX')

        leg = plt.legend(loc='best', frameon=False, fancybox=False, fontsize=9)
        leg.get_frame().set_edgecolor('w')
        frame = leg.get_frame()
        frame.set_facecolor('White')

        overfitting_plot_file_name = 'overfitting_plot_BinaryClassifier_%s.png' % (plot_title)
        print('Saving : %s%s' % (plots_dir, overfitting_plot_file_name))
        self.save_plots(dir=plots_dir, filename=overfitting_plot_file_name)
        return separations_forTable

    def separation_table(self , outputdir):
        content = r'''\documentclass{article}
\begin{document}
\begin{center}
\begin{table}
\begin{tabular}{| c | c | c | c | c |} \hline
Node \textbackslash Background & HH & yyjets & GJets & DY \\ \hline
HH & %s \\
yyjets & %s \\
GJets & %s \\
DY & %s \\ \hline
\end{tabular}
\caption{Separation power on each output node. The separation is given with respect to the `signal' process the node is trained to separate (one node per row) and the background processes for that node (one background per column).}
\end{table}
\end{center}
\end{document}
'''
        table_path = os.path.join(outputdir,'separation_table')
        table_tex = table_path+'.tex'
        print('table_tex: ', table_tex)
        with open(table_tex,'w') as f:
            f.write( content % (self.separations_categories[0], self.separations_categories[1], self.separations_categories[2], self.separations_categories[3] ) )
        return

    def overfitting(self, estimator, Y_train, Y_test, result_probs, result_probs_test, plots_dir, train_weights, test_weights, nbins=50):

        model = estimator
        data_type = type(model)

        #Arrays to store all HH values
        y_scores_train_HH_sample_HHnode = []
        y_scores_train_yyjets_sample_HHnode = []
        y_scores_train_GJets_sample_HHnode = []
        y_scores_train_DY_sample_HHnode = []

        # Arrays to store HH categorised event values
        y_scores_train_HH_sample_HH_categorised = []
        y_scores_train_yyjets_sample_HH_categorised = []
        y_scores_train_GJets_sample_HH_categorised = []
        y_scores_train_DY_sample_HH_categorised = []

        # Arrays to store all yyjets node values
        y_scores_train_HH_sample_yyjetsnode = []
        y_scores_train_yyjets_sample_yyjetsnode = []
        y_scores_train_GJets_sample_yyjetsnode = []
        y_scores_train_DY_sample_yyjetsnode = []

        # Arrays to store yyjets categorised event values
        y_scores_train_HH_sample_yyjets_categorised = []
        y_scores_train_yyjets_sample_yyjets_categorised = []
        y_scores_train_GJets_sample_yyjets_categorised = []
        y_scores_train_DY_sample_yyjets_categorised = []

        # Arrays to store all GJets node values
        y_scores_train_HH_sample_GJetsnode = []
        y_scores_train_yyjets_sample_GJetsnode = []
        y_scores_train_GJets_sample_GJetsnode = []
        y_scores_train_DY_sample_GJetsnode = []

        # Arrays to store GJets categorised events
        y_scores_train_HH_sample_GJets_categorised = []
        y_scores_train_yyjets_sample_GJets_categorised = []
        y_scores_train_GJets_sample_GJets_categorised = []
        y_scores_train_DY_sample_GJets_categorised = []

        # Arrays to store all DY node values
        y_scores_train_HH_sample_DYnode = []
        y_scores_train_yyjets_sample_DYnode = []
        y_scores_train_GJets_sample_DYnode = []
        y_scores_train_DY_sample_DYnode = []

        # Arrays to store DY categorised events
        y_scores_train_HH_sample_DY_categorised = []
        y_scores_train_yyjets_sample_DY_categorised = []
        y_scores_train_GJets_sample_DY_categorised = []
        y_scores_train_DY_sample_DY_categorised = []

        for i in range(len(result_probs)):
            train_event_weight = train_weights[i]
            if Y_train[i][0] == 1:
                y_scores_train_HH_sample_HHnode.append(result_probs[i][0])
                y_scores_train_HH_sample_yyjetsnode.append(result_probs[i][1])
                y_scores_train_HH_sample_GJetsnode.append(result_probs[i][2])
                y_scores_train_HH_sample_DYnode.append(result_probs[i][3])
                # Get index of maximum argument.
                if np.argmax(result_probs[i]) == 0:
                    y_scores_train_HH_sample_HH_categorised.append(result_probs[i][0])
                if np.argmax(result_probs[i]) == 1:
                    y_scores_train_HH_sample_yyjets_categorised.append(result_probs[i][1])
                if np.argmax(result_probs[i]) == 2:
                    y_scores_train_HH_sample_GJets_categorised.append(result_probs[i][2])
                if np.argmax(result_probs[i]) == 3:
                    y_scores_train_HH_sample_DY_categorised.append(result_probs[i][3])
            if Y_train[i][1] == 1:
                y_scores_train_yyjets_sample_HHnode.append(result_probs[i][0])
                y_scores_train_yyjets_sample_yyjetsnode.append(result_probs[i][1])
                y_scores_train_yyjets_sample_GJetsnode.append(result_probs[i][2])
                y_scores_train_yyjets_sample_DYnode.append(result_probs[i][3])
                if np.argmax(result_probs[i]) == 0:
                    y_scores_train_yyjets_sample_HH_categorised.append(result_probs[i][0])
                if np.argmax(result_probs[i]) == 1:
                    y_scores_train_yyjets_sample_yyjets_categorised.append(result_probs[i][1])
                if np.argmax(result_probs[i]) == 2:
                    y_scores_train_yyjets_sample_GJets_categorised.append(result_probs[i][2])
                if np.argmax(result_probs[i]) == 3:
                    y_scores_train_yyjets_sample_DY_categorised.append(result_probs[i][3])
            if Y_train[i][2] == 1:
                y_scores_train_GJets_sample_HHnode.append(result_probs[i][0])
                y_scores_train_GJets_sample_yyjetsnode.append(result_probs[i][1])
                y_scores_train_GJets_sample_GJetsnode.append(result_probs[i][2])
                y_scores_train_GJets_sample_DYnode.append(result_probs[i][3])
                if np.argmax(result_probs[i]) == 0:
                    y_scores_train_GJets_sample_HH_categorised.append(result_probs[i][0])
                if np.argmax(result_probs[i]) == 1:
                    y_scores_train_GJets_sample_yyjets_categorised.append(result_probs[i][1])
                if np.argmax(result_probs[i]) == 2:
                    y_scores_train_GJets_sample_GJets_categorised.append(result_probs[i][2])
                if np.argmax(result_probs[i]) == 3:
                    y_scores_train_GJets_sample_DY_categorised.append(result_probs[i][3])
            if Y_train[i][3] == 1:
                y_scores_train_DY_sample_HHnode.append(result_probs[i][0])
                y_scores_train_DY_sample_yyjetsnode.append(result_probs[i][1])
                y_scores_train_DY_sample_GJetsnode.append(result_probs[i][2])
                y_scores_train_DY_sample_DYnode.append(result_probs[i][3])
                if np.argmax(result_probs[i]) == 0:
                    y_scores_train_DY_sample_HH_categorised.append(result_probs[i][0])
                if np.argmax(result_probs[i]) == 1:
                    y_scores_train_DY_sample_yyjets_categorised.append(result_probs[i][1])
                if np.argmax(result_probs[i]) == 2:
                    y_scores_train_DY_sample_GJets_categorised.append(result_probs[i][2])
                if np.argmax(result_probs[i]) == 3:
                    y_scores_train_DY_sample_DY_categorised.append(result_probs[i][3])


        #Arrays to store all HH values
        y_scores_test_HH_sample_HHnode = []
        y_scores_test_yyjets_sample_HHnode = []
        y_scores_test_GJets_sample_HHnode = []
        y_scores_test_DY_sample_HHnode = []

        # Arrays to store HH categorised event values
        y_scores_test_HH_sample_HH_categorised = []
        y_scores_test_yyjets_sample_HH_categorised = []
        y_scores_test_GJets_sample_HH_categorised = []
        y_scores_test_DY_sample_HH_categorised = []

        # Arrays to store all yyjets node values
        y_scores_test_HH_sample_yyjetsnode = []
        y_scores_test_yyjets_sample_yyjetsnode = []
        y_scores_test_GJets_sample_yyjetsnode = []
        y_scores_test_DY_sample_yyjetsnode = []

        # Arrays to store yyjets categorised event values
        y_scores_test_HH_sample_yyjets_categorised = []
        y_scores_test_yyjets_sample_yyjets_categorised = []
        y_scores_test_GJets_sample_yyjets_categorised = []
        y_scores_test_DY_sample_yyjets_categorised = []

        # Arrays to store all GJets node values
        y_scores_test_HH_sample_GJetsnode = []
        y_scores_test_yyjets_sample_GJetsnode = []
        y_scores_test_GJets_sample_GJetsnode = []
        y_scores_test_DY_sample_GJetsnode = []

        # Arrays to store GJets categorised events
        y_scores_test_HH_sample_GJets_categorised = []
        y_scores_test_yyjets_sample_GJets_categorised = []
        y_scores_test_GJets_sample_GJets_categorised = []
        y_scores_test_DY_sample_GJets_categorised = []

        # Arrays to store all DY node values
        y_scores_test_HH_sample_DYnode = []
        y_scores_test_yyjets_sample_DYnode = []
        y_scores_test_GJets_sample_DYnode = []
        y_scores_test_DY_sample_DYnode = []

        # Arrays to store DY categorised events
        y_scores_test_HH_sample_DY_categorised = []
        y_scores_test_yyjets_sample_DY_categorised = []
        y_scores_test_GJets_sample_DY_categorised = []
        y_scores_test_DY_sample_DY_categorised = []

        for i in range(len(result_probs_test)):
            test_event_weight = test_weights[i]
            if Y_test[i][0] == 1:
                y_scores_test_HH_sample_HHnode.append(result_probs_test[i][0])
                y_scores_test_HH_sample_yyjetsnode.append(result_probs_test[i][1])
                y_scores_test_HH_sample_GJetsnode.append(result_probs_test[i][2])
                y_scores_test_HH_sample_DYnode.append(result_probs_test[i][3])
                if np.argmax(result_probs_test[i]) == 0:
                    y_scores_test_HH_sample_HH_categorised.append(result_probs_test[i][0])
                if np.argmax(result_probs_test[i]) == 1:
                    y_scores_test_HH_sample_yyjets_categorised.append(result_probs_test[i][1])
                if np.argmax(result_probs_test[i]) == 2:
                    y_scores_test_HH_sample_GJets_categorised.append(result_probs_test[i][2])
                if np.argmax(result_probs_test[i]) == 3:
                    y_scores_test_HH_sample_DY_categorised.append(result_probs_test[i][3])
            if Y_test[i][1] == 1:
                y_scores_test_yyjets_sample_HHnode.append(result_probs_test[i][0])
                y_scores_test_yyjets_sample_yyjetsnode.append(result_probs_test[i][1])
                y_scores_test_yyjets_sample_GJetsnode.append(result_probs_test[i][2])
                y_scores_test_yyjets_sample_DYnode.append(result_probs_test[i][3])
                if np.argmax(result_probs_test[i]) == 0:
                    y_scores_test_yyjets_sample_HH_categorised.append(result_probs_test[i][0])
                if np.argmax(result_probs_test[i]) == 1:
                    y_scores_test_yyjets_sample_yyjets_categorised.append(result_probs_test[i][1])
                if np.argmax(result_probs_test[i]) == 2:
                    y_scores_test_yyjets_sample_GJets_categorised.append(result_probs_test[i][2])
                if np.argmax(result_probs_test[i]) == 3:
                    y_scores_test_yyjets_sample_DY_categorised.append(result_probs_test[i][3])
            if Y_test[i][2] == 1:
                y_scores_test_GJets_sample_HHnode.append(result_probs_test[i][0])
                y_scores_test_GJets_sample_yyjetsnode.append(result_probs_test[i][1])
                y_scores_test_GJets_sample_GJetsnode.append(result_probs_test[i][2])
                y_scores_test_GJets_sample_DYnode.append(result_probs_test[i][3])
                if np.argmax(result_probs_test[i]) == 0:
                    y_scores_test_GJets_sample_HH_categorised.append(result_probs_test[i][0])
                if np.argmax(result_probs_test[i]) == 1:
                    y_scores_test_GJets_sample_yyjets_categorised.append(result_probs_test[i][1])
                if np.argmax(result_probs_test[i]) == 2:
                    y_scores_test_GJets_sample_GJets_categorised.append(result_probs_test[i][2])
                if np.argmax(result_probs_test[i]) == 3:
                    y_scores_test_GJets_sample_DY_categorised.append(result_probs_test[i][3])
            if Y_test[i][3] == 1:
                y_scores_test_DY_sample_HHnode.append(result_probs_test[i][0])
                y_scores_test_DY_sample_yyjetsnode.append(result_probs_test[i][1])
                y_scores_test_DY_sample_GJetsnode.append(result_probs_test[i][2])
                y_scores_test_DY_sample_DYnode.append(result_probs_test[i][3])
                if np.argmax(result_probs_test[i]) == 0:
                    y_scores_test_DY_sample_HH_categorised.append(result_probs_test[i][0])
                if np.argmax(result_probs_test[i]) == 1:
                    y_scores_test_DY_sample_yyjets_categorised.append(result_probs_test[i][1])
                if np.argmax(result_probs_test[i]) == 2:
                    y_scores_test_DY_sample_GJets_categorised.append(result_probs_test[i][2])
                if np.argmax(result_probs_test[i]) == 3:
                    y_scores_test_DY_sample_DY_categorised.append(result_probs_test[i][3])

        # Create 2D lists (dimension 4x4) to hold max DNN discriminator values for each sample. One for train data, one for test data.
        #
        #               HH sample | yyjets sample | GJets sample | ttZ sample | DY sample
        # HH category
        # yyjets category
        # GJets category
        # ttZ category
        # DY category

        #w, h = 4, 4
        #y_scores_train = [[0 for x in range(w)] for y in range(h)]
        #y_scores_test = [[0 for x in range(w)] for y in range(h)]
        self.yscores_train_categorised[0] = [y_scores_train_HH_sample_HH_categorised, y_scores_train_yyjets_sample_HH_categorised, y_scores_train_GJets_sample_HH_categorised, y_scores_train_DY_sample_HH_categorised]
        self.yscores_train_categorised[1] = [y_scores_train_HH_sample_yyjets_categorised, y_scores_train_yyjets_sample_yyjets_categorised, y_scores_train_GJets_sample_yyjets_categorised, y_scores_train_DY_sample_yyjets_categorised]
        self.yscores_train_categorised[2] = [y_scores_train_HH_sample_GJets_categorised, y_scores_train_yyjets_sample_GJets_categorised, y_scores_train_GJets_sample_GJets_categorised, y_scores_train_DY_sample_GJets_categorised]
        self.yscores_train_categorised[3] = [y_scores_train_HH_sample_DY_categorised, y_scores_train_yyjets_sample_DY_categorised, y_scores_train_GJets_sample_DY_categorised, y_scores_train_DY_sample_DY_categorised]

        self.yscores_test_categorised[0] = [y_scores_test_HH_sample_HH_categorised, y_scores_test_yyjets_sample_HH_categorised, y_scores_test_GJets_sample_HH_categorised, y_scores_test_DY_sample_HH_categorised]
        self.yscores_test_categorised[1] = [y_scores_test_HH_sample_yyjets_categorised, y_scores_test_yyjets_sample_yyjets_categorised, y_scores_test_GJets_sample_yyjets_categorised, y_scores_test_DY_sample_yyjets_categorised]
        self.yscores_test_categorised[2] = [y_scores_test_HH_sample_GJets_categorised, y_scores_test_yyjets_sample_GJets_categorised, y_scores_test_GJets_sample_GJets_categorised, y_scores_test_DY_sample_GJets_categorised]
        self.yscores_test_categorised[3] = [y_scores_test_HH_sample_DY_categorised, y_scores_test_yyjets_sample_DY_categorised, y_scores_test_GJets_sample_DY_categorised, y_scores_test_DY_sample_DY_categorised]

        #y_scores_train_nonCat = [[0 for x in range(w)] for y in range(h)]
        #y_scores_test_nonCat = [[0 for x in range(w)] for y in range(h)]
        self.yscores_train_non_categorised[0] = [y_scores_train_HH_sample_HHnode, y_scores_train_yyjets_sample_HHnode, y_scores_train_GJets_sample_HHnode, y_scores_train_DY_sample_HHnode]
        self.yscores_train_non_categorised[1] = [y_scores_train_HH_sample_yyjetsnode, y_scores_train_yyjets_sample_yyjetsnode, y_scores_train_GJets_sample_yyjetsnode, y_scores_train_DY_sample_yyjetsnode]
        self.yscores_train_non_categorised[2] = [y_scores_train_HH_sample_GJetsnode, y_scores_train_yyjets_sample_GJetsnode, y_scores_train_GJets_sample_GJetsnode, y_scores_train_DY_sample_GJetsnode]
        self.yscores_train_non_categorised[3] = [y_scores_train_HH_sample_DYnode, y_scores_train_yyjets_sample_DYnode, y_scores_train_GJets_sample_DYnode, y_scores_train_DY_sample_DYnode]

        self.yscores_test_non_categorised[0] = [y_scores_test_HH_sample_HHnode, y_scores_test_yyjets_sample_HHnode, y_scores_test_GJets_sample_HHnode, y_scores_test_DY_sample_HHnode]
        self.yscores_test_non_categorised[1] = [y_scores_test_HH_sample_yyjetsnode, y_scores_test_yyjets_sample_yyjetsnode, y_scores_test_GJets_sample_yyjetsnode, y_scores_test_DY_sample_yyjetsnode]
        self.yscores_test_non_categorised[2] = [y_scores_test_HH_sample_GJetsnode, y_scores_test_yyjets_sample_GJetsnode, y_scores_test_GJets_sample_GJetsnode, y_scores_test_DY_sample_GJetsnode]
        self.yscores_test_non_categorised[3] = [y_scores_test_HH_sample_DYnode, y_scores_test_yyjets_sample_DYnode, y_scores_test_GJets_sample_DYnode, y_scores_test_DY_sample_DYnode]

        node_name = ['HH','yyjets','GJets','DY']
        counter = 0
        for y_scorestrain,y_scorestest in zip(self.yscores_train_categorised,self.yscores_test_categorised):
            colours = ['r','steelblue','g','Fuchsia','darkgoldenrod']
            node_title = node_name[counter]
            plot_title = 'Categorised'
            plot_info = [node_name,colours,data_type,plots_dir,node_title,plot_title]
            self.separations_categories.append(self.draw_category_overfitting_plot(y_scorestrain,y_scorestest,plot_info))
            counter = counter +1

        counter =0
        separations_all = []
        for y_scores_train_nonCat,y_scores_test_nonCat in zip(self.yscores_train_non_categorised,self.yscores_test_non_categorised):
            colours = ['r','steelblue','g','Fuchsia','darkgoldenrod']
            node_title = node_name[counter]
            plot_title = 'Non-Categorised'
            plot_info = [node_name,colours,data_type,plots_dir,node_title,plot_title]
            separations_all.append(self.draw_category_overfitting_plot(y_scores_train_nonCat,y_scores_test_nonCat,plot_info))
            counter = counter +1

        return

    def binary_overfitting(self, estimator, Y_train, Y_test, result_probs, result_probs_test, plots_dir, train_weights, test_weights, nbins=50):

        model = estimator
        data_type = type(model)

        #Arrays to store all results
        y_scores_train_signal_sample = []
        y_scores_train_bckg_sample = []
        y_scores_test_signal_sample = []
        y_scores_test_bckg_sample = []
        for i in range(0,len(result_probs)-1):
            train_event_weight = train_weights[i]
            if Y_train[i] == 1:
                y_scores_train_signal_sample.append(result_probs[i])
            if Y_train[i] == 0:
                y_scores_train_bckg_sample.append(result_probs[i])
        for i in range(0,len(result_probs_test)-1):
            test_event_weight = test_weights[i]
            if Y_test[i] == 1:
                y_scores_test_signal_sample.append(result_probs_test[i])
            if Y_test[i] == 0:
                y_scores_test_bckg_sample.append(result_probs_test[i])

        # Create 2D lists (dimension 2x2) to hold max DNN discriminator values for each sample. One for train data, one for test data.
        yscores_train_binary=[]
        yscores_test_binary=[]
        yscores_train_binary.append([y_scores_train_signal_sample, y_scores_train_bckg_sample])
        yscores_test_binary.append([y_scores_test_signal_sample, y_scores_test_bckg_sample])

        counter =0
        separations_all = []
        for y_scores_train_nonCat,y_scores_test_nonCat in zip(yscores_train_binary,yscores_test_binary):
            colours = ['r','steelblue']
            plot_title = 'Binary'
            plot_info = [colours,data_type,plots_dir,plot_title]
            separations_all.append(self.draw_binary_overfitting_plot(y_scores_train_nonCat,y_scores_test_nonCat,plot_info,test_weights))
            counter = counter+1

        return
    def plot_dot(self, title, x, shap_values, column_headers):
        plt.figure()
        if x is None:
          print('<plotter> No x defined. Leaving class function')
          return
        shap.summary_plot(shap_values[0], features=x, feature_names=column_headers, show=False, max_display=10)
        plt.gca().set_title(title)
        plt.tight_layout()
        plt.savefig("{}/plots/{}.png".format(self.output_directory, title), bbox_inches='tight')

    def plot_dot_bar(self, title, x, shap_values, column_headers):
        plt.figure()
        if x is None:
            print('<plotter> No x defined. Leaving class function')
            return
        shap.summary_plot(shap_values[0], features=x, feature_names=column_headers, show=False,plot_type='bar',max_display=10)
        plt.gca().set_title(title)
        plt.tight_layout()
        plt.savefig("{}/plots/{}.png".format(self.output_directory,title), bbox_inches='tight')

    def plot_dot_bar_all(self , title, x, shap_values, column_headers):
        plt.figure()
        if x is None:
            print('<plotter> No x defined. Leaving class function')
            return
        shap.summary_plot(shap_values[0], features=x, feature_names=column_headers, show=False,plot_type='bar',max_display=len(column_headers))
        plt.gca().set_title(title)
        plt.tight_layout()
        plt.savefig("{}/plots/{}.png".format(self.output_directory,title), bbox_inches='tight')
