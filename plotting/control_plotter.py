##############################################
#           control_plotter.py
##############################################
# Author: Joshuha Thomas-Wilsker
# Class to be used in scripts to make control
# plots of analysis distributions. Stacked
# MC and data comparison.
##############################################
#To do:
# - Add ratio plot.

import ROOT, os, optparse
import pickle
from ROOT import gROOT, TMVA, TFile, TString, TLegend, THStack, TLatex, TH1D, TH1F
from array import array
from subprocess import call
from os.path import isfile
from collections import OrderedDict
import commands

class control_plotter(object):

    def __init__(self):
        self.output_fullpath = ''

    #Used to save pyplot images
    def save_plots(self, filename, dir='plots/'):
        self.check_dir(dir)
        filepath = os.path.join(dir,filename)
        self.fig.savefig(filepath)
        return self.fig

    def check_dir(self, dir):
        if not os.path.exists(dir):
            print 'Creating directory ',dir
            os.makedirs(dir)

    def getEOSlslist(self, directory, mask='', prepend='root://eosuser.cern.ch/'):
        eos_dir = '/eos/user/%s ' % (directory)
        eos_cmd = 'eos ' + prepend + ' ls ' + eos_dir
        out = commands.getoutput(eos_cmd)
        full_list = []
        ## if input file was single root file:
        if directory.endswith('.root'):
            if len(out.split('\n')[0]) > 0:
                return [os.path.join(prepend,eos_dir).replace(" ","")]
        ## instead of only the file name append the string to open the file in ROOT
        for line in out.split('\n'):
            print 'line: ', line
            if len(line.split()) == 0: continue
            full_list.append(os.path.join(prepend,eos_dir,line).replace(" ",""))
        ## strip the list of files if required
        if mask != '':
            stripped_list = [x for x in full_list if mask in x]
            return stripped_list
        print 'full files list from eos: ', full_list
        ## return
        return full_list

    def open_files(self, input_files_names):
        input_files_list = []
        print 'input_files_names: ', input_files_names
        for index in xrange(0,len(input_files_names)):
            input_files_list.append(TFile.Open(input_files_names[index]))
        return input_files_list

    def define_binning(self, branch_):
        nbinsx = 10
        maxX = 10
        minX = 0
        print 'Branch = ', branch_
        if 'n_presel_jet' in branch_ or 'nBJetLoose' in branch_ or 'nBJetMedium' in branch_:
            nbinsx = 10
            maxX = 10
            minX = 0
        elif '_pt' in branch_ or 'conePt' in branch_ or 'met' in branch_ or 'mass' in branch_ or 'mbb' in branch_ or 'mT_' in branch_ :
            nbinsx = 50
            maxX = 300
            minX = 0
        elif 'eta' in branch_:
            nbinsx = 10
            maxX = 3
            minX = -3
        elif 'resTop' in branch_ or 'hadTop' in branch_:
            nbinsx = 20
            maxX = 1
            minX = -1
        elif 'mindr' in branch_:
            nbinsx = 10
            maxX = 10
            minX = 0
        elif 'charge' in branch_:
            nbinsx = 20
            maxX = 1
            minX = -1
        if 'jetFwd1_eta' in branch_:
            nbinsx = 20
            maxX = 5
            minX = -5
        return [nbinsx,minX,maxX]


    def load_histos(self, input_files_, branches_, treename_,selection):
        input_hist_dict = OrderedDict([])
        file_index = 0
        for file_ in input_files_:
            print '<load_histos> file_: ', file_.GetName()
            tree_ = file_.Get(treename_)
            temp_percentage_done = 0
            for branch_ in branches_:
                print '<load_histos> branch: ', branch_
                binning_ = self.define_binning(branch_)
                htemp = TH1F(branch_,branch_,binning_[0],binning_[1],binning_[2])
                htemp.SetMinimum(0.)
                if 'Data' in file_.GetName() or 'ttJets' in file_.GetName():
                    #maxentries = 10000
                    maxentries = tree_.GetEntries()
                else:
                    #maxentries = 10000
                    maxentries = tree_.GetEntries()
                print '<load_histos> # Entries = ', tree_.GetEntries()
                # Turn off all branches and only turn on the ones you want to use.
                # This will make the scrip run a lot faster.
                tree_.SetBranchStatus("*",0)
                tree_.SetBranchStatus("n_presel_jet",1)
                tree_.SetBranchStatus("is_tH_like_and_not_ttH_like",1)
                tree_.SetBranchStatus(branch_,1)
                tree_.SetBranchStatus('EventWeight',1)

                for i in range(tree_.GetEntries()):
                    percentage_done = int(100*float(i)/float(tree_.GetEntries()))
                    if percentage_done % 10 == 0:
                        if percentage_done != temp_percentage_done:
                            temp_percentage_done = percentage_done
                            print '%.2f percent done' % (temp_percentage_done)
                    tree_.GetEntry(i)
                    # Apply selection
                    selection_criteria = 0
                    nJets_ = tree_.GetBranch('n_presel_jet').GetLeaf('n_presel_jet')
                    is_tH_like_and_not_ttH_like_ = tree_.GetBranch('is_tH_like_and_not_ttH_like').GetLeaf('is_tH_like_and_not_ttH_like')
                    selection_criteria = 1 if ((tree_.is_tH_like_and_not_ttH_like==0 or tree_.is_tH_like_and_not_ttH_like==1)) else 0
                    if selection_criteria == 0:
                        continue
                    variable_ = tree_.GetBranch(branch_).GetLeaf(branch_)
                    weight_ = tree_.GetBranch('EventWeight').GetLeaf('EventWeight')
                    htemp.Fill(variable_.GetValue(), weight_.GetValue())

                keyname_file = file_.GetName().split('/')[-1:]
                keyname = keyname_file[0].split('.')[:-1]
                keyname = keyname[0] + '_' + branch_
                keyname = keyname + '_DiLepRegion'
                print 'filling input_hist_dict with key %s' % (keyname)
                input_hist_dict[keyname] = htemp
                del htemp
            del tree_
            del file_

        return input_hist_dict

    def make_hist(self, input_hist_1, input_hist_2, input_hist_3, process, variable_name):
        c1 = ROOT.TCanvas('c1',',1000,1000')
        p1 = ROOT.TPad('p1','p1',0.0,0.0,1.0,1.0)
        p1.Draw()
        p1.SetRightMargin(0.1)
        p1.SetLeftMargin(0.1)
        p1.SetBottomMargin(0.1)
        p1.SetTopMargin(0.1)
        p1.SetGridx(True)
        p1.SetGridy(True)
        p1.cd()
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetOptTitle(0)
        hist_count = 0
        maxy = 0
        legend = ROOT.TLegend(0.7,0.8,0.9,0.9)

        input_hist_1.Scale(1/input_hist_1.Integral())
        input_hist_2.Scale(1/input_hist_2.Integral())
        input_hist_3.Scale(1/input_hist_3.Integral())

        maxy = input_hist_1.GetMaximum()*1.2 if input_hist_1.GetMaximum()>input_hist_2.GetMaximum() else input_hist_2.GetMaximum()*1.2
        if input_hist_3.GetMaximum() > maxy:
            maxy = input_hist_3.GetMaximum()*1.2

        histo_title_1 = '%s Signal Region' % (process)
        input_hist_1.SetTitle(histo_title_1)
        input_hist_1.SetLineColor(1)
        input_hist_1.GetXaxis().SetTitle('Arbitrary Units')
        input_hist_1.GetXaxis().SetTitle(variable_name)
        input_hist_1.SetMaximum(maxy)

        histo_title_2 = '%s Loose Training Region' % (process)
        input_hist_2.SetTitle(histo_title_2)
        input_hist_2.SetLineColor(2)

        histo_title_3 = '%s Fakeable Training Region' % (process)
        input_hist_3.SetTitle(histo_title_3)
        input_hist_3.SetLineColor(3)

        input_hist_1.Draw('HIST')
        legend.AddEntry(input_hist_1,input_hist_1.GetTitle(),"l")

        input_hist_2.Draw('HISTSAME')
        legend.AddEntry(input_hist_2,input_hist_2.GetTitle(),"l")

        input_hist_3.Draw('HISTSAME')
        legend.AddEntry(input_hist_3,input_hist_3.GetTitle(),"l")

        legend.Draw('SAME')
        canvas_title = 'Samples: %s ' % process
        c1.SetTitle(canvas_title)
        #slef.output_fullpath = 'invar_control_plots_190319/' + input_hist_2.GetName() + '_' + process + '.png'
        c1.SaveAs(self.output_fullpath,'png')
        #delete c1
        return

    def make_comparison(self, input_hist1, title_hist1, input_hist2, title_hist2, variable_name, output_fullpath):

        c1 = ROOT.TCanvas('c1',',1000,1000')
        p1 = ROOT.TPad('p1','p1',0.0,0.0,1.0,1.0)
        p1.Draw()
        p1.SetRightMargin(0.1)
        p1.SetLeftMargin(0.1)
        p1.SetBottomMargin(0.1)
        p1.SetTopMargin(0.1)
        p1.SetGridx(True)
        p1.SetGridy(True)
        p1.cd()
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetOptTitle(0)
        hist_count = 0
        maxy = 0
        legend = TLegend(0.7,  0.7,  0.9,  0.9)
        #legend.SetNColumns(2)

        if input_hist1.Integral() != 0:
            input_hist1.Scale(1/input_hist1.Integral())
        if input_hist2.Integral() != 0:
            input_hist2.Scale(1/input_hist2.Integral())

        maxy = input_hist1.GetMaximum()*1.2 if input_hist1.GetMaximum()>input_hist2.GetMaximum() else input_hist2.GetMaximum()*1.2

        input_hist1.SetMarkerColor(1)
        input_hist1.SetMarkerStyle(20)
        input_hist1.SetLineColor(1)
        input_hist1.SetFillColor(1)
        input_hist1.SetFillStyle(3002)
        input_hist1.SetTitle(title_hist1)
        input_hist1.GetYaxis().SetTitle('Events')
        input_hist1.GetXaxis().SetTitle(variable_name)
        input_hist1.SetMaximum(maxy)
        input_hist1.Draw('HIST')
        legend.AddEntry(input_hist1,title_hist1,'l')

        input_hist2.SetMarkerColor(2)
        input_hist2.SetMarkerStyle(20)
        input_hist2.SetLineColor(2)
        input_hist2.SetFillColor(2)
        input_hist2.SetFillStyle(3002)
        input_hist2.SetTitle(title_hist2)
        input_hist2.GetYaxis().SetTitle('Events')
        input_hist2.GetXaxis().SetTitle(variable_name)
        input_hist2.SetMaximum(maxy)
        input_hist2.Draw('HISTSAMEEP')
        legend.AddEntry(input_hist2,title_hist2,'p')

        legend.Draw('SAME')
        c1.Update()
        c1.SaveAs(self.output_fullpath,'png')
        return

    def sum_hists(self, hists_, title):
        binning_ = self.define_binning(hists_[0].GetName())
        combined_hist = ROOT.TH1F(title,title,binning_[0],binning_[1],binning_[2])
        for hist_ in hists_:
            combined_hist.Add(hist_)
            combined_hist.SetTitle(title)
        return combined_hist

    def stack_hists(self, hists_, title, variable_name, input_hist_data, dodata=0):

        c1 = ROOT.TCanvas('c1',',1000,1000')
        p1 = ROOT.TPad('p1','p1',0.0,0.0,1.0,1.0)
        p1.Draw()
        p1.SetRightMargin(0.1)
        p1.SetLeftMargin(0.1)
        p1.SetBottomMargin(0.1)
        p1.SetTopMargin(0.1)
        p1.SetGridx(True)
        p1.SetGridy(True)
        p1.cd()
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetOptTitle(0)
        hist_count = 0
        maxy = 0
        legend = TLegend(0.7,  0.7,  0.9,  0.9)
        legend.SetNColumns(2)

        process_list = OrderedDict([
        ("Conv" , 5),
        ("EWK" , 6),
        ("Fakes" , 13),
        ("Flips" , 17),
        ("Rares" , 38),
        ("TTW" , 8),
        ("TTZ" , 9),
        ("TTWW" , 3),
        ("TTH_hww" , 2),
        ("TTH_hzz" , 2),
        ("TTH_htt" , 2),
        ("TTH_hot" , 2),
        ("TTH_hmm" , 2),
        ("Data" , 1)
        ])

        stacked_hist = ROOT.THStack()
        #binning_ = self.define_binning(input_hist_data.GetName())
        for name, hist_ in hists_.iteritems():
            print 'name = ', name
            for key in process_list:
                if key in name:
                    if key == 'TTW' and 'TTWW' in name:
                        continue
                    print 'add %s to legend' % (key)
                    hist_.SetMarkerColor(process_list[key])
                    hist_.SetMarkerStyle(20)
                    hist_.SetLineColor(process_list[key])
                    hist_.SetFillColor(process_list[key])
                    #hist_.SetFillStyle(3002)
                    if 'Data' in name:
                        legend.AddEntry(hist_,key,'p')
                    else:
                        legend.AddEntry(hist_,key,'f')
            if 'Data' in name:
                continue
            stacked_hist.Add(hist_)
            stacked_hist.SetTitle(title)
        stacked_hist.SetMaximum(stacked_hist.GetStack().Last().GetMaximum() + (stacked_hist.GetStack().Last().GetMaximum()/2))
        stacked_hist.SetMinimum(0.)

        #input_hist_data = hists_.get(data_hist_name)
        maxy = stacked_hist.GetMaximum()*1.2 if stacked_hist.GetStack().Last().GetMaximum()>input_hist_data.GetMaximum() else input_hist_data.GetMaximum()*1.2

        histo_title_mc = 'MC'
        stacked_hist.Draw('HIST')
        stacked_hist.SetTitle(histo_title_mc)
        stacked_hist.GetYaxis().SetTitle('Events')
        stacked_hist.GetXaxis().SetTitle(variable_name)
        stacked_hist.SetMaximum(maxy)

        if dodata == 1:
            histo_title_data = 'DATA'
            input_hist_data.Draw('HISTSAMEEP')
            input_hist_data.SetTitle(histo_title_data)
            input_hist_data.GetYaxis().SetTitle('Events')
            input_hist_data.GetXaxis().SetTitle(variable_name)
            input_hist_data.SetMaximum(maxy)

        legend.Draw('SAME')
        c1.Update()
        c1.SaveAs(self.output_fullpath,'png')

        return stacked_hist
