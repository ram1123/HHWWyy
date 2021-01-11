import ROOT, os, argparse, math
import numpy as np
from ROOT import TMVA, TFile, TString, TLegend, THStack, TLatex, TH1D
from array import array
from subprocess import call
from os.path import isfile
from collections import OrderedDict

def GetHistoScale(histo):
    dx_hist_train = (histo.GetXaxis().GetXmax() - histo.GetXaxis().GetXmin()) / histo.GetNbinsX()
    hist_train_norm = 1./histo.GetSumOfWeights()/dx_hist_train
    histo.Scale(hist_train_norm)
    return histo

def GetSeparation(hist_sig, hist_bckg):

    # compute "separation" defined as
    # <s2> = (1/2) Int_-oo..+oo { (S(x) - B(x))^2/(S(x) + B(x)) dx }
    separation = 0;
    # sanity check: signal and background histograms must have same number of bins and same limits
    if hist_sig.GetNbinsX() != hist_bckg.GetNbinsX():
        print('Number of bins different for sig. and bckg')

    # Histogram scaled per entry per unit of X
    nBins = hist_sig.GetNbinsX()
    #dX = (hist_sig.GetXaxis().GetXmax() - hist_sig.GetXaxis().GetXmin()) / hist_sig.GetNbinsX()
    nS = hist_sig.GetSumOfWeights()
    nB = hist_bckg.GetSumOfWeights()

    if nS == 0:
        print('WARNING: no signal weights')
    if nB == 0:
        print('WARNING: no bckg weights')
    sig_bin_norm_sum=0
    bckg_bin_norm_sum=0
    for i in range(1,nBins):
        if nS != 0:
            sig_bin_norm = hist_sig.GetBinContent(i)/nS
            sig_bin_norm_sum += sig_bin_norm
        else: continue
        if nB != 0 :
            bckg_bin_norm = hist_bckg.GetBinContent(i)/nB
            bckg_bin_norm_sum += bckg_bin_norm
        else: continue
        # Separation:
        if(sig_bin_norm+bckg_bin_norm > 0):
            separation += 0.5 * ((sig_bin_norm - bckg_bin_norm) * (sig_bin_norm - bckg_bin_norm)) / (sig_bin_norm + bckg_bin_norm)
    #separation *= dX

    return separation

def GetDataOverMC(stack_mc, histo_data):

    #First on stack goes on bottom.
    DOverMC = histo_data.Clone('ratioframe')
    DOverMC.Divide(stack_mc.GetStack().Last())
    DOverMC.GetYaxis()
    DOverMC_maxY = DOverMC.GetMaximum()
    DOverMC_minY = DOverMC.GetMinimum()
    DOverMC.GetYaxis().SetTitle('Data/MC')
    DOverMC_maximum = DOverMC_maxY+(DOverMC_maxY*0.1)
    DOverMC_minimum = DOverMC_minY+(DOverMC_minY*0.1)
    if DOverMC_maximum < 2.:
        DOverMC.GetYaxis().SetRangeUser(0.5,DOverMC_maxY+(DOverMC_maxY*0.1))
    else:
        DOverMC.GetYaxis().SetRangeUser(0.5,2.)
    DOverMC.GetYaxis().SetNdivisions(6)
    DOverMC.GetYaxis().SetLabelSize(0.12)
    DOverMC.GetYaxis().SetTitleSize(0.12)
    DOverMC.GetYaxis().SetTitleOffset(0.2)
    DOverMC.GetXaxis().SetTitle('DNN Response')
    DOverMC.GetXaxis().SetLabelSize(0.15)
    DOverMC.GetXaxis().SetTitleSize(0.15)
    DOverMC.GetXaxis().SetTitleOffset(1.)
    DOverMC.SetFillStyle(0)
    DOverMC.SetMarkerStyle(2)
    DOverMC.SetMarkerColor(1)
    DOverMC.SetLineColor(1)
    return DOverMC

def GetSOverRootB(sig, bckg):
    last = bckg.GetStack().Last()
    nBins = last.GetNbinsX()
    SOverRootB_hist = ROOT.TH1F('SOverRootB_hist','SOverRootB_hist',nBins,0,1)
    for ibin in range(1,last.GetNbinsX()+1):
        if last.GetBinContent(ibin) > 0.:
            signal_ = sig.GetBinContent(ibin)
            sqrt_B = math.sqrt(last.GetBinContent(ibin))
            soverrootb = (signal_/sqrt_B)
            SOverRootB_hist.SetBinContent(ibin,soverrootb)
        else:
            SOverRootB_hist.SetBinContent(ibin,0.)
    return SOverRootB_hist

def rebinHistograms(sample_info):
    original_signal_hist = sample_info.get('HHWWgg')[2]
    original_bckg_hist = sample_info.get('DiPhoton')[2]
    nBins = original_bckg_hist.GetNbinsX()
    x_bin_edges = []
    cumulative_bckg_entries = 0
    cumulative_sig_entries = 0
    combined_bckg_hist_values = []
    combined_sig_hist_values = []
    combined_bckg_hist = ROOT.TH1F('bckg_hists','bckg_hists',nBins,0,1)
    combined_sig_hist = ROOT.TH1F('sig_hists','sig_hists',nBins,0,1)
    for x_bin_index in range(0,nBins):
        tmp_bin_content = original_signal_hist.GetBinContent(x_bin_index)
        for hist_key, info in sample_info.items():
            hist = info[2]
            if 'Data' in hist_key: continue
            elif 'HHWWgg' in hist_key:
                cumulative_sig_entries = cumulative_sig_entries + hist.GetBinContent(x_bin_index)
                combined_sig_hist.AddBinContent(x_bin_index,hist.GetBinContent(x_bin_index))
                continue
            else:
                cumulative_bckg_entries = cumulative_bckg_entries + hist.GetBinContent(x_bin_index)
            combined_bckg_hist.AddBinContent(x_bin_index,hist.GetBinContent(x_bin_index))
        if cumulative_bckg_entries!=0 and tmp_bin_content/cumulative_bckg_entries>=0.02:
            new_x_bin_edge = original_bckg_hist.GetXaxis().GetBinUpEdge(x_bin_index)
            x_bin_edges.append(new_x_bin_edge)
            cumulative_bckg_entries = 0

    ordered_bin_edges = [0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    n_xbins = len(ordered_bin_edges)-1
    x_bin_edges_fuckyouroot = array('d',ordered_bin_edges)

    for hist_key, info in sample_info.items():
        hist = info[2]
        sample_info[hist_key][2] = hist.Rebin(n_xbins, hist.GetName(), x_bin_edges_fuckyouroot)

    return sample_info, x_bin_edges_fuckyouroot

def make_plot(stacked_hist, signal_hist, norm, legend, inputs_directory, separation_, ratioplot = None, signal_scale=1., data_hist = None):

    c1 = ROOT.TCanvas('c1','c1',1000,1000)
    #c1.SetLogy()
    p1 = ROOT.TPad('p1','p1',0.0,0.2,1.0,1.0)
    p1.Draw()
    #p1.SetLogy()
    p1.SetRightMargin(0.1)
    p1.SetLeftMargin(0.1)
    p1.SetBottomMargin(0.1)
    p1.SetTopMargin(0.1)
    p1.SetGridx(True)
    p1.SetGridy(True)
    p1.cd()
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    suffix = ''

    classifier_plots_dir = os.path.join('samples_w_DNN',inputs_directory,'plots')
    print('<DNN_Evaluation_Control_Plots> using plots dir: ', classifier_plots_dir)
    if not os.path.exists(classifier_plots_dir):
        os.makedirs(classifier_plots_dir)

    stacked_hist.Draw("HIST")
    stacked_hist.GetYaxis().SetTitle('Events/Bin')
    stacked_hist.GetXaxis().SetTitle('DNN Output')

    if data_hist != None:
        data_hist.SetMarkerStyle(20)
        data_hist.SetMarkerColor(1)
        data_hist.SetLineColor(1)
        data_hist.Draw("HISTEPSAME")

    signal_hist.Scale(signal_scale)
    signal_hist.SetLineColor(2)
    signal_hist.Draw("HISTSAME")
    legend.Draw("same")

    txt2=ROOT.TLatex()
    txt2.SetNDC(True)
    txt2.SetTextFont(43)
    txt2.SetTextSize(18)
    txt2.SetTextAlign(12)
    txt2.DrawLatex(0.13,0.925,'#bf{CMS}')
    txt2.DrawLatex(0.2,0.92,'#it{Preliminary}')
    txt2.DrawLatex(0.57,0.925,'%3.1f fb^{-1} (13 TeV)' %(41860.080/1000.) )

    # Add custom title
    l1=ROOT.TLatex()
    l1.SetNDC()
    l1.SetTextFont(43)
    l1.SetTextSize(15)
    l1.SetTextAlign(12)
    latex_title = ''
    latex_title = "#it{prefit: DNN}"

    l1.DrawLatex(0.15,0.8,latex_title)

    l2=ROOT.TLatex()
    l2.SetNDC()
    l2.SetTextFont(43)
    l2.SetTextSize(12)
    separation_title = 'Separation = %s' % ("{0:.5g}".format(separation_))
    l2.DrawLatex(0.15,0.75,separation_title)

    # Draw Data/MC ratio plot
    c1.cd()
    p2 = ROOT.TPad('p2','p2',0.0,0.0,1.0,0.2)
    p2.Draw()
    p2.SetLeftMargin(0.1)
    p2.SetRightMargin(0.1)
    p2.SetTopMargin(0.05)
    p2.SetBottomMargin(0.4)
    p2.SetGridx(True)
    p2.SetGridy(True)
    p2.cd()
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)

    if ratioplot != None:
        ratioplot_maxY = ratioplot.GetMaximum()
        ratioplot.GetYaxis().SetTitle('S/#sqrt{B}')
        ratioplot_maximum = ratioplot_maxY+(ratioplot_maxY*0.1)
        ratioplot.GetYaxis().SetRangeUser(0.,ratioplot_maxY+(ratioplot_maxY*0.1))
        ratioplot.GetYaxis().SetNdivisions(6)
        ratioplot.GetYaxis().SetLabelSize(0.12)
        ratioplot.GetYaxis().SetTitleSize(0.15)
        ratioplot.GetYaxis().SetTitleOffset(0.2)
        ratioplot.GetXaxis().SetTitle('Response')
        ratioplot.GetXaxis().SetLabelSize(0.15)
        ratioplot.GetXaxis().SetTitleSize(0.15)
        ratioplot.GetXaxis().SetTitleOffset(1.)
        ratioplot.SetFillStyle(0)
        ratioplot.SetMarkerStyle(2)
        ratioplot.SetMarkerColor(1)
        ratioplot.SetLineColor(1)
        ratioplot.Draw("P")
        line = ROOT.TLine(0,1,1,1);
        line.SetLineColor(2);
        line.Draw("same");
        c1.Update()

    outfile_name = 'DNNOutput%s.pdf'%(suffix)
    output_fullpath = classifier_plots_dir + '/' + outfile_name
    c1.Print(output_fullpath,'pdf')
    return

def separation_table(outputdir,separation_dictionary):
    content = r'''\documentclass{article}
\begin{document}
\begin{center}
\begin{table}
\begin{tabular}{| c | c | c | c | c |} \hline
Option \textbackslash Node & ttH & ttJ & ttW & ttZ \\ \hline
%s \\
\hline
\end{tabular}
\caption{Separation power on each output node. The separation is given with respect to the `signal' process the node is trained to separate (one node per column) and the combined background processes for that node. The three options represent the different mehtods to of class weights in the DNN training.}
\end{table}
\end{center}
\end{document}
'''
    table_path = os.path.join(outputdir,'separation_table')
    table_tex = table_path+'.tex'
    print('table_tex: ', table_tex)
    with open(table_tex,'w') as f:
        option_1_entry = '%s & %s & %s & %s & %s' % ('Option 1', "{0:.5g}".format(separation_dictionary['option1'][0]), "{0:.5g}".format(separation_dictionary['option1'][1]), "{0:.5g}".format(separation_dictionary['option1'][2]), "{0:.5g}".format(separation_dictionary['option1'][3]))
        f.write( content % (option_1_entry) )
    return

def main():
    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('-d', '--data',        dest='data_flag'  ,      help='1 = include data from plots, 0 = exclude data from plots', default=0, type=int)
    parser.add_argument('-i', '--in_dir', dest='in_dir', help='Option to choose directory containing samples. Choose directory from samples_w_DNN', default='', type=str)
    args = parser.parse_args()
    data_flag = args.data_flag
    inputs_directory = args.in_dir
    classifier_samples_dir = os.path.join("samples_w_DNN/",inputs_directory)
    print('Reading samples from: ', classifier_samples_dir)

    signal_scale=10000.
    #additional_info = [inputs_directory]

    sample_info = OrderedDict([
    ('HHWWgg' , ['HHWWgg-SL-SM-NLO-2017',2]),
    ('DiPhoton' , ['DiPhotonJetsBox_MGG-80toInf_13TeV-Sherpa_Hadded',1]),
    ('GJet_Pt-20toInf' , ['GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_13TeV_Pythia8_Hadded',28]),
    ('GJet_Pt-20to40' , ['GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8_Hadded',28]),
    ('GJet_Pt-40toInf' , ['GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8_Hadded',28]),
    ('DYJetsToLL_M-50' , ['DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_Hadded',4]),
    ('TTGJets' , ['TTGJets_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_Hadded',5]),
    ('TTGG' , ['TTGG_0Jets_TuneCP5_13TeV_amcatnlo_madspin_pythia8_Hadded',6]),
    ('TTJets_HT-600to800' , ['TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8_Hadded',7]),
    ('TTJets_HT-800to1200' , ['TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8_Hadded',7]),
    ('TTJets_HT-1200to2500' , ['TTJets_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8_Hadded',7]),
    ('TTJets_HT-2500toInf' , ['TTJets_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8_Hadded',7]),
    #('W1JetsToLNu_LHEWpT_0-50' , ['W1JetsToLNu_LHEWpT_0-50_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',8]),
    ('W1JetsToLNu_LHEWpT_50-150' , ['W1JetsToLNu_LHEWpT_50-150_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',8]),
    ('W1JetsToLNu_LHEWpT_150-250' , ['W1JetsToLNu_LHEWpT_150-250_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',8]),
    ('W1JetsToLNu_LHEWpT_250-400' , ['W1JetsToLNu_LHEWpT_250-400_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',8]),
    ('W1JetsToLNu_LHEWpT_400-inf' , ['W1JetsToLNu_LHEWpT_400-inf_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',8]),
    ('W2JetsToLNu_LHEWpT_0-50' , ['W2JetsToLNu_LHEWpT_0-50_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',8]),
    ('W2JetsToLNu_LHEWpT_50-150' , ['W2JetsToLNu_LHEWpT_50-150_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',8]),
    ('W2JetsToLNu_LHEWpT_150-250' , ['W2JetsToLNu_LHEWpT_150-250_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',8]),
    ('W2JetsToLNu_LHEWpT_250-400' , ['W2JetsToLNu_LHEWpT_250-400_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',8]),
    ('W2JetsToLNu_LHEWpT_400-inf' , ['W2JetsToLNu_LHEWpT_400-inf_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',8]),
    ('W3JetsToLNu' , ['W3JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_Hadded',8]),
    ('W4JetsToLNu' , ['W4JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_Hadded',8]),
    ('ttHJetToGG' , ['ttHJetToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8_Hadded',9])
    ])
    #'Data' : ('Data_'+JESname+region)

    input_name_HH = '%s/%s.root' % (classifier_samples_dir,sample_info.get('HHWWgg')[0])
    input_file_HH = TFile.Open(input_name_HH)
    input_name_DiPhoton = '%s/%s.root' % (classifier_samples_dir,sample_info.get('DiPhoton')[0])
    input_file_DiPhoton = TFile.Open(input_name_DiPhoton)
    input_name_GJet_Pt_20toInf = '%s/%s.root' % (classifier_samples_dir,sample_info.get('GJet_Pt-20toInf')[0])
    input_file_GJet_Pt_20toInf = TFile.Open(input_name_GJet_Pt_20toInf)
    input_name_GJet_Pt_20to40 = '%s/%s.root' % (classifier_samples_dir,sample_info.get('GJet_Pt-20to40')[0])
    input_file_GJet_Pt_20to40 = TFile.Open(input_name_GJet_Pt_20to40)
    input_name_GJet_Pt_40toInf = ' %s/%s.root' % (classifier_samples_dir,sample_info.get('GJet_Pt-40toInf')[0])
    input_file_GJet_Pt_40toInf = TFile.Open(input_name_GJet_Pt_40toInf)
    input_name_DYJetsToLL = '%s/%s.root' % (classifier_samples_dir,sample_info.get('DYJetsToLL_M-50')[0])
    input_file_DYJetsToLL = TFile.Open(input_name_DYJetsToLL)
    input_name_TTGJets = '%s/%s.root' % (classifier_samples_dir,sample_info.get('TTGJets')[0])
    input_file_TTGJets = TFile.Open(input_name_TTGJets)
    input_name_TTGG = '%s/%s.root' % (classifier_samples_dir,sample_info.get('TTGG')[0])
    input_file_TTGG = TFile.Open(input_name_TTGG)
    input_name_TTJets_HT_600to800 = '%s/%s.root' % (classifier_samples_dir,sample_info.get('TTJets_HT-600to800')[0])
    input_file_TTJets_HT_600to800 = TFile.Open(input_name_TTJets_HT_600to800)
    input_name_TTJets_HT_800to1200 = '%s/%s.root' % (classifier_samples_dir,sample_info.get('TTJets_HT-800to1200')[0])
    input_file_TTJets_HT_800to1200 = TFile.Open(input_name_TTJets_HT_800to1200)
    input_name_TTJets_HT_1200to2500 = '%s/%s.root' % (classifier_samples_dir,sample_info.get('TTJets_HT-1200to2500')[0])
    input_file_TTJets_HT_1200to2500 = TFile.Open(input_name_TTJets_HT_1200to2500)
    input_name_TTJets_HT_2500toInf = '%s/%s.root' % (classifier_samples_dir,sample_info.get('TTJets_HT-2500toInf')[0])
    input_file_TTJets_HT_2500toInf = TFile.Open(input_name_TTJets_HT_2500toInf)
    '''input_name_W1JetsToLNu_LHEWpT_0_50 = '%s/%s.root' % (classifier_samples_dir,sample_info.get('W1JetsToLNu_LHEWpT_0-50')[0])
    input_file_W1JetsToLNu_LHEWpT_0_50 = TFile.Open(input_name_W1JetsToLNu_LHEWpT_0_50)'''
    input_name_W1JetsToLNu_LHEWpT_50_150 = '%s/%s.root' % (classifier_samples_dir,sample_info.get('W1JetsToLNu_LHEWpT_50-150')[0])
    input_file_W1JetsToLNu_LHEWpT_50_150 = TFile.Open(input_name_W1JetsToLNu_LHEWpT_50_150)
    input_name_W1JetsToLNu_LHEWpT_150_250 = '%s/%s.root' % (classifier_samples_dir,sample_info.get('W1JetsToLNu_LHEWpT_150-250')[0])
    input_file_W1JetsToLNu_LHEWpT_150_250 = TFile.Open(input_name_W1JetsToLNu_LHEWpT_150_250)
    input_name_W1JetsToLNu_LHEWpT_250_400 = '%s/%s.root' % (classifier_samples_dir,sample_info.get('W1JetsToLNu_LHEWpT_250-400')[0])
    input_file_W1JetsToLNu_LHEWpT_250_400 = TFile.Open(input_name_W1JetsToLNu_LHEWpT_250_400)
    input_name_W1JetsToLNu_LHEWpT_400_inf = '%s/%s.root' % (classifier_samples_dir,sample_info.get('W1JetsToLNu_LHEWpT_400-inf')[0])
    input_file_W1JetsToLNu_LHEWpT_400_inf = TFile.Open(input_name_W1JetsToLNu_LHEWpT_400_inf)
    input_name_W2JetsToLNu_LHEWpT_0_50 = '%s/%s.root' % (classifier_samples_dir,sample_info.get('W2JetsToLNu_LHEWpT_0-50')[0])
    input_file_W2JetsToLNu_LHEWpT_0_50 = TFile.Open(input_name_W2JetsToLNu_LHEWpT_0_50)
    input_name_W2JetsToLNu_LHEWpT_50_150 = '%s/%s.root' % (classifier_samples_dir,sample_info.get('W2JetsToLNu_LHEWpT_50-150')[0])
    input_file_W2JetsToLNu_LHEWpT_50_150 = TFile.Open(input_name_W2JetsToLNu_LHEWpT_50_150)
    input_name_W2JetsToLNu_LHEWpT_150_250 = '%s/%s.root' % (classifier_samples_dir,sample_info.get('W2JetsToLNu_LHEWpT_150-250')[0])
    input_file_W2JetsToLNu_LHEWpT_150_250 = TFile.Open(input_name_W2JetsToLNu_LHEWpT_150_250)
    input_name_W2JetsToLNu_LHEWpT_250_400 = '%s/%s.root' % (classifier_samples_dir,sample_info.get('W2JetsToLNu_LHEWpT_250-400')[0])
    input_file_W2JetsToLNu_LHEWpT_250_400 = TFile.Open(input_name_W2JetsToLNu_LHEWpT_250_400)
    input_name_W2JetsToLNu_LHEWpT_400_inf = '%s/%s.root' % (classifier_samples_dir,sample_info.get('W2JetsToLNu_LHEWpT_400-inf')[0])
    input_file_W2JetsToLNu_LHEWpT_400_inf = TFile.Open(input_name_W2JetsToLNu_LHEWpT_400_inf)
    input_name_W3JetsToLNu = '%s/%s.root' % (classifier_samples_dir,sample_info.get('W3JetsToLNu')[0])
    input_file_W3JetsToLNu = TFile.Open(input_name_W3JetsToLNu)
    input_name_W4JetsToLNu = '%s/%s.root' % (classifier_samples_dir,sample_info.get('W4JetsToLNu')[0])
    input_file_W4JetsToLNu = TFile.Open(input_name_W4JetsToLNu)
    input_name_ttHJetToGG = '%s/%s.root' % (classifier_samples_dir,sample_info.get('ttHJetToGG')[0])
    input_file_ttHJetToGG = TFile.Open(input_name_ttHJetToGG)

    hist_stack = ROOT.THStack()
    histo_HH_name = 'histo_DNN_values_HHWWgg_sample'
    histo_HH_ = input_file_HH.Get(histo_HH_name)
    sample_info.get('HHWWgg').insert(len(sample_info.get('HHWWgg')),histo_HH_)
    histo_diphoton_name = 'histo_DNN_values_DiPhoton_sample'
    histo_diphoton_ = input_file_DiPhoton.Get(histo_diphoton_name)
    sample_info.get('DiPhoton').insert(len(sample_info.get('DiPhoton')),histo_diphoton_)
    histo_GJet_Pt_20toInf_name = 'histo_DNN_values_GJet_Pt-20toInf_sample'
    histo_GJet_Pt_20toInf_ = input_file_GJet_Pt_20toInf.Get(histo_GJet_Pt_20toInf_name)
    sample_info.get('GJet_Pt-20toInf').insert(len(sample_info.get('GJet_Pt-20toInf')),histo_GJet_Pt_20toInf_)
    histo_GJet_Pt_20to40_name = 'histo_DNN_values_GJet_Pt-20to40_sample'
    histo_GJet_Pt_20to40_ = input_file_GJet_Pt_20to40.Get(histo_GJet_Pt_20to40_name)
    sample_info.get('GJet_Pt-20to40').insert(len(sample_info.get('GJet_Pt-20to40')),histo_GJet_Pt_20to40_)
    histo_GJet_Pt_40toInf_name = 'histo_DNN_values_GJet_Pt-40toInf_sample'
    histo_GJet_Pt_40toInf_ = input_file_GJet_Pt_40toInf.Get(histo_GJet_Pt_40toInf_name)
    sample_info.get('GJet_Pt-40toInf').insert(len(sample_info.get('GJet_Pt-40toInf')),histo_GJet_Pt_40toInf_)
    histo_DYJetsToLL_name = 'histo_DNN_values_DYJetsToLL_M-50_sample'
    histo_DYJetsToLL_ = input_file_DYJetsToLL.Get(histo_DYJetsToLL_name)
    sample_info.get('DYJetsToLL_M-50').insert(len(sample_info.get('DYJetsToLL_M-50')),histo_DYJetsToLL_)
    histo_TTGJets_name = 'histo_DNN_values_TTGJets_sample'
    histo_TTGJets_ = input_file_TTGJets.Get(histo_TTGJets_name)
    sample_info.get('TTGJets').insert(len(sample_info.get('TTGJets')),histo_TTGJets_)
    histo_TTGG_name = 'histo_DNN_values_TTGG_sample'
    histo_TTGG_ = input_file_TTGG.Get(histo_TTGG_name)
    sample_info.get('TTGG').insert(len(sample_info.get('TTGG')),histo_TTGG_)
    histo_TTJets_HT_600to800_name = 'histo_DNN_values_TTJets_HT-600to800_sample'
    histo_TTJets_HT_600to800_ = input_file_TTJets_HT_600to800.Get(histo_TTJets_HT_600to800_name)
    sample_info.get('TTJets_HT-600to800').insert(len(sample_info.get('TTJets_HT-600to800')),histo_TTJets_HT_600to800_)
    histo_TTJets_HT_800to1200_name = 'histo_DNN_values_TTJets_HT-800to1200_sample'
    histo_TTJets_HT_800to1200_ = input_file_TTJets_HT_800to1200.Get(histo_TTJets_HT_800to1200_name)
    sample_info.get('TTJets_HT-800to1200').insert(len(sample_info.get('TTJets_HT-800to1200')),histo_TTJets_HT_800to1200_)
    histo_TTJets_HT_1200to2500_name = 'histo_DNN_values_TTJets_HT-1200to2500_sample'
    histo_TTJets_HT_1200to2500_ = input_file_TTJets_HT_1200to2500.Get(histo_TTJets_HT_1200to2500_name)
    sample_info.get('TTJets_HT-1200to2500').insert(len(sample_info.get('TTJets_HT-1200to2500')),histo_TTJets_HT_1200to2500_)
    histo_TTJets_HT_2500toInf_name = 'histo_DNN_values_TTJets_HT-2500toInf_sample'
    histo_TTJets_HT_2500toInf_ = input_file_TTJets_HT_2500toInf.Get(histo_TTJets_HT_2500toInf_name)
    sample_info.get('TTJets_HT-2500toInf').insert(len(sample_info.get('TTJets_HT-2500toInf')),histo_TTJets_HT_2500toInf_)
    '''histo_W1JetsToLNu_LHEWpT_0_50_name = 'histo_DNN_values_W1JetsToLNu_LHEWpT_0-50_sample'
    histo_W1JetsToLNu_LHEWpT_0_50_ = input_file_W1JetsToLNu_LHEWpT_0_50.Get(histo_W1JetsToLNu_LHEWpT_0_50_name)
    sample_info.get('W1JetsToLNu_LHEWpT_0-50').insert(len(sample_info.get('W1JetsToLNu_LHEWpT_0-50')),histo_W1JetsToLNu_LHEWpT_0_50_)'''
    histo_W1JetsToLNu_LHEWpT_50_150_name = 'histo_DNN_values_W1JetsToLNu_LHEWpT_50-150_sample'
    histo_W1JetsToLNu_LHEWpT_50_150_ = input_file_W1JetsToLNu_LHEWpT_50_150.Get(histo_W1JetsToLNu_LHEWpT_50_150_name)
    sample_info.get('W1JetsToLNu_LHEWpT_50-150').insert(len(sample_info.get('W1JetsToLNu_LHEWpT_50-150')),histo_W1JetsToLNu_LHEWpT_50_150_)
    histo_W1JetsToLNu_LHEWpT_150_250_name = 'histo_DNN_values_W1JetsToLNu_LHEWpT_150-250_sample'
    histo_W1JetsToLNu_LHEWpT_150_250_ = input_file_W1JetsToLNu_LHEWpT_150_250.Get(histo_W1JetsToLNu_LHEWpT_150_250_name)
    sample_info.get('W1JetsToLNu_LHEWpT_150-250').insert(len(sample_info.get('W1JetsToLNu_LHEWpT_150-250')),histo_W1JetsToLNu_LHEWpT_150_250_)
    histo_W1JetsToLNu_LHEWpT_250_400_name = 'histo_DNN_values_W1JetsToLNu_LHEWpT_250-400_sample'
    histo_W1JetsToLNu_LHEWpT_250_400_ = input_file_W1JetsToLNu_LHEWpT_250_400.Get(histo_W1JetsToLNu_LHEWpT_250_400_name)
    sample_info.get('W1JetsToLNu_LHEWpT_250-400').insert(len(sample_info.get('W1JetsToLNu_LHEWpT_250-400')),histo_W1JetsToLNu_LHEWpT_250_400_)
    histo_W1JetsToLNu_LHEWpT_400_inf_name = 'histo_DNN_values_W1JetsToLNu_LHEWpT_400-inf_sample'
    histo_W1JetsToLNu_LHEWpT_400_inf_ = input_file_W1JetsToLNu_LHEWpT_400_inf.Get(histo_W1JetsToLNu_LHEWpT_400_inf_name)
    sample_info.get('W1JetsToLNu_LHEWpT_400-inf').insert(len(sample_info.get('W1JetsToLNu_LHEWpT_400-inf')),histo_W1JetsToLNu_LHEWpT_400_inf_)
    histo_W2JetsToLNu_LHEWpT_0_50_name = 'histo_DNN_values_W2JetsToLNu_LHEWpT_0-50_sample'
    histo_W2JetsToLNu_LHEWpT_0_50_ = input_file_W2JetsToLNu_LHEWpT_0_50.Get(histo_W2JetsToLNu_LHEWpT_0_50_name)
    sample_info.get('W2JetsToLNu_LHEWpT_0-50').insert(len(sample_info.get('W2JetsToLNu_LHEWpT_0-50')),histo_W2JetsToLNu_LHEWpT_0_50_)
    histo_W2JetsToLNu_LHEWpT_50_150_name = 'histo_DNN_values_W2JetsToLNu_LHEWpT_50-150_sample'
    histo_W2JetsToLNu_LHEWpT_50_150_ = input_file_W2JetsToLNu_LHEWpT_50_150.Get(histo_W2JetsToLNu_LHEWpT_50_150_name)
    sample_info.get('W2JetsToLNu_LHEWpT_50-150').insert(len(sample_info.get('W2JetsToLNu_LHEWpT_50-150')),histo_W2JetsToLNu_LHEWpT_50_150_)
    histo_W2JetsToLNu_LHEWpT_150_250_name = 'histo_DNN_values_W2JetsToLNu_LHEWpT_150-250_sample'
    histo_W2JetsToLNu_LHEWpT_150_250_ = input_file_W2JetsToLNu_LHEWpT_150_250.Get(histo_W2JetsToLNu_LHEWpT_150_250_name)
    sample_info.get('W2JetsToLNu_LHEWpT_150-250').insert(len(sample_info.get('W2JetsToLNu_LHEWpT_150-250')),histo_W2JetsToLNu_LHEWpT_150_250_)
    histo_W2JetsToLNu_LHEWpT_250_400_name = 'histo_DNN_values_W2JetsToLNu_LHEWpT_250-400_sample'
    histo_W2JetsToLNu_LHEWpT_250_400_ = input_file_W2JetsToLNu_LHEWpT_250_400.Get(histo_W2JetsToLNu_LHEWpT_250_400_name)
    sample_info.get('W2JetsToLNu_LHEWpT_250-400').insert(len(sample_info.get('W2JetsToLNu_LHEWpT_250-400')),histo_W2JetsToLNu_LHEWpT_250_400_)
    histo_W2JetsToLNu_LHEWpT_400_inf_name = 'histo_DNN_values_W2JetsToLNu_LHEWpT_400-inf_sample'
    histo_W2JetsToLNu_LHEWpT_400_inf_ = input_file_W2JetsToLNu_LHEWpT_400_inf.Get(histo_W2JetsToLNu_LHEWpT_400_inf_name)
    sample_info.get('W2JetsToLNu_LHEWpT_400-inf').insert(len(sample_info.get('W2JetsToLNu_LHEWpT_400-inf')),histo_diphoton_)
    histo_W3JetsToLNu_name = 'histo_DNN_values_W3JetsToLNu_sample'
    histo_W3JetsToLNu_ = input_file_W3JetsToLNu.Get(histo_W3JetsToLNu_name)
    sample_info.get('W3JetsToLNu').insert(len(sample_info.get('W3JetsToLNu')),histo_W3JetsToLNu_)
    histo_W4JetsToLNu_name = 'histo_DNN_values_W4JetsToLNu_sample'
    histo_W4JetsToLNu_ = input_file_W4JetsToLNu.Get(histo_W4JetsToLNu_name)
    sample_info.get('W4JetsToLNu').insert(len(sample_info.get('W4JetsToLNu')),histo_W4JetsToLNu_)
    histo_ttHJetToGG_name = 'histo_DNN_values_ttHJetToGG_sample'
    histo_ttHJetToGG_ = input_file_ttHJetToGG.Get(histo_ttHJetToGG_name)
    sample_info.get('ttHJetToGG').insert(len(sample_info.get('ttHJetToGG')),histo_ttHJetToGG_)

    # Rebin Histograms so > 0 total background entries per bin.
    rebinned_histograms, x_bin_edges_ = rebinHistograms(sample_info)
    bckg_hists = ROOT.TH1F('bckg_hists','bckg_hists',len(x_bin_edges_)-1,x_bin_edges_)
    sig_hists = ROOT.TH1F('sig_hists','sig_hists',len(x_bin_edges_)-1,x_bin_edges_)
    legend = TLegend(0.7,  0.7,  0.9,  0.9)
    legend.SetNColumns(2)

    signal_string = 'HHWWgg'

    plotted_samples = []
    for rebinned_hist_name, info in rebinned_histograms.items():
        plotted_samples.insert(len(plotted_samples),rebinned_hist_name)
        previously_plotted = 0
        legend_entry = ''
        for isample in range(len(plotted_samples)):
            if signal_scale != 1. and signal_string in rebinned_hist_name:
                print('signal_scale=', signal_scale)
                legend_entry="signalx"+str(signal_scale)
            elif 'TTJets' in plotted_samples[isample] and 'TTJets' in rebinned_hist_name:
                previously_plotted+=1
                legend_entry = 'TTJets'
            elif 'JetsToLNu' in plotted_samples[isample] and 'JetsToLNu' in rebinned_hist_name:
                previously_plotted+=1
                legend_entry = 'WJetsToLNu'
            elif 'GJet_Pt-' in plotted_samples[isample] and 'GJet_Pt-' in rebinned_hist_name:
                previously_plotted+=1
                legend_entry = 'GJets'
            elif 'DYJets' in plotted_samples[isample] and 'DYJets' in rebinned_hist_name:
                previously_plotted+=1
                legend_entry = 'DYJetsToLL'
            else:
                legend_entry = rebinned_hist_name

        rebinned_hist = info[2]
        rebinned_hist.SetMarkerColor(sample_info.get(rebinned_hist_name)[1])
        rebinned_hist.SetLineColor(sample_info.get(rebinned_hist_name)[1])
        rebinned_hist.GetYaxis().SetTitle("Events/bin")
        rebinned_hist.SetMarkerStyle(20)
        if 'Data' in rebinned_hist_name:
            legend.AddEntry(rebinned_hist,legend_entry,'p')
            continue
        else:
            if signal_string in rebinned_hist_name:
                legend.AddEntry(rebinned_hist,legend_entry,'l')
                rebinned_hist.SetFillStyle(0)
            if signal_string not in rebinned_hist_name:
                rebinned_hist.Scale(41.5)
                if previously_plotted<=1:
                    legend.AddEntry(rebinned_hist,legend_entry,'f')
                rebinned_hist.SetMarkerStyle(20)
                rebinned_hist.SetFillStyle(3001)
                rebinned_hist.SetFillColor(sample_info.get(rebinned_hist_name)[1])
                hist_stack.Add(rebinned_hist)
            if signal_string not in rebinned_hist_name:
                bckg_hists.Add(rebinned_hist)
            if signal_string in rebinned_hist_name:
                sig_hists.Add(rebinned_hist)

    hist_stack.SetMaximum(hist_stack.GetStack().Last().GetMaximum() + (hist_stack.GetStack().Last().GetMaximum()/2))
    hist_stack.SetMinimum(0.0001)

    separation_ =  GetSeparation(sig_hists,bckg_hists)

    # Get Data/MC agreement
    if data_flag:
        dataOverMC = GetDataOverMC(hist_stack,rebinned_histograms['Data'])
        make_plot(hist_stack, sig_hists, False, legend, inputs_directory, separation_, option_, rebinned_histograms['Data'] , dataOverMC)
    else:
        sig_mc = sig_hists.Clone('sigframe')
        bckg_mc = hist_stack.Clone('bckgframe')
        nBins = sig_mc.GetNbinsX()
        SOverRootB_hist = ROOT.TH1F('SOverRootB_hist','SOverRootB_hist',nBins,0,1)
        SOverRootB_ratio = GetSOverRootB(sig_mc, bckg_mc)
        make_plot(hist_stack, sig_hists, False, legend, inputs_directory, separation_, SOverRootB_ratio, signal_scale)

    print('<DNN_Evaluation_Control_Plots> Cleanup')
    bckg_hists.Reset()
    sig_hists.Reset()
    sample_info.clear()
    exit(0)

main()
