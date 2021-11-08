import os

# Balance     = ["BalanceYields"] #["BalanceYields", "BalanceNonWeighted"]
Balance     = ["BalanceYields", "BalanceNonWeighted"]
optimizer   = ['Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# init_mode   = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
activation  = ['softmax', 'softplus', 'softsign', 'relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
learn_rate  = [0.0001] # [0.00001, 0.0001, 0.001, 0.01]
epochs      = [400] #[100, 120, 140, 160, 180, 200, 250, 300, 400]
batch_size  = [150] # [60, 100, 150, 200, 250]
model       = ["SL","FH_ANv5","FH_ANv5_NoBN","FH_ANv5_NoBN_NoDO","SimpleV2"]
# model       = ["SL","FH_ANv5","FH_ANv5_NoBN","FH_ANv5_NoBN_NoDO","SimpleV1","SimpleV2"]
# model       = ["SimpleV2"] # ["SL","FH_ANv5","FH_ANv5_NoBN","FH_ANv5_NoBN_NoDO","SimpleV1","SimpleV2"]


# command = 'python slurm_SetupScript_BBvsAllBkg.py -j {jobName} -isTrain 1 -d Model{model}_E{epoch}_LR{learn_rateString}_B{batch_size}_{activation}_{optimizer}_SW_ManyVarLowHigLevelv2_Trial1         -e {epoch}  -lr {learn_rate} -b {batch_size} -o {optimizer} -sw True -w {BalanceYield} -a "{activation}" -dropout_rate 0.1 -json input_variables_LowHighLevelBoth_v2.json -ModelToUse "{model}"'
# command = 'python slurm_SetupScript_BBvsAllBkg.py -j {jobName} -isTrain 1 -d Model{modelString}_E{epoch}_LR{learn_rateString}_B{batch_size}_{activation}_{optimizer}_CW_ManyVarLowHigLevelv2_Trial1         -e {epoch}  -lr {learn_rate} -b {batch_size} -o {optimizer} -cw True -w {BalanceYield} -a "{activation}" -dropout_rate 0.1 -json input_variables_LowHighLevelBoth_v2.json -ModelToUse "{model}"'
command = 'python slurm_SetupScript_WWvsBB.py -j {jobName} -isTrain 1 -d 11Aug_Model{modelString}_E{epoch}_LR{learn_rateString}_B{batch_size}_{activation}_{optimizer}_CW_ManyVarLowHigLevelv2        -e {epoch}  -lr {learn_rate} -b {batch_size} -o {optimizer} -cw True -w {BalanceYield} -a "{activation}" -dropout_rate 0.1 -json input_variables_LowHighLevelBoth_v2.json -ModelToUse "{model}"'




filename = 'Command_Summary_August2021.sh'

if os.path.exists(filename):
    append_write = 'a' # append if already exists
else:
    append_write = 'w' # make a new file if not

fileToWrite = open(filename,append_write)

import datetime
now = datetime.datetime.now()
fileToWrite.write("\n#"+"="*51+"\n")
fileToWrite.write("#Current date and time : ")
fileToWrite.write(now.strftime("%Y-%m-%d %H:%M:%S")+"\n")
fileToWrite.write("#"+"="*51+"\n")


count = 0
for Balance_ in Balance:
    for optimizer_ in optimizer:
        for activation_ in activation:
            for learn_rate_ in learn_rate:
                for epochs_ in epochs:
                    for batch_size_ in batch_size:
                        for model_ in model:
                            count = count + 1
                            if count < 321: continue
                            # if Balance_ == "BalanceNonWeighted": continue
                            # if activation_ == "softmax": continue
                            # commandToRun = command.format(activation=activation_,optimizer=optimizer_,BalanceYield=Balance_)
                            commandToRun = command.format(
                                                          jobName="WWBB",
                                                          modelString=model_.replace("_",""),
                                                          model=model_,
                                                          epoch=epochs_,
                                                          learn_rateString=str("{:.0e}".format(learn_rate_)).replace("-","m"),
                                                          learn_rate=learn_rate_,
                                                          batch_size=batch_size_,
                                                          activation=activation_,
                                                          optimizer=optimizer_,
                                                          BalanceYield=Balance_)
                            print("=="*51)
                            print("==\tJob no: {}\n".format(count))
                            print(commandToRun)
                            fileToWrite.write("\n#==\tJob no: {}\n".format(count))
                            fileToWrite.write(commandToRun)
                            os.system(commandToRun)

fileToWrite.close()

print("=="*51)
print("Total number of submitted jobs are {}.".format(count))
