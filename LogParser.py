#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import os
import json

Whole_data = []
# PT plaint text
# CT cipher text
for filename in os.listdir("EE_Log"):
    with open(os.path.join("EE_Log", filename), "r") as f:
        file = csv.reader(f, delimiter=",")
        MU_mode = 0
        PT_test_duration = 0
        CT_test_duration = 0
        PT_MU = 0
        CT_MU = 0
        MU_Encrypting_testset = 0
        next(file)
        for row in file:
            Loss = row[2]
            if int(row[3]) == 1:
                pt_acc = float(row[4])
                ct_acc = float(row[5])
                one_acc_diff = pt_acc - ct_acc
            if int(row[3]) == -1:
                m1_pt_acc = float(row[4])
                m1_ct_acc = float(row[5])
                m1_acc_diff = m1_pt_acc - m1_ct_acc
            MU_mode += float(row[8])
            PT_test_duration += float(row[6])
            CT_test_duration += float(row[7])
            PT_MU += float(row[9])
            CT_MU += float(row[10])
            MU_Encrypting_testset += float(row[11])
    data = {
        "Filename": filename,
        "Loss": Loss,
        "Difference in testing accuracy of degree 1": one_acc_diff,
        "Difference in testing accuracy of degree -1": m1_acc_diff,
        "Average memory usage of the model": MU_mode / 10,
        "Average Plaintext testing duration": PT_test_duration / 10,
        "Average Ciphertext testing duration": CT_test_duration / 10,
        "Average plaintext testing memory usage": PT_MU / 10,
        "Average encrypted testing memory usage": CT_MU / 10,
        "Average memory usage of encrypting the testset": MU_Encrypting_testset / 10,
        "Percentage Change in accuracy of degree -1": (
            (m1_ct_acc - m1_pt_acc) / m1_pt_acc
        )
        * 100,
        "Percentage Change in accuracy of degree 1": ((ct_acc - pt_acc) / pt_acc) * 100,
        "Percentage Change in duration": (
            (CT_test_duration - PT_test_duration) / PT_test_duration
        )
        * 100,
        "Percentage Change in memory usage": ((CT_MU - PT_MU) / PT_MU) * 100,
    }
    Whole_data.append(data)


r_Acc_D_1 = 0
r_Acc_D_m1 = 0
r_Increase_D = 0
r_Increase_MU = 0
r_mem_of_mode = 0
for data in Whole_data:
    r_Acc_D_m1 += data["Percentage Change in accuracy of degree -1"]
    r_Acc_D_1 += data["Percentage Change in accuracy of degree 1"]
    r_Increase_D += data["Percentage Change in duration"]
    r_Increase_MU += data["Percentage Change in memory usage"]
    r_mem_of_mode += data["Average memory usage of the model"]

print(
    r_Acc_D_m1 / len(Whole_data),
    r_Acc_D_1 / len(Whole_data),
    r_Increase_D / len(Whole_data),
    r_Increase_MU / len(Whole_data),
    r_mem_of_mode / len(Whole_data),
)


with open("findings.txt", "w") as f:
    for data in Whole_data:
        f.write(json.dumps(data))
        f.write("\n")
