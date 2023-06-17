import matplotlib.pyplot as plt
import pickle as pk

def draw(path1, path2):
    with open(path1, 'rb') as f:

        test_accs = pk.load(f)


    with open(path2, 'rb') as f:

        cost = pk.load(f)

    return test_accs, cost

# no fixed epoch
# test_accs_ASP, cost_ASP = draw(path1='./results/EXP1/ASP/test_accs', path2='./results/EXP1/ASP/loss')
# test_accs_BSP, cost_BSP = draw(path1='./results/EXP1/BSP/test_accs', path2='./results/EXP1/BSP/loss')
# test_accs_SSP, cost_SSP = draw(path1='./results/EXP1/SSP/test_accs', path2='./results/EXP1/SSP/loss')
# test_accs_DSSP, cost_DSSP = draw(path1='./results/EXP1/DSSP/test_accs', path2='./results/EXP1/DSSP/loss')
# # test_accs_DC_ASGD_A, cost_DC_ASGD_A = draw(path1='./lr_0.085(0.999decay)/DC_ASGD_A_RESULTS/test_accs', path2='./lr_0.085(0.999decay)/DC_ASGD_A_RESULTS/loss')
# test_accs_DC_ASGD_C, cost_DC_ASGD_C = draw(path1='./results/EXP1/DC-ASGD-c/test_accs', path2='./results/EXP1/DC-ASGD-c/loss')
# test_accs_A2S, cost_A2S = draw(path1='./results/EXP1/A2S/test_accs', path2='./results/EXP1/A2S/loss')


# fixed epoch
test_accs_sync_switch_p1, cost_sync_switch_p1 = draw(path1='./results/EXP1/fixed_epoch/Sync-switch/p1/test_accs',
                                                     path2='./results/EXP1/fixed_epoch/Sync-switch/p1/loss')
test_accs_sync_switch_p2, cost_sync_switch_p2 = draw(path1='./results/EXP1/fixed_epoch/Sync-switch/p2/test_accs',
                                                     path2='./results/EXP1/fixed_epoch/Sync-switch/p2/loss')
test_accs_sync_switch_p3, cost_sync_switch_p3 = draw(path1='./results/EXP1/fixed_epoch/Sync-switch/p3/test_accs',
                                                     path2='./results/EXP1/fixed_epoch/Sync-switch/p3/loss')
test_accs_A2S, cost_A2S = draw(path1='./results/EXP1/fixed_epoch/A2S/test_accs', path2='./results/EXP1/fixed_epoch/A2S/loss')


prev_acc = 0
move_avg_acc = 0
biased_acc = 0
avg_acc = 0
print(len(test_accs_sync_switch_p2))
print(test_accs_sync_switch_p2)
for epoch in range(1, 201):

    avg_acc = test_accs_sync_switch_p2[epoch-1]

    delta_acc = abs(avg_acc - prev_acc)
    move_avg_acc = 0.9 * move_avg_acc + 0.1 * delta_acc
    biased_acc = move_avg_acc / (1 - 0.9 ** epoch)

    prev_acc = avg_acc
print(biased_acc)


ax=plt.gca()  #gca:get current axis得到当前轴
#设置图片的右边框和上边框为不显示
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('lightgray')
ax.spines['left'].set_color('lightgrey')

plt.figure(1)
# plt.plot(test_accs_ASP, color='blue', label='ASP')
# plt.plot(test_accs_BSP, color='green', label='BSP')
# plt.plot(test_accs_SSP, color='black', label='SSP')
# plt.plot(test_accs_DSSP, color='pink', label='DSSP')
# # plt.plot(test_accs_DC_ASGD_A, color='black', label='DC_ASGD_a')
# plt.plot(test_accs_DC_ASGD_C, color='skyblue', label='DC_ASGD_c')

# plt.plot(test_accs_sync_switch_p1, color='blue', label='Sync-switch-p1')
# plt.plot(test_accs_sync_switch_p2, color='green', label='Sync-switch-p2')
# plt.plot(test_accs_sync_switch_p3, color='black', label='Sync-switch-p3')
# plt.plot(test_accs_A2S, color='red', label='A2S', ls='--', lw=2.5)
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.grid(axis='y', color='lightgrey')
# plt.legend()
#
#
# ax=plt.gca()  #gca:get current axis得到当前轴
# #设置图片的右边框和上边框为不显示
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.spines['bottom'].set_color('lightgray')
# ax.spines['left'].set_color('lightgrey')
plt.figure(2)
# plt.plot(cost_ASP, color='blue', label='ASP')
# plt.plot(cost_BSP, color='green', label='BSP')
# plt.plot(cost_SSP, color='black', label='SSP')
# plt.plot(cost_DSSP, color='pink', label='DSSP')
# # plt.plot(cost_DC_ASGD_A, color='black', label='DC_ASGD_a')
# plt.plot(cost_DC_ASGD_C, color='skyblue', label='DC_ASGD_c')
plt.plot(cost_sync_switch_p1, color='blue', label='Sync-switch-p1')
plt.plot(cost_sync_switch_p2, color='green', label='Sync-switch-p2')
plt.plot(cost_sync_switch_p3, color='black', label='Sync-switch-p3')
plt.plot(cost_A2S, color='red', label='A2S', ls='--', lw=2.5)
plt.ylabel('LOSS')
plt.legend()
plt.xlabel('Epoch')
plt.grid(axis='y')

plt.figure(3)
# speedup = [2.84, 3.20, 4.09, 4.44]
speedup = [1, 2.46, 2.92, 3.25, 3.64, 3.68]
plt.plot([1, 4, 6, 8, 10, 12], speedup, color='r', marker='s', ls='--', label='A2S')
plt.plot([1, 4, 6, 8, 10, 12], [1, 4, 6, 8, 10, 12], color='green', ls='-', label='linear scaling')
plt.xlabel('Number of workers')
plt.ylabel('Speedup')
plt.legend()
plt.show()

