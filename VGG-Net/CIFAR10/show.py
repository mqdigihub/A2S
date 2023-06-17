import matplotlib.pyplot as plt
import pickle as pk

def draw(path1, path2):
    with open(path1, 'rb') as f:

        test_accs = pk.load(f)


    with open(path2, 'rb') as f:

        cost = pk.load(f)

    return test_accs, cost

relax_factor, ps_speed = draw(path1='./results/EXP1/A', path2='./results/EXP1/ps_speed')
test_accs_SW, cost_SW = draw(path1='./results/EXP1/SW/exp2/test_accs', path2='./results/EXP1/SW/exp2/loss')
test_accs_ASP, cost_ASP = draw(path1='./results/EXP1/ASP/test_accs', path2='./results/EXP1/ASP/loss')
test_accs_BSP, cost_BSP = draw(path1='./results/EXP1/BSP/exp2/test_accs', path2='./results/EXP1/BSP/exp2/loss')
test_accs_DSSP, cost_DSSP = draw(path1='./results/EXP1/DSSP/test_accs', path2='./results/EXP1/DSSP/loss')
test_accs_SSP, cost_SSP = draw(path1='./results/EXP1/SSP/test_accs', path2='./results/EXP1/SSP/loss')
#test_accs_DC_ASGD_A, cost_DC_ASGD_A = draw(path1='./lr_0.085(0.999decay)/DC_ASGD_A_RESULTS/test_accs', path2='./lr_0.085(0.999decay)/DC_ASGD_A_RESULTS/loss')
#test_accs_DC_ASGD_C, cost_DC_ASGD_C = draw(path1='./lr_0.085(0.999decay)/DC_ASGD_C_RESULTS/test_accs', path2='./lr_0.085(0.999decay)/DC_ASGD_C_RESULTS/loss')
test_accs_A2S, cost_A2S = draw(path1='./results/EXP1/A2S/test_accs', path2='./results/EXP1/A2S/loss')

ax=plt.gca()  #gca:get current axis得到当前轴
#设置图片的右边框和上边框为不显示
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('lightgray')
ax.spines['left'].set_color('lightgrey')


# prev_acc = 0
# move_avg_acc = 0
# biased_acc = 0
# avg_acc = 0
# for epoch in range(1, 297):
#
#     avg_acc = test_accs_SW[epoch - 1]
#     delta_acc = abs(avg_acc - prev_acc)
#     move_avg_acc = 0.9 * move_avg_acc + 0.1 * delta_acc
#     biased_acc = move_avg_acc / (1 - 0.9 ** epoch)
#
#     prev_acc = avg_acc
#     avg_acc = 0
#
# print(biased_acc)

print(relax_factor)
print(max(ps_speed))
plt.plot(ps_speed, color='purple', label='PS-Speed', ls='-')
plt.ylabel('Time')
plt.xlabel('Iterations')
plt.show()
# plt.plot(test_accs_SW, color='purple', label='SW-P2', ls='-')
# plt.plot(test_accs_ASP, color='blue', label='ASP', ls=':')
# plt.plot(test_accs_BSP, color='green', label='BSP', ls='-.')
# plt.plot(test_accs_SSP, color='gold', label='SSP', ls='dotted')
# plt.plot(test_accs_DSSP, color='skyblue', label='DSSP', ls='-')
# plt.plot(test_accs_DC_ASGD_A, color='black', label='DC_ASGD_a')
# plt.plot(test_accs_DC_ASGD_C, color='skyblue', label='DC_ASGD_c')
# plt.plot(test_accs_A2S, color='red', label='A2S', ls='-')
# plt.axhline(0.90, color='black', label='90% Accuracy', ls='dotted')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
plt.grid(axis='y', color='lightgrey')
plt.legend()

# plt.figure(2)
# plt.plot(cost_SW, color='pink', label='SW-12.5%')
# plt.plot(cost_ASP, color='blue', label='ASP')
# plt.plot(cost_BSP, color='green', label='BSP')
# plt.plot(cost_SSP, color='black', label='SSP')
# plt.plot(cost_DSSP, color='skyblue', label='DSSP')
# # plt.plot(cost_DC_ASGD_A, color='black', label='DC_ASGD_a')
# # plt.plot(cost_DC_ASGD_C, color='pink', label='DC_ASGD_c')
# plt.plot(cost_A2S, color='red', label='A2S')
# plt.ylabel('LOSS')
# plt.legend()
# plt.xlabel('Epoch')
# plt.grid(axis='y')



