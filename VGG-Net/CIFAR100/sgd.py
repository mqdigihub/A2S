

def SGD_Optimizer(lr, parameters, grads):

    l = len(parameters)//8

    for i in range(l+1):

        step_w = -lr*grads["dw" + str(i + 1)]
        step_b = -lr*grads["db" + str(i + 1)]

        parameters["w" + str(i + 1)] = parameters["w" + str(i + 1)] + step_w
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] + step_b

        if i != l:
            step_gama = -lr * grads["dgama" + str(i + 1)]
            step_beta = -lr * grads["dbeta" + str(i + 1)]

            step_se_w1 = -lr*grads["se"+str(i + 1)+"_dw1"]
            step_se_w2 = -lr*grads["se"+str(i + 1)+"_dw2"]

            step_se_b1 = -lr*grads["se"+str(i + 1)+"_db1"]
            step_se_b2 = -lr*grads["se"+str(i + 1)+"_db2"]

            parameters["gama" + str(i + 1)] = parameters["gama" + str(i + 1)] + step_gama
            parameters["beta" + str(i + 1)] = parameters["beta" + str(i + 1)] + step_beta
            parameters["se" + str(i + 1) + "_w1"] = parameters["se" + str(i + 1) + "_w1"] + step_se_w1
            parameters["se" + str(i + 1) + "_w2"] = parameters["se" + str(i + 1) + "_w2"] + step_se_w2
            parameters["se" + str(i + 1) + "_b1"] = parameters["se" + str(i + 1) + "_b1"] + step_se_b1
            parameters["se" + str(i + 1) + "_b2"] = parameters["se" + str(i + 1) + "_b2"] + step_se_b2

    return parameters
