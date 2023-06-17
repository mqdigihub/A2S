

def SGD_Optimizer(lr, parameters, grads):

    l = len(parameters)//4

    for i in range(l+1):

        step_w = -lr*grads["dw" + str(i + 1)]
        step_b = -lr*grads["db" + str(i + 1)]

        parameters["w" + str(i + 1)] = parameters["w" + str(i + 1)] + step_w
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] + step_b

        if i != l:
            step_gama = -lr * grads["dgama" + str(i + 1)]
            step_beta = -lr * grads["dbeta" + str(i + 1)]

            parameters["gama" + str(i + 1)] = parameters["gama" + str(i + 1)] + step_gama
            parameters["beta" + str(i + 1)] = parameters["beta" + str(i + 1)] + step_beta

    return parameters
