

def SGD_Optimizer(lr, parameters, grads):

    l = len(parameters)//2

    for i in range(l):

        step_w = -lr*grads["dw" + str(i + 1)]
        step_b = -lr*grads["db" + str(i + 1)]

        parameters["w" + str(i + 1)] = parameters["w" + str(i + 1)] + step_w
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] + step_b

    return parameters
