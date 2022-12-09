import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


x_egreedy = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
egreedy = [0.399225, 0.41654749999999996, 0.4239016666666666, 0.43100625, 0.434495, 0.4357375, 0.43575142857142857, 0.43761312500000005, 0.43859777777777775, 0.44401, 0.43287, 0.42038500000000006, 0.407955, 0.39551400000000003, 0.38293999999999995, 0.3702307142857143, 0.35741875, 0.3445322222222223]
softmax = [0.28415, 0.2982775, 0.3078699999999999, 0.31484625, 0.32116800000000006, 0.32515, 0.3288028571428572, 0.331430625, 0.33412111111111115, 0.35432, 0.35058249999999996, 0.34392542372881363, 0.3375479452054795, 0.3322426829268293, 0.3300094117647059, 0.3300094117647059, 0.3300094117647059, 0.3300094117647059]

assert len(x_egreedy) == len(egreedy)

print("epsilon: ", x_egreedy[np.argmax(egreedy)])
print("Gradient Bandit: ", x_egreedy[np.argmax(softmax)])

# plot
fig, ax1 = plt.subplots()
# plot and create legend index
lns1 = ax1.plot(x_egreedy,egreedy, color='blue', label='Epsilon Greedy')
lns2 = ax1.plot(x_egreedy,softmax, color='purple', label='Gradient Bandit')
lns = lns1+lns2
# lns = lns1
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)

# set axis label
ax1.set_xlabel('epsilon/step size')
ax1.set_ylabel('Average rewards after 10000th episode')

# save figure
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95)
fig.savefig('./outputs/sensitivity/sensitive_study_egreedy_gradientbandit.png')


# ucb
x_ucb = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,2,3,4,5,6,7,8,9]
ucb_sensitivity = [
        0.4765999999999999, 0.4766, 0.4766000000000001, 0.4766, 0.47659999999999997, 0.47660000000000013, 0.4765999999999999, 0.4766, 0.47659999999999997, 
        0.4765999999999999, 0.47635000000000005, 0.4752666666666666, 0.47467500000000007, 0.47320000000000007, 0.4717833333333333, 0.4705142857142857, 0.4694625, 0.4679111111111111,
        0.44909999999999994, 0.42779999999999996, 0.40353333333333335, 0.382725, 0.36695999999999984, 0.3542833333333332, 0.3432714285714286, 0.334275, 0.3268
        ]
assert len(x_ucb) == len(ucb_sensitivity)

print("UCB: ", x_ucb[np.argmax(ucb_sensitivity)])


# plot
fig, ax1 = plt.subplots()
# plot and create legend index
lns1 = ax1.plot(x_ucb, ucb_sensitivity, color='green', label='UCB')
lns = lns1
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)

# set axis label
ax1.set_xlabel('c')
ax1.set_ylabel('Average rewards after 10000th episode')

# save figure
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95)
fig.savefig('./outputs/sensitivity/sensitive_study_ucb.png')