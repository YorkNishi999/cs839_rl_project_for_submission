# Git clone and move current directry
```
git clone git@github.com:YorkNishi999/cs839_rl_project_for_submission.git
cd /path/to/repo/gym-adserver
```

# Build environment

```
conda env create -f gym-adserver.yml
```

# Comparison analysis
1. Check below in all agents py files.
```
# outdir = './outputs/sensitivity/{agent}.txt' # for sensitivity test
outdir = './outputs/{agent}.txt' # for comparison test
```
2. Then do the command.
```
python comparison_test.py
```
3. You can see output images in `./outputs/imgs`

# sensitivity analysis
## ucb1
1. Check below.
```gym_adserver/agents/ucb1_agent.py
outdir = './outputs/sensitivity/ucb1.txt' # for sensitivity test
# outdir = './outputs/ucb1.txt' # for comparison test
```
2. Then do the command.
```
python sensitivity_ucb.py
```
3. You can see output images in `./outputs/sensitivity/`


## Gradient Bandit
1. Check below.
```gym_adserver/agents/softmax_agent.py
outdir = './outputs/sensitivity/softmax.txt' # sensitivity test
# outdir = './outputs/softmax.txt' # comparison.py
```
2. Then do the command.
```
python sensitivity_softmax.py
```
3. You can see output images in `./outputs/sensitivity/`

## egreedy
1. Check below.
```gym_adserver/agents/epsilon_greedy_agent.py
outdir = './outputs/sensitivity/egreedy.txt' # sensitivity 
# outdir = './outputs/egreedy.txt' # conparison
```
2. Then do the command.
```
python sensitivity_egreedy.py
```
3. You can see output images in `./outputs/sensitivity/`

