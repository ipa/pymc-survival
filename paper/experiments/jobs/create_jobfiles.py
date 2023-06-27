import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--user')
parser.add_argument('--email')
parser.add_argument('--container')
parser.add_argument('--folder')
args = parser.parse_args()

os.makedirs(args.folder, exist_ok=True)

models = {
            'cox': {'short': 'cox', 'nodes': 10, 'walltime': 6}, 
            'deepsurv': {'short': 'deepsurv', 'nodes': 10, 'walltime': 24}, 
            'pmsurv_exponential': {'short': 'exp', 'nodes': 10, 'walltime': 24}, 
            'pmsurv_weibull_nn': {'short': 'nnwb', 'nodes': 10, 'walltime': 24}, 
            'rsf': {'short': 'rsf', 'nodes': 10, 'walltime': 12}, 
            'pmsurv_weibull_linear': {'short': 'wb', 'nodes': 10, 'walltime': 24}
          }
datasets = ['aids', 'gbcs', 'pbc', 'veteran', 'whas']


with open('job-template.lsf', 'r') as file:
    template = file.read() 


for model in models:
    jobfile = template
    jobfile = jobfile.replace('[container]', args.container)
    jobfile = jobfile.replace('[model]', model)
    jobfile = jobfile.replace('[user]', args.user)
    jobfile = jobfile.replace('[email]', args.email)
    jobfile = jobfile.replace('[model_short]', models[model]['short'])
    jobfile = jobfile.replace('[walltime]', str(models[model]['walltime']))

    for ds in datasets:
        jobfile_ds = jobfile.replace('[dataset]', ds)

        with open(os.path.join(args.folder, f'{models[model]["short"]}_{ds}.lsf'), 'w') as file:
            file.write(jobfile_ds)

print('done...')
