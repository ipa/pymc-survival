import os
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--user')
parser.add_argument('--email')
parser.add_argument('--container')
parser.add_argument('--folder')
parser.add_argument('--config-file')
args = parser.parse_args()

os.makedirs(args.folder, exist_ok=True)

with open(args.config_file, 'r') as file:
    config = yaml.full_load(file)
    models = config['models']
    datasets = config['datasets']
    template_file = config['template']


with open(template_file, 'r') as file:
    template = file.read() 


for model_name, model in models.items():
    print(model)
    jobfile = template
    jobfile = jobfile.replace('[container]', args.container)
    jobfile = jobfile.replace('[model]', model['name'])
    jobfile = jobfile.replace('[nodes]', str(model['nodes']+1))
    jobfile = jobfile.replace('[n_jobs]', str(model['nodes']))
    jobfile = jobfile.replace('[memory]', str(model['memory']))
    jobfile = jobfile.replace('[user]', args.user)
    jobfile = jobfile.replace('[email]', args.email)
    jobfile = jobfile.replace('[model_short]', model['short_name'])
    jobfile = jobfile.replace('[walltime]', str(model['walltime']))
    queue = 'short' if model['walltime'] <= 3 else 'medium'
    jobfile = jobfile.replace('[queue]', queue)

    for ds in datasets:
        jobfile_ds = jobfile.replace('[dataset]', ds)

        if ds == 'aids':
            jobfile_ds = jobfile_ds.replace('[n_partitions]', '10')
        else:
            jobfile_ds = jobfile_ds.replace('[n_partitions]', '6')

        with open(os.path.join(args.folder, f'{model["short_name"]}_{ds}.lsf'), 'w') as file:
            file.write(jobfile_ds)

print('done...')
