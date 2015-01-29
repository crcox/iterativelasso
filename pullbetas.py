#!/usr/bin/env python

import json
import os
import readbin

with open('summary.json','rb') as f:
    jdat = json.load(f)

for iter_ in range(jdat['niter']):
    iterdir = 'iter{i:02d}'.format(i=iter_)

    for cv in range(jdat['ncv']):
        beta = []

        cvdir = 'cv{i:02d}'.format(i=cv+1)
        beta_path = os.path.join('iterations',iterdir,cvdir,'beta.bin')

        with open(beta_path,'rb') as f:
            beta.append(readbin.mat(f))

    betacsv = 'beta_iter{i:02d}.csv'.format(i=iter_)
    with open(betacsv,'wb') as f:
        for i in range(len(beta[0])):
            x = [str(b[i]) for b in beta]
            f.write(','.join(x)+'\n')

iterdir = 'final'
for cv in range(jdat['ncv']):
    cvdir = 'cv{i:02d}'.format(i=cv+1)
    beta_path = os.path.join(iterdir,cvdir,'beta.bin')

    with open(beta_path,'rb') as f:
        beta.append(readbin.mat(f))

betacsv = 'beta_final.csv'.format(i=iter_)
with open(betacsv,'wb') as f:
    for i in range(len(beta[0])):
        x = [str(b[i]) for b in beta]
        f.write(','.join(x)+'\n')
