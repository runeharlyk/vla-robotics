import pandas as pd, numpy as np
df1=pd.read_csv('visual_diagnostic/results/visual_cor_suc_s1/visual_noise_rollouts_raw.csv')
df3=pd.read_csv('visual_diagnostic/results/visual_cor_suc_s3/visual_noise_rollouts_raw.csv')
df5=pd.read_csv('visual_diagnostic/results/visual_cor_suc_s5/visual_noise_rollouts_raw.csv')
df=pd.concat([df1,df3,df5],ignore_index=True)
print('Total rows:', len(df))
print('Severities:', sorted(df['severity'].unique()))
print('Noise types:', sorted(df[df['noise_type']!='clean']['noise_type'].unique()))
print('Suites:', sorted(df['suite'].unique()))
print()
clean=df[df['noise_type']=='clean']
print('Clean baseline per suite:')
print(clean.groupby('suite')['success'].mean().round(4))
print()
print('Overall clean baseline:', clean['success'].mean().round(4))
print()
print('Success rate by severity+noise (all suites):')
print(df[df['noise_type']!='clean'].groupby(['severity','noise_type'])['success'].agg(['count','mean']).round(4))
print()
print('Success rate by suite + noise (averaged over severities):')
noised = df[df['noise_type']!='clean']
print(noised.groupby(['suite','noise_type'])['success'].mean().round(4).unstack())
print()
print('Success rate by suite + severity:')
print(noised.groupby(['suite','severity'])['success'].mean().round(4).unstack())
print()
print('Clean success per task:')
print(clean.groupby(['suite','task_id','task_description'])['success'].agg(['count','mean']).round(4).to_string())
print()
# For heatmap: task drop at s3
s3 = df[df['severity']==3]
clean_task = clean.groupby(['task_id'])['success'].mean().rename('clean_sr')
noised_s3 = s3[s3['noise_type']!='clean']
task_noise = noised_s3.groupby(['task_id','task_description','noise_type'])['success'].mean().reset_index()
task_noise = task_noise.merge(clean_task, on='task_id')
task_noise['drop'] = task_noise['clean_sr'] - task_noise['success']
print('Task drop at s3 (top 5 biggest drops):')
print(task_noise.nlargest(10, 'drop')[['task_description','noise_type','clean_sr','success','drop']].to_string(index=False))
print()
print('Task drop at s3 (negative drops = improvements):')
print(task_noise.nsmallest(10, 'drop')[['task_description','noise_type','clean_sr','success','drop']].to_string(index=False))
print()
# Combined delta bars (averaged over sev): drop per suite per noise
clean_suite = clean.groupby('suite')['success'].mean().rename('clean_sr')
suite_noise = noised.groupby(['suite','noise_type'])['success'].mean().reset_index()
suite_noise = suite_noise.merge(clean_suite, on='suite')
suite_noise['drop'] = suite_noise['clean_sr'] - suite_noise['success']
print('Combined delta (avg over severities) per suite per noise:')
print(suite_noise.pivot(index='suite', columns='noise_type', values='drop').round(4))
