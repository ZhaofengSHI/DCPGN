
import pandas as pd
import math
import os
import csv


# noun
df_noun = pd.read_csv('/data1/zhaofeng/TTA/EEL-benchmark-features-szf/EEL-annotations/narration_noun_taxonomy.csv')
# value = df_noun.iloc[0, df_noun.columns.get_loc('label')]
# print(value)

with open('/data1/zhaofeng/TTA/EEL-benchmark-features-szf/EEL-annotations/valid_nouns.txt','r') as f1:
    noun_txt = f1.readlines()

with open('noun_vocabulary_final.txt','w') as f2:

    for noun_txt_idx in noun_txt:

        f2.write(df_noun.iloc[int(noun_txt_idx.rstrip()), df_noun.columns.get_loc('label')] + '\n')

# verb
df_verb = pd.read_csv('/data1/zhaofeng/TTA/EEL-benchmark-features-szf/EEL-annotations/narration_verb_taxonomy.csv')

with open('/data1/zhaofeng/TTA/EEL-benchmark-features-szf/EEL-annotations/valid_verbs.txt','r') as f1:
    verb_txt = f1.readlines()

with open('verb_vocabulary_final.txt','w') as f2:

    for verb_txt_idx in verb_txt:

        f2.write(df_verb.iloc[int(verb_txt_idx.rstrip()), df_verb.columns.get_loc('label')] + '\n')

