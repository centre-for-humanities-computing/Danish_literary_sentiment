''''''
# This script is used to analyze the sentiment of the HCA data.
# Annotator data is loaded from the HCA_annot data folder and the sentiment is analyzed with various libraries.

# The sentiment analysis is done with the following libraries:
# - Afinn
# - Sentida
# - Asent
# - Various tranformer models (not implemented in this script, data is loaded remotely (data folder))

# The sentiment analysis is done on the sentence level.
# The sentiment scores are then used to calculate the sentiment arcs.

# By default, the script opens the txt that is in the data folder (not the txt folder)

''''''
# %%
#%pip install -r requirements.txt

# %%
import os

from utils import *
from detrend_and_plot import *


# %%
# set out path for visualizations
output_path = 'figures/'

# %% 
# just getting some stats for the txts

for file in os.listdir("data/txt/"):
    if file.endswith(".txt"):
        name = file
        title = name.split('_')[-1].split('.')[0]
        full_title = name.split('.')[0].split('_')[1:]
        full_title = ' '.join(full_title).title()
        with open(f"data/txt/{name}", 'r') as f:
            text = f.read()
        print('\n', name)
        print('len text:', len(text))
        print('len sents:', len(nltk.sent_tokenize(text)))
        print('len words:', len(nltk.word_tokenize(text)))


#%%
# set the story you want to analyze
story = 'havfrue'

# %%
# open a txt in data
for file in os.listdir("data/txt/"):
    if file.endswith(".txt"):
        if story in file:
            print('filename treated:', file)
            name = file
            title = name.split('_')[-1].split('.')[0]

            full_title = name.split('.')[0].split('_')[1:]
            full_title = ' '.join(full_title).title()
            print(full_title)

with open(f"data/txt/{name}", 'r') as f:
    text = f.read()

# %%
# get the google translated sentences
df_data = pd.read_excel(f'data/xlsx/{title}_sentiment_data.xlsx')
df_data.columns

# %%
# get the English sentences
from deep_translator import GoogleTranslator

sents = list(df_data['sents'])

# Translating all sentences
translated_sents = []

for sent in sents:
    translated = GoogleTranslator(source='da', target='en').translate(sent)
    translated_sents.append(translated)

print('len en sents:', len(translated_sents))

df_data['en_sents'] = translated_sents
    
# %%
sents = nltk.sent_tokenize(text)
sents_en = list(df_data['en_sents'])
print('len sents:', len(sents), len(sents_en))

# %%
# sentence tokennize
nlp = spacy.load('da_core_news_sm')
doc = nlp(text)

spacy_sents = [str(x) for x in doc.sents]

print('len tokenized (spacy), len tokenized (nltk):', len([sent for sent in doc.sents]), len(sents))


# %%
# SA (takes a bit)
sent_methods = ['afinn', 'sentida']

afinn = get_afinn_arc(sents)
sentida = get_sentida_arc(sents)

print('len of dictionary-based arcs:', len(afinn), len(sentida))
# test
print(afinn[-3:])
syu[-3:]

# %% and we get the syuzhet and vader for comparison
# applied on en sentences
vader = sentimarc_vader(sents_en, untokd=False)
syu = syuzhet_sentiment(sents_en, untokd=False)

# %%
# Getting human scores
# with open(f'data/{title}_all_human_scores.json', 'r') as f:
#     human_scores = json.load(f)
# print('human arcs keys:', human_scores.keys())

len(df_data['PASC']), len(df_data['MIA']), len(df_data['EA'])

# df = pd.DataFrame.from_dict(human_scores, orient='index').T
# df.head()

# %%
df_data['afinn'] = afinn
df_data['sentida'] = sentida
df_data['vader'] = vader
df_data['syuzhet'] = syu
df_data.head(5)

# %%
# wrong label correction
df_data['asent'] = df_data['tr_dacy']

#all_scores = Merge(machine_scores, dictbased_scores)
human_scores = {'A1': list(df_data['PASC']), 'A2': list(df_data['MIA']), 'A3': list(df_data['EA']), 'mean': list(df_data['MEAN'])}
all_scores = {'mean': list(df_data['MEAN']), #'asent': list(df['asent']), 
              'afinn': list(df_data['afinn']), 'sentida': list(df_data['sentida']), 
              'tr_alexandra': list(df_data['tr_alexandrainst']), 'tr_senda': list(df_data['tr_senda']), 'tr_xlm_roberta': list(df_data['tr_xlm_roberta']), 'asent': list(df_data['asent'])}

all_scores_plus = {'mean': list(df_data['MEAN']), #'asent': list(df['asent']), 
                   'afinn': list(df_data['afinn']), 'sentida': list(df_data['sentida']), 
              'tr_alexandra': list(df_data['tr_alexandrainst']), 'tr_senda': list(df_data['tr_senda']), 'tr_xlm_roberta': list(df_data['tr_xlm_roberta']), 'asent': list(df_data['asent']), 'vader': list(df_data['vader']), 'syuzhet': list(df_data['syuzhet'])}



print('len full dict:', len(all_scores))
print(all_scores.keys())


# %%
# plot the arcs of annotators and the mean
title_story = f'{full_title} H.C. Andersen'
#labels_annot = ['A1', 'A2', 'A3', 'mean']

figure(human_scores.values(), list(human_scores.keys()), title_story, l=18, h=5)
figure(all_scores.values(), list(all_scores.keys()), title_story, l=18, h=5)

# %%
# inter annotator reliability
# Spearman correlation between annotators

pairs = [('A1', 'A2'), ('A1', 'A3'), ('A2', 'A3')]
average = []
for pair in pairs:
    correlation, p_value = spearmanr(human_scores[pair[0]], human_scores[pair[1]])
    print(pair, "Spearman:", round(correlation, 3), "p-value:", round(p_value,5))
    average.append(correlation)
print('mean corr (spearman): ', round(sum(average) / 3,4))

plt.figure(figsize=(8,8), dpi=500)
df_annot = pd.DataFrame.from_dict(human_scores)
df_annot.columns = ['A1', 'A2', 'A3', 'mean']
corr_annot = df_annot.corr(method='spearman')
sns.heatmap(corr_annot, cbar=False, annot=True)
plt.yticks(rotation=60)
plt.savefig(f'{output_path}{title}_annotator_correlation_raw.png')
plt.show()

# IRR 2, fleiss & krippendorff
# Need to transpose here cause fleiss expects certain format
transposed = np.array([x for x in human_scores.values()]).transpose()
print('fleiss_kappa: ', irr.fleiss_kappa(irr.aggregate_raters(transposed)[0], method='fleiss'))
# from web: Note that Fleiss is not perfectly applicable to a rating situation with a relative metric: it assumes that this is a classification task, not a ranking. 
# Fleiss is not sensitive to how far apart the ratings are; it knows only that the ratings differed: a (0,1) paring is just as damaging as a (0,3) pairing.

# No need to transpose for krippendorff
print('krippendorf: ', kd.alpha([x for x in human_scores.values()], level_of_measurement='interval')) # steven's level of measurement must be one of 'nominal', 'ordinal', 'interval', 'ratio'
# I'm assuming here that our ratings are on interval scale (absolute 0 and absolute 10) and not ordinal

# Ok, it's perhaps nice to do, but we can also just go with the correlation

# %%
# plot the arcs of the dictionary-based methods & transformer-based & human mean
title_story = f'{full_title} H.C. Andersen'

figure([x for x in all_scores.values()], [x for x in all_scores.keys()], title_story, l=18, h=5)

# %%
# correlation of raw arcs
df = pd.DataFrame.from_dict(all_scores)

correlation_matrix = df.corr(method='spearman')

plt.figure(figsize=(7, 4), dpi=500)
sns.heatmap(correlation_matrix, annot=True, cbar=False)
plt.xticks(rotation=40, ha='right')
plt.savefig(f'{output_path}{title}_systems_correlation_raw.png')
plt.show()


# %%

syu_values = syuzhet_sentiment(list(df_data['en_sents']), untokd=False)
vad_values = sentimarc_vader(list(df_data['en_sents']), untokd=False)

df['SYUZHET'] = syu_values
df['VADER'] = vad_values
# %%

df = df[['mean', #'SENTENCE','SENTENCE_ENGLISH'
       'afinn', 'sentida', 'tr_senda', 'tr_alexandra', 'asent', #'asent',
       'tr_xlm_roberta']]# 'SYUZHET', 'VADER']]
# %%
# Now correlations of detrended arcs
# Detrending from figs
dict_df = df.to_dict()
# Detrend arcs and make new list of data
arcs = [x for x in dict_df.values()]
labels = [x for x in dict_df.keys()]

detrended_arcs = []

for arc in arcs:
    d_arc = detrend(arc)
    detrended_arcs.append(d_arc)
detrended_arcs = np.squeeze(detrended_arcs)

detrended_df = pd.DataFrame(detrended_arcs, index=labels).T
detrended_dict = detrended_df.to_dict()

title_story = 'Den Lille Havfrue, H.C. Andersen'
figure([x for x in all_scores.values()], [x for x in all_scores.keys()], title_story, l=18, h=5)

# %%
# correlations between human detrended
# Detrend arcs and make new list of data
arcs_humans = [human_scores['A1'], human_scores['A2'], human_scores['A3'], human_scores['mean']]

detrended_arcs_humans = []
for arc in arcs_humans:
    d_arc = detrend(arc)
    detrended_arcs_humans.append(d_arc)
detrended_arcs_human = np.squeeze(detrended_arcs_humans)

labels_humans = ['A1', 'A2', 'A3', 'mean']

detrended_df = pd.DataFrame(detrended_arcs_human, index=labels_humans).T
correlation_matrix = detrended_df.corr(method='spearman')

plt.figure(figsize=(8, 8), dpi=500)
sns.heatmap(correlation_matrix, annot=True, cbar=False)
plt.yticks(rotation=60)
plt.savefig(f'{output_path}{title}_annotator_correlation_detrended.png')
plt.show()


# %%
df.columns
# %%
# Intermission

# we want to get two matrices, one detrended and one not detrended with the syuzhet and vader as well

# correlation of modern vs. original arcs
df = pd.DataFrame.from_dict(all_scores_plus)
m_vs_o = df[['mean', 'afinn', 'sentida','tr_senda', 'tr_alexandra',
       'asent', 'tr_xlm_roberta', 'vader', 'syuzhet']]

correlation_matrix = m_vs_o.corr(method='spearman')

# define the vertical columns
selected_rows = ['mean']
# define the horizontal columns
select = ['afinn', 'sentida','tr_senda', 'tr_alexandra',
       'asent', 'tr_xlm_roberta', 'vader', 'syuzhet']
# round
selected_data = round(correlation_matrix[select].loc[selected_rows], 2)

plt.figure(figsize=(15, 1.5), dpi=500)
sns.heatmap(selected_data, annot=True, cbar=False)
plt.xticks(rotation=40, ha='right')
plt.savefig(f'{output_path}_{name}_with_syuzhet.png')
plt.show()

# %%
# and detrended as well (with syuzhet and vader )

m_vs_o = df[['mean', 'afinn', 'sentida',
       'asent', 'tr_senda', 'tr_alexandra',
       'tr_xlm_roberta', 'vader', 'syuzhet']]

correlation_matrix = detrended_df_plus.corr(method='spearman')

# define the vertical columns
selected_rows = ['mean']
# define the horizontal columns
select = ['afinn', 'sentida',
       'asent', 'tr_senda', 'tr_alexandra',
       'tr_xlm_roberta', 'vader', 'syuzhet']
# round
selected_data = round(correlation_matrix[select].loc[selected_rows], 2)

plt.figure(figsize=(15, 1.5), dpi=500)
sns.heatmap(selected_data, annot=True, cbar=False)
plt.xticks(rotation=40, ha='right')
plt.savefig(f'{output_path}_{name}_with_syuzhet.png')
plt.show()


# %%
# human correlations when detrended

# Detrend arcs and make new list of data
labels = ['A1', 'A2', 'A3']
detrended_arcs_humans_dict = dict(zip(labels, detrended_arcs_humans))

pairs = [('A1', 'A2'), ('A1', 'A3'), ('A2', 'A3')]
average = []
for pair in pairs:
    correlation, p_value = spearmanr(detrended_arcs_humans_dict[pair[0]], detrended_arcs_humans_dict[pair[1]])
    print(pair, "Spearman:", round(correlation, 3), "p-value:", round(p_value,5))
    average.append(correlation)
print('mean corr (spearman): ', round(sum(average) / 3,4))

# IRR 2, fleiss & krippendorff
reformatted_data_human_detrended = [detrended_df['A1'], detrended_df['A2'], detrended_df['A3']]
# Need to transpose here cause fleiss expects certain format
transposed = np.array([x for x in reformatted_data_human_detrended]).transpose()
print('fleiss_kappa: ', irr.fleiss_kappa(irr.aggregate_raters(transposed)[0], method='fleiss'))
# # from web: Note that Fleiss is not perfectly applicable to a rating situation with a relative metric: it assumes that this is a classification task, not a ranking. 
# Fleiss is not sensitive to how far apart the ratings are; it knows only that the ratings differed: a (0,1) paring is just as damaging as a (0,3) pairing.

# No need to transpose for krippendorff
print('krippendorf: ', kd.alpha([x for x in reformatted_data_human_detrended], level_of_measurement='interval')) # steven's level of measurement must be one of 'nominal', 'ordinal', 'interval', 'ratio'
# I'm assuming here that our ratings are on interval scale (absolute 0 and absolute 10) and not ordinal

# %%
# A few visualizations of the distributions of raw values (non-detrended)


dist_data = pd.DataFrame.from_dict(all_scores)
scores = list(dist_data.columns)

res = plot_kdeplots_or_histograms(dist_data, scores, 'histplot', title_story, 4, l=30, h=4)


# %%
# all done
print('All done')
# %%
