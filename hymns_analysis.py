''''''
# This script is used to analyze the sentiment of the Hymns data.
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


#%%
with open(f"data/HYMNS_all_values.json", 'r') as f:
    all_data = json.load(f)

merged = pd.DataFrame.from_dict(all_data)

# %%
# inter annotator reliability
# Spearman correlation between annotators

correlation, p_value = spearmanr(merged['ANNOTATOR_1'], merged['ANNOTATOR_2'])
print("IRR: Spearman:", round(correlation, 3), "p-value:", round(p_value,5))

# and krippendorff
from krippendorff import alpha as krippendorff_alpha

# Convert annotation data to float
annotator_1_float = merged['ANNOTATOR_1'].astype(float)
annotator_2_float = merged['ANNOTATOR_2'].astype(float)

# Calculate Krippendorff's alpha
krip = krippendorff_alpha([annotator_1_float, annotator_2_float])
print("IRR: Krippendorff:", round(krip, 3))

# %%
text_list = list(merged['SENTENCE'])
text_list_modern = list(merged['SENTENCE_MODERN'])

print('len of text:', len(text_list))

# #%%
nlp = spacy.load('da_core_news_sm')

# SA (takes a bit)
sent_methods = ['afinn', 'sentida']

afinn = get_afinn_arc(text_list)
sentida = get_sentida_arc(text_list)
print('len of dictionary-based arcs:', len(afinn), len(sentida))

afinn_modern = get_afinn_arc(text_list_modern)
sentida_modern = get_sentida_arc(text_list_modern)

print('len of dictionary-based arcs, modern:', len(afinn_modern), len(sentida_modern))#, len(asent_modern))
# Inspect
#print(asent[-3:])
print(afinn[-3:])
sentida[-3:]

# %%
# We could also get syuzhet arcs
text_list_english = list(merged['SENTENCE_ENGLISH'])

# Syuzhet & VADER
syu = syuzhet_sentiment(text_list_english, untokd=False)
vader = sentimarc_vader(text_list_english, untokd=False)

# %%
merged['afinn'] = afinn
merged['afinn_MODERN'] = afinn_modern

merged['sentida'] = sentida
merged['sentida_MODERN'] = sentida_modern

merged['syuzhet'] = syu
merged['vader'] = vader
merged.head(5)

# %%
# correlation of raw arcs
columns = ['HUMAN', 'tr_alexandrainst', 'tr_senda', 'tr_xlm_roberta', 'asent', 'afinn',
       'sentida'] # 'ANNOTATOR_1', 'ANNOTATOR_2',
correlation_matrix = merged[columns].corr(method='spearman')

plt.figure(figsize=(14, 5), dpi=500)
sns.heatmap(correlation_matrix, annot=True, cbar=False)
plt.xticks(rotation=40, ha='right')
plt.savefig(f'{output_path}HYMNS_systems_correlation_raw.png')
plt.show()

# %%
columns = ['HUMAN', 'tr_alexandrainst_MODERN', 'tr_senda_MODERN', 'tr_xlm_roberta_MODERN', 'asent_MODERN', 'afinn_MODERN',
       'sentida_MODERN', 'syuzhet', 'vader'] # 'ANNOTATOR_1', 'ANNOTATOR_2',
correlation_matrix = merged[columns].corr(method='spearman')
selected_rows = ['HUMAN']
select = ['tr_alexandrainst_MODERN', 'tr_senda_MODERN', 'tr_xlm_roberta_MODERN', 'asent_MODERN', 'afinn_MODERN',
       'sentida_MODERN', 'syuzhet', 'vader'] 
selected_data = round(correlation_matrix[select].loc[selected_rows], 2)
plt.figure(figsize=(14, 1.5), dpi=500)
sns.heatmap(selected_data, annot=True, cbar=False)
plt.xticks(rotation=40, ha='right')
plt.savefig(f'{output_path}HYMNS_systems_correlation_raw.png')
plt.show()

# %%
# correlation of modern vs. original arcs
m_vs_o = merged[['HUMAN','sentida_MODERN','sentida', 'asent_MODERN', 'asent', 
               'afinn_MODERN', 'afinn',
         'tr_alexandrainst_MODERN', 'tr_alexandrainst', 'tr_senda_MODERN', 'tr_senda',
           'tr_xlm_roberta_MODERN', 'tr_xlm_roberta']]

correlation_matrix = m_vs_o.corr(method='spearman')

# define the vertical columns
selected_rows = ['HUMAN']
# define the horizontal columns
select = ['sentida_MODERN','sentida', 'asent_MODERN', 'asent', 
               'afinn_MODERN', 'afinn',
         'tr_alexandrainst_MODERN', 'tr_alexandrainst', 'tr_senda_MODERN', 'tr_senda',
           'tr_xlm_roberta_MODERN', 'tr_xlm_roberta']
# round
selected_data = round(correlation_matrix[select].loc[selected_rows], 2)

plt.figure(figsize=(15, 1.5), dpi=500)
sns.heatmap(selected_data, annot=True, cbar=False)
plt.xticks(rotation=40, ha='right')
plt.savefig(f'{output_path}HYMNS_systems_correlation_modern_vs_original.png')
plt.show()

# this computes the p-values
pvalues = m_vs_o.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(len(m_vs_o.columns)) 
pvalues

# %%
# A few visualizations of the distributions of raw values (non-detrended)
dist_data = merged[columns]

res = plot_kdeplots_or_histograms(dist_data, columns, 'histplot', 'HYMNS',5, l=30, h=4)

# %%
df1 = all_data.loc[all_data['YEAR'] == 1873]
df2 = all_data.loc[all_data['YEAR'] == 1857]
df3 = all_data.loc[all_data['YEAR'] == 1798]
dfs = [df1, df2, df3]   

years = [1873, 1857, 1798]

columns = ['HUMAN', 'tr_alexandrainst_MODERN', 'tr_senda_MODERN', 'tr_xlm_roberta_MODERN', 'asent_MODERN', 'afinn_MODERN',
       'sentida_MODERN', 'syuzhet', 'vader'] # 'ANNOTATOR_1', 'ANNOTATOR_2',
for df in dfs:
    correlation_matrix = df[columns].corr(method='spearman')
    selected_rows = ['HUMAN']
    select = ['tr_alexandrainst_MODERN', 'tr_senda_MODERN', 'tr_xlm_roberta_MODERN', 'asent_MODERN', 'afinn_MODERN',
        'sentida_MODERN', 'syuzhet', 'vader'] 
    selected_data = round(correlation_matrix[select].loc[selected_rows], 2)
    plt.figure(figsize=(14, 1.5), dpi=500)
    sns.heatmap(selected_data, annot=True, cbar=False)
    plt.xticks(rotation=40, ha='right')
    plt.show()


# %%
# We want to plot the sentence that have most disagreement

# normalize
all_data['sentida_MOD_NORM'] = normalize(all_data['sentida_MODERN'])
all_data['sentida_NORM'] = normalize(all_data['sentida'])
all_data['HUMAN_NORMALIZED'] = normalize(all_data['HUMAN'])


# We fint the disagreement by looking at the difference between the human and the model scores
all_data['DIFF_HUMAN_SENTIDA'] = abs(abs(all_data['HUMAN_NORMALIZED']) - abs(all_data['sentida_NORM']))
all_data['diff_sentida_MODERN'] = abs(abs(all_data['HUMAN_NORMALIZED']) - abs(all_data['sentida_MOD_NORM']))


all_data['DIFF_HUMAN_ROBERTA'] = abs(abs(all_data['HUMAN_NORMALIZED']) - abs(all_data['tr_xlm_roberta']))

# %%
import matplotlib as mpl
# Set font family to serif
mpl.rcParams['font.family'] = 'serif'

## Sort the DataFrame by the absolute values of 'diff_Sentida' column
sorted_df_sentida = all_data.reindex(all_data['DIFF_HUMAN_SENTIDA'].sort_values(ascending=False).index)
my_range = range(1, len(sorted_df_sentida.index) +1)

# Select the top 10 rows
top_15_diff = sorted_df_sentida.head(20)

# Reverse the order of the DataFrame
top_15_diff = top_15_diff[::-1].reset_index(drop=True)


y_labels = top_15_diff['SENTENCE'] + "\n" + "EN: " +  top_15_diff['SENTENCE_ENGLISH'] + "\n"

# Plotting
plt.figure(figsize=(8.5, 16))
sns.set_style("whitegrid")

# Plotting horizontal lines and adding text
for index, row in top_15_diff.iterrows():
    plt.hlines(index, xmin=row['HUMAN_NORMALIZED'], xmax=row['sentida_NORM'], color='grey', alpha=0.35, linewidth=3.5)  # Horizontal line
    plt.hlines(index, xmin=row['HUMAN_NORMALIZED'], xmax=row['tr_xlm_roberta'], color='grey', alpha=0.5, linestyle='--', linewidth=2)  # Horizontal line
    human_dot = plt.scatter(row['HUMAN_NORMALIZED'], index, s=100, color='blue', alpha=1)  # Human dot with adjusted size
    sentida_dot = plt.scatter(row['sentida_NORM'], index, s=110, marker='^', color='lightcoral', alpha=1)  # Sentida dot with adjusted size and marker
    roberta_dot = plt.scatter(row['tr_xlm_roberta'], index, s=100, marker='x', color='green', alpha=1)  # RoBERTa dot with adjusted size and marker
    #plt.text(row['HUMAN_NORM'], index + 0.2, row['SENTENCE'], ha='right', va='center', fontsize=10)  # Text

# Set y-tick labels as sentence strings
plt.yticks(range(len(top_15_diff)), y_labels, fontsize=11, ha='right')  # Adjusted yticks


# Set labels and title
plt.xlabel('Valence')

# Show legend with manually specified labels
plt.legend([human_dot, sentida_dot, roberta_dot], ['Human', 'Sentida', 'RoBERTa'],loc='lower right', fontsize=9.5)

plt.tight_layout()
plt.show()



# %% 
# EXTRA

# %%
from seaborn_qqplot import pplot

for i, df in enumerate(dfs):
    plt.figure(figsize=(20, 10))
    pplot(data=all_data, x='HUMAN_NORMALIZED', y='sentida', kind='qq') # alpha=0.5,
    plt.title(f'HYMNS {years[i]}')
    plt.show()

# %%
# normalize the human values
all_data['DIFF_HUMAN_SENTIDA'] = abs(all_data['HUMAN_NORMALIZED']) - abs(all_data['sentida_MODERN'])

# do a small correlation
correlation, p_value = spearmanr(all_data['sentida_MODERN'], all_data['YEAR'])
print('corr:', correlation, 'p:', p_value)

# visualize
sns.scatterplot(data=all_data, x='YEAR', y='DIFF_HUMAN_SENTIDA', alpha=0.5)

# %%
# all done
print('All done')





# %%
