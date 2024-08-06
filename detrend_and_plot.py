
''''''
# functions for sentiment analysis (dictionary-based methods)
# and for plotting (adjusted from figs.py)
# for the Danish HCA SA study

''''''

import os
from utils import *

''''''
# SENTIMENT ANALYSIS
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr



''''''


# For SENTIMENT ANALYSIS

# to convert transformer scores to the same scale as the dictionary-based scores
def conv_scores(lab, sco, spec_lab): #insert exact labelnames in order positive, negative og as positive, neutral, negative
    
    converted_scores = []
    
    if len(spec_lab) == 2:
        spec_lab[0] = "positive"
        spec_lab[1] = "negative"

        for i in range(0, len(lab)):
            if lab[i] == "positive":
                converted_scores.append(sco[i])
            else:
                converted_scores.append(-sco[i])
            
    if len(spec_lab) == 3:
        spec_lab[0] = "positive"
        spec_lab[1] = "neutral"
        spec_lab[2] = "negative"
        
        for i in range(0, len(lab)):
            if lab[i] == "positive":
                converted_scores.append(sco[i])
            elif lab[i] == "neutral":
                converted_scores.append(0)
            else:
                converted_scores.append(-sco[i])
    
    return converted_scores

## SENTIMENT ANALYSIS

# make functions for AFINN & SENTIDA
def get_afinn_arc(sentences: list) -> list:
    afinn = Afinn(language="da")
    afinn_arc = [afinn.score(sentence) for sentence in sentences]
    return afinn_arc

def get_sentida_arc(sentences: list) -> list:
    SV = Sentida()
    sentida_arc = []
    for s in sentences:
        try:
            sentida_score = SV.sentida(text=s, output='mean', normal=True)
            sentida_arc.append(sentida_score)
        except UnboundLocalError:
            pass
    return sentida_arc


# # functions for ASENT
# load spacy pipeline
nlp = spacy.load('da_core_news_sm')

# add the rule-based sentiment model
nlp.add_pipe("asent_da_v1")

def get_asent_arc(text):
    doc = nlp(text)
    asent_arc = [sentence._.polarity.compound for sentence in doc.sents]
    return asent_arc

def get_asent_arc_w_nltk_tok(listed_sents):
    asent_arc = []
    for sent in listed_sents:
        doc = nlp(sent, disable=["sentencizer"])
        len_sents = [x for x in doc.sents]
        if len(len_sents) > 1:
            # get the average
            score = np.mean([x._.polarity.compound for x in doc.sents])
        else:
            score = doc._.polarity.compound
        asent_arc.append(score)
    return asent_arc


# SYUZHET
# Load Python libraries to exchange data with R Program Space and read R Datafiles
# To run syuzhet, make sure that syuzhet is installed in your local R environment

syuzhet = importr('syuzhet')

def syuzhet_sentiment(text, untokd=True):
    if untokd:
        sents = nltk.sent_tokenize(text)
        print(len(sents))
    else: sents = text
    syuzhet_score = syuzhet.get_sentiment(sents, method='syuzhet')
    return list(syuzhet_score)

# and VADER 


sid =  SentimentIntensityAnalyzer()

def sentimarc_vader(text, untokd=True):
    if untokd:
        sents = nltk.sent_tokenize(text)
        print(len(sents))
    else: sents = text
    arc=[]
    for sentence in sents:
        compound_pol = sid.polarity_scores(sentence)['compound']
        arc.append(compound_pol)
    return arc


# SENTIMENT ANALYSIS with transformers

# funtion for converting transformer scores
def conv_scores(lab, sco, spec_lab): #insert exact labelnames in order positive, negative og as positive, neutral, negative
    
    converted_scores = []
    
    if len(spec_lab) == 2:
        spec_lab[0] = "positive"
        spec_lab[1] = "negative"

        for i in range(0, len(lab)):
            if lab[i] == "positive":
                converted_scores.append(sco[i])
            else:
                converted_scores.append(-sco[i])
            
    if len(spec_lab) == 3:
        spec_lab[0] = "positive"
        spec_lab[1] = "neutral"
        spec_lab[2] = "negative"
        
        for i in range(0, len(lab)):
            if lab[i] == "positive":
                converted_scores.append(sco[i])
            elif lab[i] == "neutral":
                converted_scores.append(0)
            else:
                converted_scores.append(-sco[i])
    
    return converted_scores






## PLOTTING FUNCTIONS

# Plotting ARCS
# detrending fuctions
def integrate(x):
    return np.mat(np.cumsum(x) - np.mean(x))

# normalization
def normalize(ts, scl01 = False):
    ts01 = (ts - np.min(ts)) / (np.max(ts) - np.min(ts))
    ts11 = 2 * ts01 -1
    if scl01:
        return ts01
    else:
        return ts11
    
# plotting function
def figure(sentiment_arcs, method, plottitle, l=12, h=20, plot=True):

    plt.figure(figsize=(l, h), dpi=300)
    
    # Getting colors
    colors = sns.color_palette("Paired", len(sentiment_arcs))

    for j, story_arc in enumerate(sentiment_arcs):

        y = integrate(story_arc)
        uneven = y.shape[1] % 2
        if uneven:
            y = y[0, :-1]

        # afa
        # n = 500
        step_size = 1
        q = 3
        order = 1
        xy = md.multi_detrending(y, step_size, q, order)
        
        ## slope
        x = np.squeeze(np.asarray(xy[0]))
        y = np.squeeze(np.asarray(xy[1]))

        # p = np.poly1d(np.polyfit(x, y, order))
        # xp = np.linspace(0, len(x), len(x))

        sns.set_style("whitegrid")
        
        X = np.mat([float(x) for x in story_arc])
        #plt.plot(X.T, color=colors[j], linestyle='-', alpha=0.15, label=method[j]) # Uncomment to see the raw arc; changed to grey
        n = len(story_arc)
        w = int(4 * np.floor(n / 20) + 1)

        # format
        if method[j] == 'human' or method[j] == 'mean':
            for i in range(2, 3):
                try:
                    _, trend_ww_1 = dm.detrending_method(X, w, i)
                    plt.plot(normalize(trend_ww_1).T, label=str(method[j]), linewidth=2.5, color='k', linestyle='dashed')
                except:
                    print("error")
                    X = np.mat([float(x) for x in story_arc + [0]])
                    plt.plot(X.T, 'black', label='story arc')
                    n = len(story_arc)
                    w = int(4 * np.floor(n / 20) + 1)
        else:
            for i in range(2, 3):
                try:
                    _, trend_ww_1 = dm.detrending_method(X, w, i)
                    plt.plot(normalize(trend_ww_1).T, label=str(method[j]), linewidth=2.5, color=colors[j])
                except:
                    print("error")
                    X = np.mat([float(x) for x in story_arc + [0]])
                    plt.plot(X.T, 'black', label='story arc')
                    n = len(story_arc)
                    w = int(4 * np.floor(n / 20) + 1)
                    pass

    plt.title(plottitle)
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$valence$')

    plt.tight_layout()

    if plot == True:
        if os.path.exists('figures') == True:
            save_title = plottitle.split(' ')[:3]
            save_title = '_'.join(save_title)
            plt.savefig(f'figures/{save_title}_{str(len(sentiment_arcs))}arcs_plot.png')
    plt.show()  # Show the figure after all arcs are plotted

    H = round(np.polyfit(x, y, 1)[0], 2)
    print('Hurst: ', H)
    return

# Detrending from figs
def detrend(story_arc):
        X = np.mat([float(x) for x in story_arc])
        n = len(story_arc)
        w = int(4 * np.floor(n / 20) + 1)

        for i in range(2, 3):
            _, trend_ww_1 = dm.detrending_method(X, w, i)
        return normalize(trend_ww_1).T



def plot_kdeplots_or_histograms(df, scores_list, plot_type, plottitle, plts_per_row, l, h, kde=False):
    plots_per_row = plts_per_row
    num_scores = len(scores_list)

    # Calculate the number of rows needed
    rows = (num_scores + plots_per_row - 1) // plots_per_row
    
    fig, axes_list = plt.subplots(rows, plots_per_row, figsize=(l, h * rows), dpi=300)
    fig.tight_layout(pad=3)
    fig.suptitle(plottitle, fontsize=20, y=1.02)
    
    # Flatten axes_list for easy iteration, handle the case with a single row
    if rows == 1:
        axes_list = axes_list if num_scores > 1 else [axes_list]
    else:
        axes_list = axes_list.flat

    labels = [x.replace('_', ' ') for x in scores_list]

    for i, score in enumerate(scores_list):
        sns.set(style="whitegrid", font_scale=2, font='serif')

        ax = axes_list[i]

        if plot_type == 'histplot':
            if labels[i].startswith('tr'):
                sns.histplot(data=df[score].sort_values(), ax=ax, color='#38a3a5', kde=kde)
            elif labels[i].lower() == 'human':
                sns.histplot(data=df[score].sort_values(), ax=ax, color='lightgrey', kde=kde)
            else:
                sns.histplot(data=df[score].sort_values(), ax=ax, color='lightcoral', kde=kde)
        else:
            sns.kdeplot(data=df[score].sort_values(), ax=ax, log_scale=False, color='#38a3a5')

        # Set labels
        ax.set_xlabel(labels[i])
        if rows > 1 and i % plots_per_row != 0:
            ax.set_ylabel('')  # Remove y-label for non-leftmost plots

        # Ensure x-axis is sorted
        min_val, max_val = int(df[score].min()), int(df[score].max())

    # Remove empty subplots
    for j in range(i + 1, len(axes_list)):
        fig.delaxes(axes_list[j])

    plt.tight_layout()

    if not os.path.exists('figures'):
        os.makedirs('figures')
        
    save_title = '_'.join(plottitle.split()[:3])
    plt.savefig(f'figures/{save_title}_distribution.png', bbox_inches='tight')
    
    plt.show()
    return fig




# plotting BOXPLOTS for comparing two gorups
def pairwise_boxplots_canon(df, measures, category, category_labels, plottitle, outlier_percentile, h, w, remove_outliers=False, save=False):
# Only works for 5 boxplots for now!

    plots_per_row = len(measures) # just for now make number that are passed

    if len(measures) <= plots_per_row:
        fig, axes = plt.subplots(1, len(measures), figsize=(w, h), dpi=300)
    else:
        num_rows = math.ceil(len(measures) / plots_per_row)

        fig, axes = plt.subplots(num_rows, len(measures), figsize=(18, 8), dpi=300) # (18, 8 * rows), dpi=300)

    cat1_df = df.loc[df[category] == 1]
    cat2_df = df.loc[df[category] != 1]

    labels = [x.split('_')[1].lower() for x in measures]

    # Iterate over the significant columns
    for i, column in enumerate(measures):
        ax = axes[i]
        #df_dfered = df.loc[df[column].notnull()]
        cat1_df = cat1_df.loc[cat1_df[column].notnull()]
        cat2_df = cat2_df.loc[cat2_df[column].notnull()]
        
        # Boxplot
        ax.boxplot([cat1_df[column], cat2_df[column]],
                labels=category_labels,
                boxprops=dict(alpha=1, linewidth=1),
                widths=[0.75, 0.75], showfliers=False)
        ax.set_ylabel(labels[i], fontsize=24)


        # Scatterplot within boxplot
        colors = ['#C1666B', '#38a3a5']

        for j, group in enumerate([cat1_df, cat2_df]):
            column_data = group[column]

            if remove_outliers == True:
                # Calculate the 99.5th percentile
                percentile_95 = np.percentile(column_data, outlier_percentile)
                # dfer data points
                data = group[column][group[column] <= percentile_95]
            else:
                data = group[column]
            
            # creating random x coordinates to plot as a bulk
            x = np.random.normal(j + 1, 0.12, size=len(data))
            # Plot scatterpoints
            ax.plot(x, data, '.', alpha=0.65, color=colors[j], markersize=10)

    fig.suptitle(f'{plottitle}', fontsize=24)
    sns.set_style("whitegrid")
    plt.tight_layout()
    if save == True:
        plt.savefig(f'figures/features_boxplot_{plottitle}.png')
    # Show the plot
    plt.show()
    return fig

# plotting CEDs
## function to calculate KS test for two samples
def get_kstest(implicit_df, explicit_df, measure_list, labels):
    stats_all = []

    for i, measure in enumerate(measure_list):
        values_impl = [x for x in implicit_df[measure] if not pd.isna(x)]
        values_expl = [x for x in explicit_df[measure] if not pd.isna(x)]

        #a, b = [e[0] for e in measure[0]], [e[0] for e in measure[1]]
        ks_stat, ks_p_value = stats.ks_2samp(values_impl, values_expl)
        stats_all.append([round(ks_stat,3), round(ks_p_value,3)])
        print(f'{labels[i]} - KS Statistic: {ks_stat}, p-value: {ks_p_value}')

    return stats_all

## compute cdf
def compute_cdf(data):
        n = len(data)
        x = np.sort(data)
        y = np.arange(1, n+1) / n
        return x, y


### CED plot
def ced_plot(implicit_df, explicit_df, measure_list, labels, save=False):

    stats_all = get_kstest(implicit_df, explicit_df, measure_list, labels)

    apos = '**' # for p-value < 0.01

    fig, axes = plt.subplots(1, len(measure_list), figsize=(22, 4), sharey=True, dpi=500)

    for i, measure in enumerate(measure_list):
        #a, b = [e[0] for e in measure[0]], [e[0] for e in measure[1]]
        values_impl = implicit_df[measure]
        values_expl = explicit_df[measure]
        print(len(values_impl), len(values_expl))

        # Calculate CDF for each population
        x_a, y_a = compute_cdf(values_impl)
        x_b, y_b = compute_cdf(values_expl)

        sns.set_theme(style="whitegrid", font_scale=1.5)
        # Plotting CDF
        axes[i].plot(x_a, y_a, marker='.', markersize=8, alpha=0.45, linestyle='none', label='' if i > 0 else 'Implicit')
        axes[i].plot(x_b, y_b, marker='.', markersize=8, alpha=0.45, linestyle='none', label='' if i > 0 else 'Explicit')

        #axes[i].set_title(f'CED {labels[i]}')
        axes[i].set_title(f'CED {labels[i]}, KS: {stats_all[i][0]}{apos if stats_all[i][1] < 0.01 else ""}')

        axes[i].set_xlabel(f'{labels[i]}')

        # if i < 3:
        #     axes[i].legend_.remove()

    axes[0].set_ylabel('Cumulative Probability')

    axes[0].legend()  # Adding legend to the last subplot

    plt.tight_layout()

    if save == True:
        plt.savefig(f'figures/{str(len(measure_list))}_measures_CED.png')
    plt.show()

