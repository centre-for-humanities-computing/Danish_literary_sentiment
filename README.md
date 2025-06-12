# The Fiction2 Danish Literature Corpus [![Static Badge](https://img.shields.io/badge/acl-WASSA-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKAAAABwAgMAAADkn5ORAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAlQTFRFAAAA7R4k7RwkoncKaQAAAAF0Uk5TAEDm2GYAAAAJcEhZcwAACxMAAAsTAQCanBgAAAJVSURBVFjD7Zc7ksMwDEPTpMnptmGD023DhqdcApCT2NvYnJTxTH7K09CiSAi+3b7X23VHIQLxQbAiA/lREJFAzsCIyOQLUZVcQvCjYx9CnwZLYxwuENI0zUZNQUbtmw8mJHsePzt6z5qBDVX/lcxwpXZO6a4enoFKSFZTvRoCXIuwnIJgVKAa1JqgdTSHIahUdIq6CHQTTDTSuZ+B/ZuJ8ZqwtpOz+3ZmYJaKIXLVG/evv2pwBopy0MVoN8GJM1D5SFc/3AX9zowFZqAiKT+K6mrztCHIPoCaQcrEmGyJXmPmDOwKcBmk+iHUXKW5QzBiKwgo3SFFgMJPQQ7Gz/vYfaEzEJSRxO8O5NTCELTS1TF0/ivc06AkL2sX+hFWwiFIpas8gCGNmYFSThwTLiWoGahujUN6WgBZJ0NQ+o4d+IDldAaGy3aXHpZZK3XOQGvefgvvkqwYgrQEnfE9WBKCGIJs/8MWPiQEmIIUpWNRgCdoDUElgoUqRUke7kjLtIafAn0a1Flhped/9g2wQZLLyasgj9yKzXj55JT4ydvgJSznwfZuPVQRyxttJ6jekNdBtb+NUdhplY4O9jDLpa6DaQcn/1aAK8xHk/J1HWRM0CF5gg63lS2b2eugV2LpxMr+2td3m3QN1OazIGwNZbyWMrzq8SxYKzRrNL0OuTme9/I0cRVMLIte9gnhHijXbb0WcxpUPGCVlsL1Uio2h/N8WDkN2nLRc0HGCzov5Wwq483KnQZ1mtlZOtjm5NxwhQmY6zGiKFd6GGCr5noOyKvg9/peo+sPhLv+BGIWS+UAAAAldEVYdGRhdGU6Y3JlYXRlADIwMTMtMDgtMDRUMjE6NTc6MjkrMDg6MDAj62PfAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDEzLTA4LTA0VDIxOjU3OjI5KzA4OjAwUrbbYwAAAABJRU5ErkJggg==)](https://aclanthology.org/2024.wassa-1.15/) <a href="https://centre-for-humanities-computing.github.io/fabula-net/"><img src="https://github.com/yuri-bizzoni/EmoArc/raw/main/static/fabulanet_logo.png" width="15%" align="right"/></a> <a href="https://chc.au.dk"><img src="https://github.com/centre-for-humanities-computing/chicago_corpus/raw/main/documentation/img/CHC_logo.png" width="7%" align="right"/></a>


## 🌡️📖 Danish literary sentiment

This repository holds the data for comparing Sentiment Analysis methods on Danish literature - specifically fairy tales and religious hymns of the 19th century.
Our study compares human annotations to the continuous valence scores of both transformer- and dictionary-based sentiment analysis methods to assess their performance, seeking to understand how distinct methods handle the language of Danish prose and poetry.

## 🔎 What is included

- Original and modernized Danish text
- Continuous valence annotation (0-10) by human annotators (n=2-3) per sentence/verse
- Automatic annotation scores per sentence/verse (using dictionary- and transformer-based Sentiment Analysis tools)

This data allows for the comparison of human/human and human/model sentiment scoring on Danish literary texts.
<img src="https://github.com/centre-for-humanities-computing/Danish_literary_sentiment/raw/main/figures/Den_Lille_Havfrue_8arcs_plot.png" width="100%" align="right" />
<img src="https://github.com/centre-for-humanities-computing/Danish_literary_sentiment/raw/main/figures/Den_Lille_Havfrue_4arcs_plot.png" width="100%" align="right" />

## 🔬 Data

We use two datasets: i) H.C. Andersen fairy tales, and ii) Religious hymns

|             | No. texts | No. annotations   | No. words  | Mean no. verses/sents per text | Period     |
|-------------|-----|------|--------|--------------|------------|
| **HCA**     | 3   | 791   | 18,910 | 263.7        | 1837-1847  |
| **Hymns**   | 65  | 1,914 | 10,303 | 32.9         | 1798-1873  |

## 📖 Documentation

Code for the **hymns** and **fairy tales** analysis (separately) -- annotator agreement and human/model correlations -- is available in this folder, while the SHAP values analysis of RoBERTa scores is available in [a seperate GitHub repository](https://github.com/centre-for-humanities-computing/fabula-shap).

|                             |                                                                                   |
| --------------------------- | --------------------------------------------------------------------------------- |
| 📄   **[Paper]**              | Link to our paper comparing SA resources on Danish literary texts.                                                       |
| 🔬    **[CHC]**        | Center for Humanities Computing, hosting the project.                            |
| ✉️    **[Contact]**        | Contact the authors.                           |





[Paper]: https://aclanthology.org/2024.wassa-1.15/
[CHC]: https://chc.au.dk
[Contact]: mailto:pascale.moreira@cc.au.dk


