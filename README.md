Applying Singular Value Decomposition on Paired Data
==============================
This is a submission of **assignment 2** for the **CIS726** course.

It contains the code necessary to conduct simple SVD transformation and inspect the resulting effects.

The dataset from the [eighth task](https://competitions.codalab.org/competitions/33835) in SemEval-2022 has been used.


Collaborators 🥇
------------
* Tariq Sha'ban
* Rand Agha
* Lujain Ghazalat


Getting Started
------------
Clone the project from GitHub

`$ git clone https://github.com/tariqshaban/svd-transformation.git`

No further configuration is required.


Usage
------------
Simply run the notebook on any IPython distro.


Methodology
------------
> ![methodology.png](assets/images/methodology.png)
> <br> <br>
> The main operations conducted in this repository are thus:
> * Read the CSV file in order to acquire the JSON files ID, along with the true class.
> * Fetch the textual data based on the JSON ID.
> * Removing rows in which that have empty/null text values.
> * Discarding unnecessary columns.
> * Applying NLP methods, which should improve the results.
> * Vectorize both Deutsch and English texts, based on user-selected vectorization technique.
> * Calculate the similarity of the vectors.
> * Calculate the correlation between the founded similarity and the true class prior to the SVD transformation.
> * Perform the SVD transformation
> * Calculate the similarity of the vectors after the transformation.
> * Calculate the correlation between the founded similarity and the true class after the SVD transformation.
> * Repeat the steps while modifying the following:
>     * Vectorization technique
>     * Sigma percentile, which denotes the extent of the dimensionality reduction for the SVD
>     * Whether to perform NLP lemmatization or not
>     * Whether to normalize the vectors or not 
> * Observe each parameter's effect on the correlation.

Findings
------------
> **NOTE:** Bold numbers denote a relatively noticeable improvement

<details>
  <summary>TF-IDF</summary>

| Method         | Correlation Before | Correlation After |
|:---------------|:------------------:|:-----------------:|
| Pearson        |    -48.18527587    |    -7.65953271    |
| Spearman       |    -51.96585235    |    -8.40274874    |
| Point Biserial |    -48.18527587    |    -7.65953271    |
| Kendall’s Tau  |    -41.21671761    |    -6.24882997    |

[Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

</details>

<br>

<details>
  <summary>Count Vectorizer</summary>

| Method         | Correlation Before | Correlation After |
|:---------------|:------------------:|:-----------------:|
| Pearson        |    -46.41424174    |    -11.2748719    |
| Spearman       |    -51.03216794    |    -12.5920341    |
| Point Biserial |    -46.41424174    |    -11.2748719    |
| Kendall’s Tau  |    -40.19397027    |    -9.5002537     |

[Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

</details>

<br>

<details>
  <summary>Hashing Vectorizer</summary>

|     Method     | Correlation Before | Correlation After |
|:--------------:|:------------------:|:-----------------:|
|    Pearson     |    -44.87897647    |   -20.94264864    |
|    Spearman    |    -48.06676036    |   -24.07069957    |
| Point Biserial |    -44.87897647    |   -20.94264864    |
| Kendall’s Tau  |    -38.58276655    |   -19.10211439    |

[Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html)

</details>

<br>

<details>
  <summary>Doc2Vec Vectorizer</summary>

| Method         | Correlation Before | Correlation After |
|----------------|:------------------:|:-----------------:|
| Pearson        |    -13.04976246    |  **-14.2113953**  |
| Spearman       |    -22.69244976    | **-26.81448982**  |
| Point Biserial |    -13.04976246    |  **-14.2113953**  |
| Kendall’s Tau  |    -17.93064231    | **-21.68947384**  |

[Documentation](https://radimrehurek.com/gensim/models/doc2vec.html)

</details>

<br>

<details>
  <summary>SentenceTransformers Vectorizer</summary>

| Method         | Correlation Before | Correlation After |
|----------------|:------------------:|:-----------------:|
| Pearson        |    -43.06046748    |   -11.38683546    |
| Spearman       |    -43.16662277    |    -7.58305467    |
| Point Biserial |    -43.06046748    |   -11.38683546    |
| Kendall’s Tau  |    -33.73352116    |    -6.29963347    |

[Documentation](https://www.sbert.net)

</details>

<br>

<details>
  <summary>InferSent Vectorizer</summary>

| Method         | Correlation Before | Correlation After |
|----------------|:------------------:|:-----------------:|
| Pearson        |    -33.22372089    |    -23.4371719    |
| Spearman       |    -34.21124055    |   -25.97421831    |
| Point Biserial |    -33.22372089    |    -23.4371719    |
| Kendall’s Tau  |    -27.22840212    |   -20.42130159    |

[Repository](https://github.com/facebookresearch/InferSent)

</details>

<br>

<details>
  <summary>Universal Sentence Encoder</summary>

| Method         | Correlation Before | Correlation After |
|----------------|:------------------:|:-----------------:|
| Pearson        |    -62.09828636    |   -14.98977225    |
| Spearman       |    -54.08919363    |   -11.55237896    |
| Point Biserial |    -62.09828636    |   -14.98977225    |
| Kendall’s Tau  |    -43.58939933    |    -8.73820126    |

[Documentation](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3)

</details>

Discussion
------------
After conducting the previous operations, we can conclude that:
* It appears that the correlations are negative in general, this is the expected behaviour since the true class is
  descending. Refer to the
  [SemEval-2022 eighth task data description](https://competitions.codalab.org/competitions/33835#learn_the_details-timetable);
  the website states: `This is a document-level similarity task in the applied domain of news articles,
  rating them pairwise on a 4-point scale from most to least similar`.
* The SVD transformation generally increased the correlation in different proportions,
  depending on the vectorization technique.
* The `Universal Sentence Encoder` transformation has negatively affected the correlation, due to the fact that the
  resulting vectors already project on the same dimensionality, hence, 
  finding the similarity does not require any transformation. Refer to [this notebook](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/cross_lingual_similarity_with_tf_hub_multilingual_universal_encoder.ipynb).
* It can be concluded that the `Universal Sentence Encoder` vectorization technique obtained the highest correlation 
  before the transformation.
* It can be concluded that the `InferSent` vectorization technique obtained the highest correlation after the
  transformation.

Notes
------------
* The correlation results are inconsistent, and liable to marginal change; due to seeding and randomization.
* Although the correlation has improved after the transformation, the results are generally poor and should not be
  trusted, further investigation is required.
* `Universal Sentence Encoder` is implemented on `TensorFlow`, it is recommended to capitalize the GPU resources; to
  result in an efficient execution.

--------