# COVID19 search engine

Group project assignment for [Text Analysis and Retrieval](https://www.fer.unizg.hr/en/course/taar) course held at
 [University of Zagreb, Faculty of Electrical Engineering and Computing](https://www.fer.unizg.hr/en). <br>

Amid a global crisis caused by COVID-19 pandemic, 
we decided to tackle [COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) and make our humble contribution by creating a search engine for
COVID-related information. A total of six versions of search engines are made and evaluated. 

Each model assigns a vector representation to every paragraph in the corpus sorts paragpraph by cosine similarity with generated incoming query vector representation. 
All our code is available in this repo, you can check our [System description paper](https://www.fer.unizg.hr/_download/repository/TAR-2020-ProjectReports.pdf#page=33&zoom=100,76,94) for more details.

You can see our model in action in this [notebook](https://github.com/patrikmaric/COVID19-search-engine/blob/master/notebooks/demo.ipynb).
