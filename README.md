
# CLASS-CP

This repository contains source code of CLASS-CP, an extension for the CP (Canonical/Parafac) decomposition algorithm that combines tensor factorization and classification in a joint learning process for detecting fake news posts in social media. 

# Overview

As the detection of fake news is increasingly considered a technological problem, it has attracted considerable research. Most of these studies primarily focus on utilizing information extracted from textual news content. In contrast, we focus on detecting fake news solely based on structural information of social networks. We suggest that the underlying network connections of users that share fake news are discriminative enough to support the detection of fake news. Thereupon, we model each post as a network of friendship interactions and represent a collection of posts as a multidimensional tensor. Taking into account the available labeled data, we propose a tensor factorization method which associates the class labels of data samples with their latent representations. Specifically, we combine a classification error term with the standard factorization in a unified optimization process. Results on FakeNewsNet dataset demonstrate that our proposed method is competitive against state-of-the-art methods by implementing an arguably simpler approach. 

Link to the full paper that was presented as a poster in ROME2019 will be available soon. 



