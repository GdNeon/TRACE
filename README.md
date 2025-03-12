# Egocentric early action prediction via multimodal transformer-based dual action prediction.（published on TCSVT, CCF-B)
## Abstract
Egocentric early action prediction, which aims to recognize the on-going action in the video captured in the first-person view as early as possible before the action is fully executed, is a new yet challenging task due to the limited partial video input. Pioneer studies focused on solving this task with LSTMs as the backbone and simply compiling the observed video segment and unobserved video segment into a single vector, which hence suffer from two key limitations: lack the non-sequential relation modeling with the video snippet sequence and the correlation modeling between the observed and unobserved video segment. To address these two limitations, in this paper, we propose a novel multimodal TransfoRmer-based duAl aCtion prEdiction (mTRACE) model for the task of egocentric early action prediction, which consists of two key modules: the early (observed) segment action prediction module and the future (unobserved) segment action prediction module. Both modules take Transformer encoders as the backbone for encoding all the potential relations among the input video snippets, and involve several single-modal and multi-modal classifiers for comprehensive supervision. Different from previous work, each of the two modules outputs two multi-modal feature vectors: one for encoding the current input video segment, and the other one for predicting the missing video segment. For optimization, we design a two-stage training scheme, including the mutual enhancement stage and end-to-end aggregation stage. The former stage alternatively optimizes the two action prediction modules, where the correlation between the observed and unobserved video segment is modeled with a consistency regularizer, while the latter seamlessly aggregates the two modules to fully utilize the capacity of the two modules. Extensive experiments have demonstrated the superiority of our proposed model.
## Pipeline
![pipeline](https://github.com/GdNeon/TRACE/edit/main/struct2.png) 
