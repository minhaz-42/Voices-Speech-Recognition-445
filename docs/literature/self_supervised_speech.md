# Literature notes: self-supervised speech models

Background for report section 2 (related work), deep-learning side, and for explaining the untrained wav2vec 2.0 baseline (T18) and the linear probe (T46). Task T38, M4 · Shafil.

Every reference below was checked against its arXiv page (title, authors, year) and, where it has one, the publisher's page or DOI record. Whisper is covered by M1; MFCC and classical classifiers by M3 (T37).

## Papers

##### wav2vec 2.0 (Baevski, 2020)
- **Reference:** A. Baevski, Y. Zhou, A. Mohamed, M. Auli. "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*. https://arxiv.org/abs/2006.11477
- **What they did:** A convolutional encoder turns raw audio into latent frames, part of them is masked, and a Transformer learns to pick the true quantised latent for each masked frame among distractors (a contrastive task), using unlabelled audio only. Fine-tuned on LibriSpeech, it reaches 1.8 / 3.3 WER (clean / other) with all labels, and still 4.8 / 8.2 with only ten minutes of labelled speech after pre-training on 53k hours.
- **What we take from it:** It is our main model (`facebook/wav2vec2-base`): with only 7 training speakers per fold and 1,000 clips, we depend on what it learnt without labels. The ten-minute result is why we expect fine-tuning to work on so little data, and why we freeze the CNN feature encoder during fine-tuning, as the authors do.

##### HuBERT (Hsu, 2021)
- **Reference:** W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, A. Mohamed. "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units." *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 29, 2021. https://arxiv.org/abs/2106.07447 · https://doi.org/10.1109/TASLP.2021.3122291
- **What they did:** Instead of a contrastive loss, HuBERT clusters the audio offline (k-means on MFCCs first, then on its own features in a second round) and trains a BERT-like Transformer to predict the cluster label of masked frames. It matches or beats wav2vec 2.0 on the LibriSpeech and Libri-light fine-tuning benchmarks.
- **What we take from it:** HuBERT Base (T45, M3) has the same size and input as wav2vec 2.0 Base but a different pre-training target, so fine-tuning both with the same script shows whether the pre-training objective matters for recognising our 50 phrases.

##### WavLM (Chen, 2022)
- **Reference:** S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, N. Kanda, T. Yoshioka, X. Xiao, J. Wu, L. Zhou, S. Ren, Y. Qian, Y. Qian, J. Wu, M. Zeng, X. Yu, F. Wei. "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing." *IEEE Journal of Selected Topics in Signal Processing*, vol. 16, 2022. https://arxiv.org/abs/2110.13900 · https://doi.org/10.1109/JSTSP.2022.3188113
- **What they did:** WavLM keeps HuBERT's masked prediction but trains on speech with other speakers or noise mixed in (denoising), adds a gated relative position bias to the Transformer and uses 94k hours of audio. WavLM Large reached the top of the SUPERB benchmark, with gains on speaker and paralinguistic tasks as well as recognition.
- **What we take from it:** Because WavLM saw overlapped and noisy speech in pre-training, WavLM Base+ (T44, M2) is the natural comparison for our noise tests: it shows whether robustness can come from pre-training rather than only from our white-noise augmentation.

##### XLS-R (Babu, 2021)
- **Reference:** A. Babu, C. Wang, A. Tjandra, K. Lakhotia, Q. Xu, N. Goyal, K. Singh, P. von Platen, Y. Saraf, J. Pino, A. Baevski, A. Conneau, M. Auli. "XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale." arXiv preprint, 2021. https://arxiv.org/abs/2111.09296
- **What they did:** The wav2vec 2.0 recipe scaled to models of up to 2 billion parameters, pre-trained on about 500,000 hours of public speech in 128 languages. It improves speech recognition (14–34% relative error reduction on average), speech translation and language identification, and at large size cross-lingual pre-training even helps English-to-other translation.
- **What we take from it:** Our Base models were pre-trained on English audiobooks only, while TenVoices-L2 speakers have five different first languages. XLS-R is the obvious next model for accented speech and is listed under *Future work* in the README.

##### SUPERB (Yang, 2021)
- **Reference:** S. Yang, P.-H. Chi, Y.-S. Chuang, C.-I J. Lai, K. Lakhotia, Y. Y. Lin, A. T. Liu, J. Shi, X. Chang, G.-T. Lin, T.-H. Huang, W.-C. Tseng, K. Lee, D.-R. Liu, Z. Huang, S. Dong, S.-W. Li, S. Watanabe, A. Mohamed, H. Lee. "SUPERB: Speech processing Universal PERformance Benchmark." *Proc. Interspeech 2021*. https://arxiv.org/abs/2105.01051
- **What they did:** A benchmark that keeps each self-supervised model **frozen** and trains only a light task-specific head on a weighted sum of its hidden layers, over ten tasks from phoneme recognition to emotion. Two tasks are close to ours: keyword spotting (Speech Commands v1.0) and intent classification on **Fluent Speech Commands**, the corpus TenVoices-FSC is drawn from. Frozen representations proved competitive across tasks.
- **What we take from it:** It is the model for our linear probe (T46): frozen wav2vec 2.0 features plus a simple classifier. Unlike SUPERB we use one layer, mean-pooled over time, and logistic regression, and we test on unseen speakers with only 10 of the 97 FSC speakers, so their numbers are a reference point, not a target.

##### Speech Commands (Warden, 2018)
- **Reference:** P. Warden. "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition." arXiv preprint, 2018. https://arxiv.org/abs/1804.03209
- **What they did:** A public keyword-spotting dataset of 105,829 one-second utterances of 35 words from 2,618 speakers, with the collection, verification and versioning process and a reproducible accuracy measure. The reference model in the paper reaches 88.2% top-one accuracy on version 2.
- **What we take from it:** Like Speech Commands, TenVoices is a closed-vocabulary task scored by accuracy, but our classes are whole commands and sentences that often share most of their words, and we have 10 speakers instead of thousands. Its emphasis on keeping speakers apart between training and test is the same idea as our speaker-independent folds.

##### Fluent Speech Commands and SLU pre-training (Lugosch, 2019)
- **Reference:** L. Lugosch, M. Ravanelli, P. Ignoto, V. S. Tomar, Y. Bengio. "Speech Model Pre-training for End-to-End Spoken Language Understanding." *Proc. Interspeech 2019*. https://arxiv.org/abs/1904.03670
- **What they did:** They introduce Fluent Speech Commands (30,043 utterances, 97 speakers, 19 hours, 248 distinct phrases mapped to 31 intents) and an end-to-end model that goes straight from audio to intent. Pre-training the lower layers to recognise phonemes and words first makes the model accurate even with small training subsets; their best model reaches 98.8% test accuracy.
- **What we take from it:** It is the source of TenVoices-FSC, and an early case of what we test: a speech model pre-trained on other data and then adapted to a small command set. Our task is harder in one way (10 speakers, unseen-speaker folds) and simpler in another (50 phrases instead of 248).
