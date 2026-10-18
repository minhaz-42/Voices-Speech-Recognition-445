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
