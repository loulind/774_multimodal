# Methods and preprocessing

Generated from the executed plan (source: llm). Final configuration: 'primary'.

## Data and split

- table; n=27,000; image (image, 27,000), text (text, 26,999); label 'modality' 6 classes on 99% of samples; 21341 'article' groups; predefined split 20,000/2,000/5,000
- Split: predefined, grouped by 'article'; 1 fold(s). Every preprocessing statistic (feature selection, normalisation, PCA, label codes, regression targets) is fitted on the fold's training samples only. Model selection uses validation samples; test samples are scored once.

## Preprocessing and encoders

- **image** (image): biomedclip_image (PMC-15M: 15M figure-caption pairs from PubMed Central; licence MIT), frozen; greyscale replicated to RGB.
- **text** (text): biomedclip_text (PMC-15M: 15M figure-caption pairs from PubMed Central; licence MIT), frozen; text truncated at 128 tokens (basic cleaning).

## Representation learning

- Strategy: contrastive_alignment (joint fusion); projection heads of 2 layers (hidden 512, dropout 0.1) into a 512-dimensional L2-normalised joint space; residual heads (pretrained embedding + a learned correction that starts at zero), so training starts from the pretrained space and the untrained start is a candidate in model selection.
- Objective: 1.0 x infonce + 0.5 x supervised_ce.
- Optimiser AdamW, lr 0.001, weight decay 0.0001, batch 512, up to 300 epochs, early stopping after 15 epochs without validation improvement, time cap 180s per fit; best validation epoch kept. Seed 774.

## Evaluation

- Primary metric: recall@10 (mean of both directions).
- Metrics: retrieval_recall_at_k, linear_probe, knn_accuracy, clustering_ari_nmi, retrieval_label_match_at_k, alignment_uniformity, modality_gap, effective_rank. Baselines: unimodal, chance, cca, early_fusion, late_fusion, zero_shot.
- Uncertainty: mean and SD across folds when there are several; otherwise paired-bootstrap 95% intervals over held-out samples.
- Downstream analyses: retrieval_examples, classification.

## Comparison configurations

- **partial_finetune**: Tests the frozen-backbone decision. ~20k training pairs and about 2 epochs fit in the 1200 s fine-tuning budget on mps. Batch raised to 128 so the comparison has more negatives per positive than at 64.
- **contrastive_only**: Tests whether the supervised CE term helps classification without costing retrieval recall.
- Critic revision 'primary_revised': discarded.
