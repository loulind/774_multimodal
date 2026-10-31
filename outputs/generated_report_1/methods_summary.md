# Methods and preprocessing

Generated from the executed plan (source: llm). Final configuration: 'hard_infonce'.

## Data and split

- spatial_anndata; n=13,620; expression (matrix, 13,619), spatial (spatial, 13,620), image (image, 13,620); label 'label' 6 classes on 23% of samples; 8 'patient' groups
- Split: grouped_kfold, grouped by 'patient'; 4 fold(s). Every preprocessing statistic (feature selection, normalisation, PCA, label codes, regression targets) is fitted on the fold's training samples only. Model selection uses validation samples; test samples are scored once.

## Preprocessing and encoders

- **image** (image): phikon (iBOT self-supervision on 40M TCGA histology tiles; licence Owkin non-commercial), frozen; patch = 1.0 x the spot diameter, reinhard colour normalisation per image group.
- **expression** (matrix): library-size normalised + log1p; top 2000 variable features, standardised, PCA to 100; encoded by an MLP trained from scratch.
- Regression target: library-size normalised + log1p; top 2000 variable features, standardised.

## Representation learning

- Strategy: contrastive_alignment (joint fusion); projection heads of 2 layers (hidden 512, dropout 0.2) into a 256-dimensional L2-normalised joint space.
- Objective: 1.0 x infonce + 1.0 x cross_modal_regression.
- Optimiser AdamW, lr 0.001, weight decay 0.001, batch 512, up to 300 epochs, early stopping after 15 epochs without validation improvement, time cap 135s per fit; best validation epoch kept. Seed 774.

## Evaluation

- Primary metric: recall@10 (mean of both directions).
- Metrics: retrieval_recall_at_k, prediction_pearson, linear_probe, knn_accuracy, clustering_ari_nmi, retrieval_label_match_at_k, alignment_uniformity, modality_gap, effective_rank, group_silhouette. Baselines: unimodal, chance, cca, ridge, early_fusion, late_fusion.
- Uncertainty: mean and SD across folds when there are several; otherwise paired-bootstrap 95% intervals over held-out samples.
- Downstream analyses: retrieval_examples, cross_modal_prediction, spatial_maps.

## Comparison configurations

- **hard_infonce**: Tests the soft-target decision (near_twin_fraction 0.232); identical to primary otherwise.
- **patch_context_2x**: Tests the patch-size decision: 2 x 145 px = 290 px, about the centre-to-centre spacing (273 px), so it adds surrounding tissue context without reaching far into neighbouring spots.
- Critic revision 'hard_infonce': kept.
