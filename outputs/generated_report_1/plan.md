# Plan (llm)

Objective: "Learn joint embeddings of tissue image patches and spot-level gene expression to enable cross-modal retrieval and prediction of gene expression from histology."

- Tasks: joint_embedding, retrieval, cross_modal_prediction (primary: retrieval)
- Modalities aligned: image + expression; prediction image -> expression
- Split: grouped_kfold by 'patient', 4 fold(s)
- Primary metric: retrieval_mean_recall_at_10; metrics: retrieval_recall_at_k, prediction_pearson, linear_probe, knn_accuracy, clustering_ari_nmi, retrieval_label_match_at_k, alignment_uniformity, modality_gap, effective_rank, group_silhouette
- Baselines: unimodal, chance, cca, ridge, early_fusion, late_fusion
- Downstream: retrieval_examples, cross_modal_prediction, spatial_maps

## primary (primary)

Frozen phikon (best screening on recall@10, ridge R2 and probe F1) + MLP on 100 expression PCs; soft InfoNCE because 23% of expression spots have near-twins; joint regression head for image->expression prediction.

- Strategy: contrastive_alignment, fusion joint
- Losses: soft_infonce x 1.0, cross_modal_regression x 1.0
- Head: embedding 256, hidden 512, 2 layers, dropout 0.1
- Training: batch 512, lr 0.001, weight decay 0.0001, up to 300 epochs, patience 15, cap 180s per fit, temperature 0.07 (learned)
- image: phikon, frozen, patch_context 1.0, colour_normalisation reinhard, text_cleaning basic
- expression: mlp, from_scratch, text_cleaning basic, n_variable_features 2000, matrix_input pca, pca_components 100

## hard_infonce (comparison)

Tests the soft-target decision (near-twin fraction 0.232).

- Strategy: contrastive_alignment, fusion joint
- Losses: infonce x 1.0, cross_modal_regression x 1.0
- Head: embedding 256, hidden 512, 2 layers, dropout 0.1
- Training: batch 512, lr 0.001, weight decay 0.0001, up to 300 epochs, patience 15, cap 135s per fit, temperature 0.07 (learned)
- image: phikon, frozen, patch_context 1.0, colour_normalisation reinhard, text_cleaning basic
- expression: mlp, from_scratch, text_cleaning basic, n_variable_features 2000, matrix_input pca, pca_components 100

## patch_context_2x (comparison)

Tests the patch-size decision: spot spacing is 1.88x the spot diameter, so a 2x patch adds surrounding tissue context up to about the neighbouring spots without overlapping their centres.

- Strategy: contrastive_alignment, fusion joint
- Losses: soft_infonce x 1.0, cross_modal_regression x 1.0
- Head: embedding 256, hidden 512, 2 layers, dropout 0.1
- Training: batch 512, lr 0.001, weight decay 0.0001, up to 300 epochs, patience 15, cap 135s per fit, temperature 0.07 (learned)
- image: phikon, frozen, patch_context 2.0, colour_normalisation reinhard, text_cleaning basic
- expression: mlp, from_scratch, text_cleaning basic, n_variable_features 2000, matrix_input pca, pca_components 100
