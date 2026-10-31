# Plan (llm)

Objective: "Learn joint embeddings of tissue image patches and spot-level gene expression to enable cross-modal retrieval and prediction of gene expression from histology."

- Tasks: joint_embedding, retrieval, cross_modal_prediction (primary: retrieval)
- Modalities aligned: image + expression; prediction image -> expression
- Split: grouped_kfold by 'patient', 4 fold(s)
- Primary metric: retrieval_mean_recall_at_10; metrics: retrieval_recall_at_k, prediction_pearson, linear_probe, knn_accuracy, clustering_ari_nmi, retrieval_label_match_at_k, alignment_uniformity, modality_gap, effective_rank, group_silhouette
- Baselines: unimodal, chance, cca, ridge, early_fusion, late_fusion
- Downstream: retrieval_examples, cross_modal_prediction, spatial_maps

## primary (primary)

Frozen phikon (best on all three screening metrics) + MLP on 100 PCs of 2,000 HVGs; soft InfoNCE for 23% near-twin expression spots; joint image→expression regression head; slightly stronger regularisation because each fold trains on only 5 patients and patient explains 77% of expression variance.

- Strategy: contrastive_alignment, fusion joint
- Losses: soft_infonce x 1.0, cross_modal_regression x 1.0
- Head: embedding 256, hidden 512, 2 layers, dropout 0.2
- Training: batch 512, lr 0.001, weight decay 0.001, up to 300 epochs, patience 15, cap 180s per fit, temperature 0.07 (learned)
- image: phikon, frozen, patch_context 1.0, colour_normalisation reinhard, text_cleaning basic
- expression: mlp, from_scratch, text_cleaning basic, n_variable_features 2000, matrix_input pca, pca_components 100

## hard_infonce (comparison)

Tests the soft-target decision (near_twin_fraction 0.232); identical to primary otherwise.

- Strategy: contrastive_alignment, fusion joint
- Losses: infonce x 1.0, cross_modal_regression x 1.0
- Head: embedding 256, hidden 512, 2 layers, dropout 0.2
- Training: batch 512, lr 0.001, weight decay 0.001, up to 300 epochs, patience 15, cap 135s per fit, temperature 0.07 (learned)
- image: phikon, frozen, patch_context 1.0, colour_normalisation reinhard, text_cleaning basic
- expression: mlp, from_scratch, text_cleaning basic, n_variable_features 2000, matrix_input pca, pca_components 100

## patch_context_2x (comparison)

Tests the patch-size decision: 2 x 145 px = 290 px, about the centre-to-centre spacing (273 px), so it adds surrounding tissue context without reaching far into neighbouring spots.

- Strategy: contrastive_alignment, fusion joint
- Losses: soft_infonce x 1.0, cross_modal_regression x 1.0
- Head: embedding 256, hidden 512, 2 layers, dropout 0.2
- Training: batch 512, lr 0.001, weight decay 0.001, up to 300 epochs, patience 15, cap 135s per fit, temperature 0.07 (learned)
- image: phikon, frozen, patch_context 2.0, colour_normalisation reinhard, text_cleaning basic
- expression: mlp, from_scratch, text_cleaning basic, n_variable_features 2000, matrix_input pca, pca_components 100
