# Plan (llm)

Objective: "Learn a joint image–caption embedding for image–text retrieval and imaging-modality classification."

- Tasks: joint_embedding, retrieval, classification (primary: retrieval)
- Modalities aligned: image + text
- Split: predefined by 'article', 1 fold(s)
- Primary metric: retrieval_mean_recall_at_10; metrics: retrieval_recall_at_k, linear_probe, knn_accuracy, clustering_ari_nmi, retrieval_label_match_at_k, alignment_uniformity, modality_gap, effective_rank
- Baselines: unimodal, chance, cca, early_fusion, late_fusion, zero_shot
- Downstream: retrieval_examples, classification

## primary (primary)

Frozen BiomedCLIP dual encoder (best screening: alignment R@10 0.966, image probe F1 0.934). Residual 512-d heads start from the zero-shot solution. InfoNCE plus a CE term with weight 0.5 on 'modality', because the objective asks for classification.

- Strategy: contrastive_alignment, fusion joint
- Losses: infonce x 1.0, supervised_ce x 0.5
- Head: embedding 512, hidden 512, 2 layers, dropout 0.1
- Training: batch 512, lr 0.001, weight decay 0.0001, up to 300 epochs, patience 15, cap 180s per fit, temperature 0.07 (learned)
- image: biomedclip_image, frozen, greyscale_to_rgb True, text_cleaning basic
- text: biomedclip_text, frozen, text_max_tokens 128, text_cleaning basic

## partial_finetune (comparison)

Tests the frozen-backbone decision. ~20k training pairs and about 2 epochs fit in the 1200 s fine-tuning budget on mps. Batch raised to 128 so the comparison has more negatives per positive than at 64.

- Strategy: contrastive_alignment, fusion joint
- Losses: infonce x 1.0, supervised_ce x 0.5
- Head: embedding 512, hidden 512, 2 layers, dropout 0.1
- Training: batch 128, lr 0.001, weight decay 0.0001, up to 5 epochs, patience 2, cap 1200s per fit, temperature 0.07 (learned)
- image: biomedclip_image, partial_finetune, trainable_blocks 2, greyscale_to_rgb True, augment True, text_cleaning basic
- text: biomedclip_text, partial_finetune, trainable_blocks 2, text_max_tokens 128, text_cleaning basic

## contrastive_only (comparison)

Tests whether the supervised CE term helps classification without costing retrieval recall.

- Strategy: contrastive_alignment, fusion joint
- Losses: infonce x 1.0
- Head: embedding 512, hidden 512, 2 layers, dropout 0.1
- Training: batch 512, lr 0.001, weight decay 0.0001, up to 300 epochs, patience 15, cap 180s per fit, temperature 0.07 (learned)
- image: biomedclip_image, frozen, greyscale_to_rgb True, text_cleaning basic
- text: biomedclip_text, frozen, text_max_tokens 128, text_cleaning basic

## primary_revised (primary)

Critic revision of primary: In all trained configurations, validation R@10 drops right after the first epoch (primary 0.871 → 0.856 → 0.830) while training loss falls steadily (2.34 → 0.60). The same happens in contrastive_only, so the CE term is not the cause. At lr 1e-3, one epoch (about 40 steps at batch 512) moves the residual heads well away from an already well-aligned BiomedCLIP space; this data is likely close to BiomedCLIP's PMC pretraining distribution. A roughly 30× smaller learning r

- Strategy: contrastive_alignment, fusion joint
- Losses: infonce x 1.0, supervised_ce x 0.5
- Head: embedding 512, hidden 512, 2 layers, dropout 0.1
- Training: batch 512, lr 3e-05, weight decay 0.01, up to 300 epochs, patience 15, cap 180s per fit, temperature 0.07 (learned)
- image: biomedclip_image, frozen, greyscale_to_rgb True, text_cleaning basic
- text: biomedclip_text, frozen, text_max_tokens 128, text_cleaning basic
