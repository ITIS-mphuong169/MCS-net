> **DRAFT STATUS (delete this box before submission)**
> Numbers below are measured unless marked **[TBD]**. Open items:
> 1. **[TBD]** `detail` run: test Acc/F1 (Table 1, Table 4), McNemar vs `single`, learned view weight.
> 1b. **[TBD]** `soft` run (artist-informed label smoothing, Section 3.5): test numbers, McNemar vs `single` (Table 5).
> 1c. **[TBD]** WiSE-FT curve (Section 3.6): chosen alpha, val/test accuracy per alpha, McNemar vs `single` (Table 5, Figure).
> 2. **[TBD]** zoom control run (optional, `python server\clip.py train zoom`). Without it, Section 5.4 can only claim "more views", not "new pixels".
> 3. **[TBD]** test number of the 100-epoch ResNet-101 baseline (only validation 67.04 is known).
> 4. **[TBD]** confusion matrix (from `predictions.npz` written by `eval`), second seed for `single`.
> 5. References must be checked against the originals before submission.
> 6. Protocol of the short-schedule ResNet-101 ablation (Table 3): confirm initialisation and wording against `RUNS_TRACKER.csv`.

# What Matters for Art Style Classification? A Controlled Study of Backbone, Fine-Grained Attention and High-Resolution Detail Views on WikiArt

**Authors:** [TBD]
**Target venue:** PLOS ONE (the venue of the MCS-Net paper we compare against) or a similar journal that accepts replication and negative results.

## Abstract

Recent work on art style classification adds fine-grained components to a convolutional backbone: attention-guided crops, contrastive and counterfactual attention, part-relation modules. We ask how much of the reported progress survives when the backbone is replaced by a vision-language model. Fine-tuning a single CLIP ViT-B/16 with a text-anchored cosine classifier reaches 75.86% accuracy and 74.99% macro-F1 on the 27-class WikiArt test set (8,144 images), against 68.62% and 70.17% reported for MCS-Net, a ResNet-101 model with several dedicated modules. The two numbers come from different random splits of the same dataset, so the comparison is indicative, not paired. On top of the CLIP backbone we add, one at a time, MCS-Net's attention and contrastive modules, a multi-head part-relation module, a REINFORCE crop selector and a label-query attention head. None gives a gain beyond seed noise; the learned gate of the label-query head stays at 0.001 after 100 epochs. The one idea we found that was not already refuted is to give the network more pixels: the attention-crop designs we tested cut a region from the 224 px input and enlarge it, whereas the WikiArt originals we measured have a median short side of about 1,400 px. We test a global view plus high-resolution detail crops. **[TBD: result of the detail run and McNemar test.]** We also show that label ambiguity limits the task: an artist's most frequent style predicts the style of one of their works only 42.6% of the time, and top-5 accuracy is about 98% for a short fine-tuning schedule. Our results suggest that, for this dataset, the choice and scale of the pretrained backbone matters more than fine-grained architecture.

**Keywords:** art style classification, WikiArt, CLIP, fine-grained attention, negative results.

## 1. Introduction

Art style classification assigns a painting to a movement such as Impressionism or Baroque. The task is fine-grained in an unusual way: classes differ by brushwork, palette, composition and period, not by object parts, and the boundaries between neighbouring movements are blurred by the artists themselves. A line of work therefore borrows tools from fine-grained recognition. MCS-Net [Wang et al., 2026] is a recent example. It combines a ResNet-101 with an attention generation module, a contrastive learning module and a counterfactual attention module, and reports 68.62% accuracy on WikiArt.

These designs were developed on convolutional backbones pretrained on ImageNet. Vision-language models such as CLIP [Radford et al., 2021] have since become the default starting point, and they have seen far more paintings than ImageNet models. We ask a plain question: if the backbone is replaced by a fine-tuned CLIP, do the fine-grained components still help?

We answer with controlled experiments on a single dataset, one test set and one training recipe. Our contributions:

1. **A strong simple baseline.** Fine-tuned CLIP ViT-B/16 with a text-anchored cosine classifier: 75.86% accuracy, 74.99% macro-F1, 94.88% top-5 on the 8,144-image WikiArt test set. This is +7.2 accuracy and +4.8 F1 over the published MCS-Net numbers (different split, Section 4.1).
2. **Controlled negative results.** Attention-guided crops, contrastive and counterfactual modules, multi-head and low-rank part-relation modules, REINFORCE crop selection and a label-query attention head all fail to beat the plain model beyond seed variation (Sections 5.2 and 5.3).
3. **A test of the one untried idea: more pixels.** Earlier "look closer" methods cut from the downsized input and add no information. We feed high-resolution crops from the original file and compare against the global view and a zoom control. **[TBD]**
4. **Analysis of what limits accuracy:** the backbone and its scale, label ambiguity, and overconfidence under long fine-tuning (Section 6).

## 2. Related work

**Art style classification.** Early deep-learning work fine-tuned ImageNet CNNs on WikiArt [Saleh and Elgammal, 2015; Tan et al., 2016]. Pandora [Florea et al., 2016] adds a second dataset of 18 art movements. MCS-Net [Wang et al., 2026] reports the strongest published WikiArt accuracy we are aware of for a ResNet-101 pipeline.

**Fine-grained recognition modules.** WS-DAN [Hu et al., 2019] generates attention maps and crops the most attended region for a second pass. Counterfactual attention learning [Rao et al., 2021] rewards attention that changes the prediction. MCS-Net combines both ideas with a contrastive loss. REINFORCE [Williams, 1992] has been used to choose where to look; we use it to choose crop positions.

**Vision-language backbones.** CLIP [Radford et al., 2021] aligns images and text. Its text encoder can supply classifier weights from class names, which we use to initialise and anchor the classifier.

## 3. Method

### 3.1 Pipeline

```
 original painting file (median short side about 1,400 px)
   |
   |-- global view : resize short side to 256, centre crop 224
   |
   '-- detail views: resize short side to 512, centre 448x448 square,
                     train: 2 random 224 crops + random flip
                     test : 3x3 grid of 224 crops (9 views)
   |
   v
 CLIP ViT-B/16 image encoder (one shared copy, fine-tuned, lr x 0.01)
   |
   v   per view: CLS embedding -> L2 norm
 text-anchored cosine classifier: logit_k = s * cos(f, t_k), t_k = CLIP text
   embedding of class name k (4 prompt templates averaged), s learnable <= 100
   |
   |  (ablation) label-query multi-head attention over patch tokens,
   |  added to the logit through a gate initialised at 0
   v
 fusion: z = w * z_global + (1 - w) * mean_j z_detail_j,  w = sigmoid(v), v learned, init 0.5
 loss  : CE(z) + 0.5 * mean over views of CE(z_view), label smoothing 0.1
```

The `single` model uses only the global view and the first term of the loss. Everything else (backbone, classifier, optimiser, schedule, augmentation) is identical between `single` and `detail`.

### 3.2 Text-anchored classifier

The 27 class weights are the CLIP text embeddings of "a painting in the style of {name}.", "an artwork in the {name} style.", "a {name} painting." and "a photo of a {name} artwork.", averaged and normalised. Weights are trainable. At initialisation the model equals CLIP zero-shot classification; we verified that our logits match the Hugging Face CLIP zero-shot logits (maximum difference below 1e-3 in our test). Image features and weights are L2-normalised and scaled by a learnable temperature capped at 100. We use the `quickgelu` variant of the timm ViT because the plain variant applies GELU to weights trained with QuickGELU and its embeddings only reach cosine 0.92 with the original.

### 3.3 Label-query attention head (ablation)

To put multi-head attention inside the model we add a head whose queries are the class prototypes (plus a zero-initialised dataset embedding) and whose keys and values are the patch tokens of the last layer: 8 heads, one attention map per class. Its output is added to the CLS cosine logit through a gate initialised at 0, so at the start the model equals the baseline and the gate value measures the head's contribution.

### 3.4 High-resolution detail views

Let $x$ be the original image and $m=2$ the magnification. The detail square is the central $224m \times 224m$ area of $x$ resized to short side $256m$, so it covers the field of view of the global view at $m$ times the linear resolution. Training samples $K=2$ random $224\times224$ crops from it with random flips. Validation uses 2 fixed crops, testing the full $3\times3$ grid (9 crops, centre cell first). The **zoom control** cuts the same crops from the 224 global view enlarged $m$ times, so the field of view and number of views are the same but no new pixels are added. The difference between `detail` and `zoom` isolates the value of pixels the global view does not contain.

### 3.5 Ambiguity-aware label smoothing (proposed)

Uniform label smoothing with $\varepsilon = 0.1$ moves a tenth of the target mass away from the labelled class and spreads it evenly over the other 26 classes. Style labels are not evenly ambiguous: an Impressionist canvas is far more likely to be filed under Post-Impressionism than under Ukiyo-e, and the same painter is routinely filed under several neighbouring movements (Section 6.2). We keep the amount of smoothing identical to uniform label smoothing (the labelled class keeps $1-\varepsilon+\varepsilon/C$ exactly as PyTorch's `label_smoothing` does) and change only where the remaining mass goes. For a training image by artist $a$ with label $y$, the mass on another class $k$ is proportional to the number of other training works by $a$ labelled $k$ (leave-one-out), plus one pseudo-count spread uniformly so that an artist with no other-style work falls back to uniform smoothing. The artist is read from the WikiArt file name (`artist_title.jpg`); only training labels are used, and nothing beyond the image is used at test time. The comparison against the `single` run is therefore controlled: same data, seed, schedule and $\varepsilon$, different smoothing distribution. A language-prior variant that distributes the mass by the CLIP text similarity of the class names is implemented but not run.

### 3.6 Weight-space ensembling with the zero-shot model

Fine-tuning moves the network away from the zero-shot CLIP, whose knowledge of art styles is what made the frozen features strong (Table 2). Following WiSE-FT [Wortsman et al., 2022] we interpolate the fine-tuned weights $\theta_{ft}$ with the zero-shot weights $\theta_{zs}$ (pretrained image tower, text prototypes as classifier), $\theta_\alpha = (1-\alpha)\theta_{zs} + \alpha\theta_{ft}$, choose $\alpha$ on the validation split and report the test accuracy at that $\alpha$. $\alpha=1$ is the plain fine-tuned model and is always a candidate, so the selection cannot hurt on validation. No training is involved.

## 4. Experimental setup

### 4.1 Data and protocol

WikiArt (Kaggle `steubk/wikiart`): 81,444 paintings in 27 style classes. We split it 80/10/10 with a fixed seed (1231): 65,156 training, 8,144 validation, 8,144 test images. MCS-Net used 57,011/16,289/8,144 (about 70/20/10). The test sets have the same size but are **different images**, and our training set is larger. Comparisons with published numbers are therefore indicative only; every comparison among our own models uses the same split, checkpoint rule and test images, and paired tests (McNemar, paired bootstrap) are used where predictions are available.

### 4.2 Metrics

Accuracy, macro precision, macro recall, macro F1 and top-5 accuracy on the test set. Checkpoints: the **last epoch** is the primary result; the best-validation epoch is reported alongside. A binomial 95% interval for an accuracy near 76% on 8,144 images is about $\pm$0.9 points.

### 4.3 Training recipe (all CLIP runs)

AdamW, learning rate $10^{-3}$ for the classifier and $10^{-5}$ for the backbone, weight decay 0.05, cosine schedule with 2 warm-up epochs, batch size 32 (gradient accumulation where memory requires), bfloat16, label smoothing 0.1, stochastic depth 0.1. Training augmentation: RandomResizedCrop 224 and horizontal flip. Test: resize 256, centre crop 224. The main runs last 100 epochs on one RTX 4070 Super (12 GB); shorter runs (8 epochs, T4 GPUs) are used for the module ablations. ResNet-101 runs use the repository's MCS-Net configuration (SGD, 224 px).

## 5. Results

### 5.1 A fine-tuned CLIP beats the published MCS-Net numbers

**Table 1. WikiArt, 27 classes, test set of 8,144 images (macro averages).**

| Model | Acc | Precision | Recall | F1 | Top-5 |
|---|---|---|---|---|---|
| ResNet-101 (published, MCS-Net paper) | 64.11 | | | | |
| CAL (published, MCS-Net paper) | 67.57 | | | | |
| MCS-Net (published) | 68.62 | 69.40 | 72.32 | 70.17 | |
| CLIP B/16, `single`, 100 ep, last epoch (ours) | **75.86** | 77.76 | 73.01 | 74.99 | 94.88 |
| CLIP B/16, `single`, best-val epoch 93 (ours) | 75.88 | 77.45 | 72.52 | 74.55 | 94.94 |
| CLIP B/16, `detail`, 100 ep, last epoch (ours) | **[TBD]** | | | | |

Published rows use the MCS-Net split; our rows use ours. The gain is mostly precision (+8.4); recall is about the same (+0.7). The validation accuracy of the `single` run was 76.20 (best 76.29) against a training accuracy of 96.86, a large generalisation gap (Section 6.3).

### 5.2 The backbone explains most of the difference

**Table 2. Backbone comparison on our split (test Acc / macro-F1).**

| Backbone | Frozen, linear probe | Fine-tuned |
|---|---|---|
| ResNet-101 (ImageNet) | 52.91 / 49.23 | 60.61 / 59.24 (30 ep, our implementation without part-relation module) |
| ConvNeXt-B (ImageNet-22k) | 58.23 / 56.44 | not run |
| CLIP ViT-B/16 | 63.74 / 62.08 | 74.29 / 72.71 (8 ep); 75.86 / 74.99 (100 ep) |

A frozen CLIP already matches a fine-tuned ResNet-101 at 30 epochs. Fine-tuning improves ResNet-101 by 7.7 points and CLIP by 10.5 to 12.1. On a 2,400-image pilot of real WikiArt images, a frozen ViT-L/14 beat ViT-B/16 by +3.96 points (95% bootstrap CI [+2.04, +5.88]), about 2.5 times the gain of detail over the global view (+1.62) and four times the gain of new pixels alone (+0.96) in the same pilot (Section 5.4).

We did not reproduce the published ResNet-101 numbers: our best ResNet-101 variants reach 60 to 62% at 30 epochs, and the 100-epoch ResNet-101 baseline reached 67.04% validation accuracy (test not evaluated). The comparison with 68.62% therefore rests on the published figure. **[TBD: test number of the 100-epoch ResNet-101 baseline, if available.]**

### 5.3 Fine-grained components do not help on a strong backbone

**On CLIP.** Two seeds of the plain 8-epoch CLIP model give 74.29 / 72.71 and 73.94 / 71.83 (Acc / F1), mean 74.12 $\pm$ 0.25. The same backbone with the complete MCS-Net (AGM, SCLM, CCAM) and a rank-64 part-relation module (MLA) reaches 73.64 / 71.77 (single seed), inside the seed range. A two-view variant that adds the attention crop gives 73.78 / 71.70. The label-query attention head of Section 3.3 does not contribute either: its gate is 0.0011 at the end of the 100-epoch run, i.e. the head is effectively switched off.

**On ResNet-101, short schedule.** Because the conclusions on CLIP could be an artefact of the backbone, we also ran every component on ResNet-101 in a cheap protocol (25% of the training data, 8 epochs, 3 seeds).

**Table 3. Short-schedule ResNet-101 ablation, test Acc / macro-F1, mean over 3 seeds ($\pm$ sample SD of Acc).**

| Variant | Acc | F1 |
|---|---|---|
| Baseline (plain ResNet-101) | 44.02 $\pm$ 1.97 | 37.92 |
| + multi-head part relation (MHA) | 46.74 $\pm$ 0.78 | 32.04 |
| + low-rank part relation (MLA-64) | 47.42 $\pm$ 0.60 | 35.17 |
| + MLA-64 + REINFORCE crop selection (v1) | 45.72 $\pm$ 1.47 | 31.42 |

Part-relation modules raise accuracy by 2.7 to 3.4 points over the baseline but lower macro-F1: rare classes collapse in 8 epochs. MLA-64 vs MHA is not significant (accuracy +0.68, paired $t$-test $p \approx 0.31$; F1 +3.13 in 3/3 seeds, $p \approx 0.02$ with $n=3$). The REINFORCE selector is worse than plain MLA in all three seeds, and its policy entropy stayed at 3.40 to 3.45 against the maximum $\ln 32 = 3.47$, so it barely learned. A second version that used the policy at test time and rewarded the fused prediction also gave nothing in one seed: centre 47.65 vs its control 48.17, and policy, mean-attention and sampled crops 47.43, 47.59, 47.34. In 2-epoch fine-tunes started from a self-attention checkpoint, the MLA vs MHA difference did not replicate across seeds (seed 1: McNemar $p = 0.0008$; seed 2: $p = 0.72$; the seed-1 comparison was confounded by data-augmentation replay).

Taken together, with one exception that we flag as unreplicated, the evidence is that attention-based part relations and learned crop selection add nothing measurable once seed variance is accounted for, on either backbone.

### 5.4 High-resolution detail views

**Preliminary frozen-feature study.** On 2,400 real WikiArt images (CLIP ViT-B/16 frozen, logistic regression, 5-fold cross-validation):

| Input | Acc |
|---|---|
| global view | 53.83 |
| + zoom crops (no new pixels) | 54.50 |
| + detail crops, $m=2$ (2x2 grid) | 55.46 |

Detail minus global is +1.62 (95% bootstrap CI [+0.12, +3.17]); detail minus zoom, the effect of new pixels alone, is +0.96 (CI [-0.46, +2.38]). So detail is a small lever, and in this pilot not significantly separable from zoom.

**Full fine-tuning (100 epochs).** **[TBD: fill Table 4 from `summary_clip.txt`.]**

**Table 4. `single` vs `detail`, test set, last epoch.**

| View used at test | Acc | F1 | Top-5 |
|---|---|---|---|
| `single` model | 75.86 | 74.99 | 94.88 |
| `detail` model, fused prediction | [TBD] | [TBD] | [TBD] |
| `detail` model, global view only | [TBD] | [TBD] | [TBD] |
| `detail` model, 9 detail crops only | [TBD] | [TBD] | [TBD] |

McNemar test `detail` vs `single`: **[TBD]**. Learned weight of the global view $w$ at epoch 100: **[TBD]** (0.5 at initialisation; a value near 1 would mean the network ignores the detail crops).

### 5.5 Ambiguity-aware smoothing and weight-space ensembling

**[TBD: fill from `summary_clip.txt` after `train soft`, `eval soft` and `eval wise`.]**

**Table 5. Training-target and weight-space variants of the `single` model, test set, last epoch.**

| Model | Acc | F1 | Top-5 | McNemar vs `single` |
|---|---|---|---|---|
| `single` (uniform label smoothing 0.1) | 75.86 | 74.99 | 94.88 | |
| artist-informed label smoothing 0.1 (Section 3.5) | [TBD] | [TBD] | [TBD] | [TBD] |
| WiSE-FT, $\alpha$ = [TBD] chosen on val (Section 3.6) | [TBD] | [TBD] | [TBD] | [TBD] |
| CLIP zero-shot ($\alpha$ = 0) | [TBD] | [TBD] | [TBD] | |

Artist statistics of the training split (printed by the loader): **[TBD]** artists, **[TBD]**% of training images with an artist-informed target, mean smoothing entropy **[TBD]** bits against $\log_2 26 = 4.70$ for uniform smoothing.

## 6. Analysis and discussion

### 6.1 Per-class behaviour

The best classes are Ukiyo-e (F1 0.98), Northern Renaissance (0.91), Early Renaissance (0.86), Rococo (0.85) and Baroque (0.84): distinctive subjects, palettes or techniques. The weakest are Fauvism (0.43), the 9-image Action painting class (0.63), Post-Impressionism (0.65), Expressionism (0.68), Cubism (0.69) and Symbolism (0.69): movements that overlap with their neighbours. **[TBD: confusion matrix and the three most frequent confusions, from `predictions.npz`.]**

### 6.2 Label ambiguity

Style is not a function of the artist. On 9,049 WikiArt images, predicting each work's style by the most frequent style of its other works gives only 42.6% accuracy (leave-one-out), and 76% of the images belong to artists whose main style covers less than 70% of their work. Top-5 accuracy of the 8-epoch CLIP model is 98.2%, so errors are mostly between neighbouring movements. Together these point to a ceiling set by label noise and ambiguity, not by model capacity alone.

### 6.3 Overconfidence under long fine-tuning

Going from 8 to 100 epochs (with the extra regularisation of the final recipe) raises top-1 accuracy from about 74.1 to 75.9 but lowers top-5 from 98.2 to 94.9, with training accuracy at 96.9. The model becomes sharper and more wrong on its failures. We did not evaluate calibration or weight-space ensembling and leave them as the obvious next step.

### 6.4 Limitations

- **Splits differ** from MCS-Net, so the 7.2-point difference to the published figure is not a paired comparison.
- **We did not reproduce 68.62%** with the ResNet-101 pipeline; the gap to the published ResNet-101 baseline (64.11) was not closed in our runs.
- **One seed** for the 100-epoch runs, and one seed for several module ablations. Seed variation of about 2 points was seen in short schedules, so differences below that need paired tests.
- **Short-schedule ablations** (Table 3) use 25% of the data and 8 epochs; the conclusions may differ at full scale. The CLIP-based ablations use 8 epochs, single seed for the full MCS-Net variant.
- **Single dataset.** A second dataset (Pandora, 18 movements, which reuses WikiArt images and needs de-duplication) was prepared but not evaluated.
- The pilot results of Section 5.4 use a 2,400-image subset and frozen features.

## 7. Conclusion

On WikiArt, a fine-tuned CLIP ViT-B/16 with a text-anchored classifier reaches 75.86% accuracy and 74.99% macro-F1, above the published ResNet-101 MCS-Net result, and none of the fine-grained components we tested (attention crops, contrastive and counterfactual modules, part-relation attention, REINFORCE selection, label-query attention) gives a gain beyond seed noise on top of it. In this setting, the backbone and its scale matter more than architecture on top. **[TBD: one sentence on the detail views once the run is done.]** We release code and recipes so that other fine-grained modules can be tested against the same baseline.

## Reproducibility

Code: the repository branch `exp/regularize` (`server/clip.py` runs the 100-epoch `single` and `detail` runs; `infer.py` and `mcnemar.py` produce the test numbers). W&B project `mcs-net-wikiart`, runs tagged `srv_clip_single` and `srv_clip_detail`. The data split is determined by seed 1231.

## References (verify each before submission)

- Wang, Xia, Yang, Zou, Zhou. An art style classification network integrating contrastive learning and counterfactual attention. PLOS ONE 21(6): e0343954, 2026. doi:10.1371/journal.pone.0343954
- Radford et al. Learning transferable visual models from natural language supervision. ICML 2021.
- Dosovitskiy et al. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR 2021.
- Hu, Chen, Qi, Zheng, Hua, Bao. See better before looking closer: weakly supervised data augmentation network for fine-grained visual classification. arXiv 2019.
- Rao, Xu, Liu, Zhang, Chen, Wang. Counterfactual attention learning for fine-grained visual categorization and re-identification. ICCV 2021.
- Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning 8, 1992.
- Saleh, Elgammal. Large-scale classification of fine-art paintings: learning the right metric on the right feature. arXiv 2015.
- Tan, Chan, Aguirre, Tanaka. Ceci n'est pas une pipe: a deep convolutional network for fine-art paintings classification. ICIP 2016.
- Florea et al. Pandora: description of a painting database for art movement identification (check exact title and venue). 2016.
- Loshchilov, Hutter. Decoupled weight decay regularization. ICLR 2019.
- Wortsman et al. Robust fine-tuning of zero-shot models. CVPR 2022.
- Szegedy et al. Rethinking the Inception architecture for computer vision. CVPR 2016. (label smoothing)
- Müller, Kornblith, Hinton. When does label smoothing help? NeurIPS 2019.
- McNemar. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika 12, 1947.
