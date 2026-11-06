"""Text-anchored classifier for CLIP backbones (TEXT_INIT=1).

CLIP's image tower was trained to match a text tower in a shared 512-d
space, so the style names themselves give a classifier before any training:
the prototype of class c is the (prompt-ensembled) text embedding of its
name, and logits are scaled cosine similarities (CLIP's zero-shot head).
Here that head is the *initialisation* of the classifier, fine-tuned with
the rest of the model, instead of a randomly initialised Linear layer. A
class with 98 training images (Action_painting) starts from what CLIP
already knows about the words "action painting" rather than from noise.
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# WikiArt folder name -> how the style is actually called
STYLE_NAMES = {
    'Action_painting': 'Action painting',
    'Art_Nouveau_Modern': 'Art Nouveau',
    'Color_Field_Painting': 'Color Field painting',
    'Mannerism_Late_Renaissance': 'Mannerism (Late Renaissance)',
    'Naive_Art_Primitivism': 'Naive Art (Primitivism)',
    'Ukiyo_e': 'Ukiyo-e',
}
TEMPLATES = [
    'a painting in the style of {}.',
    'an artwork in the {} style.',
    'a {} painting.',
    'a photo of a {} artwork.',
]
# timm image tower -> HF checkpoint with the matching text tower. Use the
# '*_clip_quickgelu_*' timm variants: the plain '*_clip_*' ones load the
# OpenAI weights but run GELU instead of OpenAI's QuickGELU, and their image
# embedding then only has cosine ~0.92 with the HF/OpenAI one (checked on
# ViT-B/32), so the text prototypes would sit in a slightly different space.
# The quickgelu variants reproduce HF's image embedding exactly.
TEXT_MODELS = {
    'vit_base_patch16_clip_quickgelu_224.openai': 'openai/clip-vit-base-patch16',
    'vit_base_patch32_clip_quickgelu_224.openai': 'openai/clip-vit-base-patch32',
    'vit_large_patch14_clip_quickgelu_224.openai': 'openai/clip-vit-large-patch14',
}


def style_name(folder):
    folder = folder.split('|')[0]  # LABEL_SPACE=separate names are '<style>|<dataset>'
    return STYLE_NAMES.get(folder, folder.replace('_', ' '))


@torch.no_grad()
def text_prototypes(class_names, net):
    """(num_classes, embed_dim) L2-normalised text embeddings, averaged over
    the prompt templates. TEXT_MODEL overrides the checkpoint."""
    from transformers import CLIPTokenizer, CLIPTextModelWithProjection
    model_id = os.environ.get('TEXT_MODEL', TEXT_MODELS.get(net))
    assert model_id, 'no text tower known for %s (use a *_clip_quickgelu_* model, or set TEXT_MODEL)' % net
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    text_model = CLIPTextModelWithProjection.from_pretrained(model_id).eval()
    protos = []
    for c in class_names:
        prompts = [t.format(style_name(c)) for t in TEMPLATES]
        emb = text_model(**tokenizer(prompts, padding=True, return_tensors='pt')).text_embeds
        protos.append(F.normalize(F.normalize(emb, dim=-1).mean(0), dim=0))
    print('text prototypes from {}: {} classes x {} dims'.format(model_id, len(protos), protos[0].numel()))
    return torch.stack(protos)


class CosineClassifier(nn.Module):
    """logits = s * cos(x, w_c). Prototypes w_c start from the text
    embeddings (or random directions without them) and are fine-tuned
    unless TEXT_FREEZE=1. s starts at CLIP's logit scale (100) and is
    learned, capped at 100 like CLIP's."""

    def __init__(self, in_dim, num_classes, prototypes=None, scale=100.):
        super(CosineClassifier, self).__init__()
        if prototypes is None:
            prototypes = F.normalize(torch.randn(num_classes, in_dim), dim=-1)
        assert prototypes.shape == (num_classes, in_dim), prototypes.shape
        self.weight = nn.Parameter(prototypes.clone().float(), requires_grad=os.environ.get('TEXT_FREEZE', '0') == '0')
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)))

    def forward(self, x):
        scale = self.log_scale.exp().clamp(max=100.)
        return scale * F.normalize(x.float(), dim=-1) @ F.normalize(self.weight, dim=-1).t()

    def pairwise(self, x):
        """x: (B, C, D), one feature per class -> (B, C) logits, feature c
        scored against prototype c only."""
        scale = self.log_scale.exp().clamp(max=100.)
        return scale * (F.normalize(x.float(), dim=-1) * F.normalize(self.weight, dim=-1)).sum(-1)


class LabelQueryAttention(nn.Module):
    """Label-query multi-head attention head (LABEL_ATTN=1).

    The cosine classifier sees one global vector per image (the CLS token).
    Here every class also looks at the image itself: its text prototype is
    the query of a multi-head cross-attention over the 14x14 patch tokens
    (keys / values), so each style pools the patches that are evidence for
    it (label queries as in Query2Label / ML-Decoder, but with the queries
    anchored to the CLIP text embeddings of the style names). The attended
    feature o_c of class c is scored against class c's own prototype and
    added to the CLS logit through a gate that starts at zero:

        logit_c = s * cos(z, w_c) + gate * s * cos(o_c, w_c)

    so training starts exactly from the text-anchored classifier and the
    attention branch contributes only what it learns (the gate is logged).
    The query gets no residual path to the output: o_c must come from the
    image, otherwise cos(o_c, w_c) would be high for every image.

    Multi-dataset: a learned embedding of the sample's dataset (zero-init)
    is added to the queries, so the same style name can be looked for
    differently in WikiArt and in Pandora18k - dataset-aware queries, the
    per-dataset expert of Erudite at the cost of one vector per dataset.
    """

    def __init__(self, patch_dim, embed_dim, num_datasets=1, heads=8, proj_init=None):
        super(LabelQueryAttention, self).__init__()
        # patch tokens -> joint image-text space; starts as CLIP's own visual
        # projection (the one applied to the CLS token)
        self.patch_proj = nn.Linear(patch_dim, embed_dim, bias=False)
        if proj_init is not None and proj_init.shape == self.patch_proj.weight.shape:
            with torch.no_grad():
                self.patch_proj.weight.copy_(proj_init)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.ds_embed = nn.Embedding(max(num_datasets, 1), embed_dim)
        nn.init.zeros_(self.ds_embed.weight)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, patches, prototypes, ds=None, need_weights=False):
        """patches (B, N, patch_dim), prototypes (C, embed_dim), ds (B,) dataset
        index or None -> attended class features (B, C, embed_dim) and, with
        need_weights, the head-averaged attention (B, C, N)."""
        B = patches.size(0)
        keys = self.patch_norm(self.patch_proj(patches))
        queries = F.normalize(prototypes, dim=-1).to(keys.dtype).unsqueeze(0).expand(B, -1, -1)
        if ds is None:
            ds = torch.zeros(B, dtype=torch.long, device=patches.device)
        queries = self.query_norm(queries + self.ds_embed(ds).to(keys.dtype).unsqueeze(1))
        out, weights = self.attn(queries, keys, keys, need_weights=need_weights)
        return out, weights
