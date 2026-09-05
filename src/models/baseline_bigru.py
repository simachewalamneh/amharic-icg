"""
Baseline reproduction: Inception-v3 CNN encoder + visual attention +
GRU attention decoder, matching Solomon & Abebe (2023).

IMPORTANT FIX: an earlier version of this file added a bidirectional GRU
("context_bigru") over the full caption sequence to more literally match
the paper's "Bi-GRU" description. That was a bug: the backward direction of
a bidirectional RNN necessarily has access to FUTURE tokens (including the
exact token being predicted at each step), which is data leakage under
teacher forcing. It produced deceptively low train/val loss that would NOT
reflect real generation quality (no future tokens exist at actual inference
time). This version removes that component. The decoder is now a standard
causal, step-by-step GRU decoder with visual attention.

Target to beat (reported in the paper, Flickr8k+BNATURE combined):
  BLEU-1: 60.6   BLEU-2: 50.1   BLEU-3: 43.7   BLEU-4: 38.8
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3


class InceptionEncoder(nn.Module):
    def __init__(self, fine_tune: bool = False):
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        inception = inception_v3(weights=weights, aux_logits=True)
        inception.aux_logits = False
        inception.AuxLogits = None
        self.features = nn.Sequential(*list(inception.children())[:-3])
        if not fine_tune:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.features(images)
        feats = feats.flatten(2).permute(0, 2, 1)
        return feats


class VisualAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attn_dim: int):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attn_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attn_dim)
        self.full_att = nn.Linear(attn_dim, 1)

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor):
        att1 = self.encoder_proj(encoder_out)
        att2 = self.decoder_proj(decoder_hidden).unsqueeze(1)
        att = self.full_att(torch.tanh(att1 + att2)).squeeze(2)
        alpha = F.softmax(att, dim=1)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class GRUAttentionDecoder(nn.Module):
    """Causal, step-by-step GRU decoder with visual attention — never sees future tokens."""

    def __init__(self, vocab_size, embed_dim=256, encoder_dim=2048,
                 decoder_dim=512, attn_dim=256, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.attention = VisualAttention(encoder_dim, decoder_dim, attn_dim)
        self.gru_cell = nn.GRUCell(embed_dim + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.fc_out = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def init_hidden(self, encoder_out: torch.Tensor) -> torch.Tensor:
        mean_feats = encoder_out.mean(dim=1)
        return torch.tanh(self.init_h(mean_feats))

    def forward(self, encoder_out: torch.Tensor, captions: torch.Tensor):
        batch_size, T = captions.shape
        device = captions.device
        embeddings = self.embedding(captions)
        h = self.init_hidden(encoder_out)
        outputs = torch.zeros(batch_size, T - 1, self.fc_out.out_features, device=device)

        for t in range(T - 1):
            word_embed = embeddings[:, t, :]  # only token t
            visual_context, _ = self.attention(encoder_out, h)
            gru_input = torch.cat([word_embed, visual_context], dim=1)
            h = self.gru_cell(gru_input, h)
            h = self.dropout(h)
            outputs[:, t, :] = self.fc_out(h)

        return outputs

    @torch.no_grad()
    def generate(self, encoder_out: torch.Tensor, sos_id: int, eos_id: int, max_len: int = 30):
        """Real autoregressive generation — used for actual BLEU evaluation."""
        batch_size = encoder_out.size(0)
        device = encoder_out.device
        h = self.init_hidden(encoder_out)
        current_token = torch.full((batch_size,), sos_id, dtype=torch.long, device=device)
        generated = [current_token]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            word_embed = self.embedding(current_token)
            visual_context, _ = self.attention(encoder_out, h)
            gru_input = torch.cat([word_embed, visual_context], dim=1)
            h = self.gru_cell(gru_input, h)
            logits = self.fc_out(h)
            current_token = logits.argmax(dim=1)
            generated.append(current_token)
            finished = finished | (current_token == eos_id)
            if finished.all():
                break

        return torch.stack(generated, dim=1)


class BiGRUBaselineModel(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int = 0, fine_tune_encoder: bool = False):
        super().__init__()
        self.encoder = InceptionEncoder(fine_tune=fine_tune_encoder)
        self.decoder = GRUAttentionDecoder(vocab_size=vocab_size, pad_id=pad_id)

    def forward(self, images: torch.Tensor, captions: torch.Tensor):
        encoder_out = self.encoder(images)
        return self.decoder(encoder_out, captions)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, sos_id: int, eos_id: int, max_len: int = 30):
        encoder_out = self.encoder(images)
        return self.decoder.generate(encoder_out, sos_id, eos_id, max_len)


if __name__ == "__main__":
    vocab_size = 7984
    model = BiGRUBaselineModel(vocab_size=vocab_size, pad_id=0)
    dummy_images = torch.randn(4, 3, 299, 299)
    dummy_captions = torch.randint(0, vocab_size, (4, 17))

    logits = model(dummy_images, dummy_captions)
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (4, 16, vocab_size)
    print("Shape check passed.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_params:,} / Total params: {n_total:,}")