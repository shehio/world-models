"""K-step unroll shape tests.

  - initial_inference output shapes match (B, latent_channels, 8, 8) for s_0
  - recurrent_inference: s_k shape preserved over k steps
  - reward / value scalars are (B,)
  - policy logits are (B, 4672)
  - gradient flow: backward through K=5 unroll doesn't blow up

NOT YET IMPLEMENTED.
"""
