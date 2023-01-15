
_MODALITIES = ['rgb', 'spectrogram']

class EncoderBlock(nn.Module):
  """Transformer encoder block.
  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of heads.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    attention_kernel_initializer: Initializer to use for attention
      layers.
    droplayer_p: Probability of dropping a layer.
  Returns:
    Output after transformer encoder block.
  """
  mlp_dim: int
  num_heads: int
  dtype: torch.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  attention_kernel_initializer: nn.init.xavier_uniform_
  droplayer_p: float = 0.0

  '''
  #  層のDropout→データセットが小さい時ほど有効っぽいからとりあえずなしにしている
  def get_drop_pattern(self, x, deterministic):
    if not deterministic and self.droplayer_p:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
          self.make_rng('dropout'), self.droplayer_p, shape).astype('float32')
    else:
      return 0.0
  '''

  @nn.compact
  def __call__(self, inputs: torch.Tensor, deterministic: bool) -> torch.Tensor:
    """Applies Encoder1DBlock module."""

    # Attention block.
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=self.attention_kernel_initializer,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        dtype=self.dtype)(x, x, deterministic=deterministic)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)

    drop_pattern = self.get_drop_pattern(x, deterministic)
    x = x * (1.0 - drop_pattern) + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = attention_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(
            y, deterministic=deterministic)

    drop_pattern = self.get_drop_pattern(x, deterministic)
    return y * (1.0 - drop_pattern) + x


class Encoder(nn.Module):
  """Transformer Encoder.
  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of attention heads.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_droplayer_rate: Probability of dropping a layer linearly
      grows from 0 to the provided value. Our implementation of stochastic
      depth follows timm library, which does per-example layer dropping and
      uses independent dropping patterns for each skip-connection.
    modality_fusion: Tuple with modalities to combine.
    fusion_layer: Which layer to fuse modalities. fusion_layer == 0 provides
      early fusion.
    use_bottleneck: If True, adds self-attention bottleneck.
    test_with_bottlenecks: Whether to use bottlenecks at test time.
    share_encoder: If True, different modalities share the same encoder weights
      for the layers before fusion.
    dtype: The dtype of the computation (default: float32).
  """
  mlp_dim: int
  num_layers: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_droplayer_rate: float = 0.0
  modality_fusion: Tuple[str] = ('spectrogram',)
  fusion_layer: int = 0
  use_bottleneck: bool = False
  test_with_bottlenecks: bool = True
  share_encoder: bool = False
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: Dict[str, Any],
               bottleneck: jnp.ndarray, *,
               train: bool):
    """Applies Transformer model on the inputs."""

    def get_encoder_block(encoder_block, droplayer_p, name):
      """Returns the encoder block for a single layer."""
      dtype = jax.dtypes.canonicalize_dtype(self.dtype)
      return encoder_block(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          droplayer_p=droplayer_p,
          name=name,
          dtype=dtype)

    def get_context(target_modality, modality_fusion, x):
      """Returns list of context modalities."""
      context = []
      for modality in _MODALITIES:
        if modality != target_modality and modality in modality_fusion:
          context.append(x[modality])
      return context

    def combine_context(x, other_modalities):
      """Combine x with a list of other modalities."""
      num_tokens = x.shape[1]
      # Append x to the end of the list
      other_modalities.append(x)
      x_combined = jnp.concatenate(other_modalities, axis=1)
      return x_combined, num_tokens

    assert self.modality_fusion

    # Add positional embeddings
    for modality in self.modality_fusion:
      if modality == 'rgb':
        name = ''
      else:
        name = '_' + modality
      x[modality] = add_positional_embed(x[modality], 'posembed_input' + name)

    use_bottlenecks = train or self.test_with_bottlenecks
    x_combined = None
    # Input Encoder
    for lyr in range(self.num_layers):
      droplayer_p = (
          lyr / max(self.num_layers - 1, 1)) * self.stochastic_droplayer_rate
      encoders = {}
      encoders['rgb'] = get_encoder_block(EncoderBlock, droplayer_p,
                                          f'encoderblock_{lyr}')

      for modality in self.modality_fusion:
        if modality != 'rgb':
          if self.share_encoder:
            encoders[modality] = encoders['rgb']
          else:
            encoders[modality] = get_encoder_block(
                EncoderBlock, droplayer_p,
                f'encoderblock_{lyr}_' + modality)

      if (lyr < self.fusion_layer or len(self.modality_fusion) == 1 or
          (self.use_bottleneck and not use_bottlenecks)):
        for modality in self.modality_fusion:
          x[modality] = encoders[modality](x[modality], deterministic=not train)
      else:
        if self.use_bottleneck:
          bottle = []
          for modality in self.modality_fusion:
            t_mod = x[modality].shape[1]
            in_mod = jnp.concatenate([x[modality], bottleneck], axis=1)
            out_mod = encoders[modality](in_mod, deterministic=not train)
            x[modality] = out_mod[:, :t_mod]
            bottle.append(out_mod[:, t_mod:])
          bottleneck = jnp.mean(jnp.stack(bottle, axis=-1), axis=-1)
        else:
          if not self.share_encoder and len(self.modality_fusion) > 1:
            x_new = {}
            for modality in self.modality_fusion:
              other_modalities = get_context(modality, self.modality_fusion, x)
              combined_mods, t = combine_context(x[modality], other_modalities)
              combined_mods = encoders[modality](
                  combined_mods, deterministic=not train)
              x_new[modality] = combined_mods[:, -t:]
            x = x_new

          elif self.share_encoder and len(self.modality_fusion) > 1:
            if x_combined is None:
              x_combined = []
              for modality in self.modality_fusion:
                x_combined.append(x[modality])
              x_combined = jnp.concatenate(x_combined, axis=1)
            x_combined = encoders['rgb'](x_combined, deterministic=not train)
    if x_combined is not None:
      x_out = x_combined
    else:
      x_out = []
      for modality in self.modality_fusion:
        x_out.append(x[modality])
      x_out = jnp.concatenate(x_out, axis=1)
    encoded = nn.LayerNorm(name='encoder_norm')(x_out)

    return encoded