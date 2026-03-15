import importlib.util


def load_config(path):
    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.config


def reshape_transform(result, height=14, width=14, timesteps=30):
    # Result shape is typically [batch*time, 196, embedding_dim] for video
    # and [batch, 196, embedding_dim] for image-only SigLIP.
    BT, N, C = result.shape
    print("Tensor Shape:", result.shape)

    if N != height * width:
        raise ValueError(
            f"Unexpected token count {N}; expected {height * width} for {height}x{width} patches."
        )

    effective_timesteps = timesteps if BT >= timesteps and BT % timesteps == 0 else 1
    batch_size = BT // effective_timesteps

    result = result.reshape(batch_size, effective_timesteps, height, width, C)

    # Transpose dimensions to [batch, embedding_dim, time, height, width].
    result = result.permute(0, 4, 1, 2, 3)
    print("Reshaped Tensor Shape:", result.shape)
    return result
