import importlib.util


def load_config(path):
    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.config


def reshape_transform(result, height=14, width=14, timesteps=30):
    # Result shape: [batch* time, 196, embedding_dim]
    BT, N, C = result.shape
    print("Tensor Shape:", result.shape)
    result = result.unsqueeze(0)
    result = result.reshape(BT // timesteps, timesteps, height, width, C)

    # Transpose dimensions to get [batch, embedding_dim, time, height, width].
    result = result.permute(0, 4, 1, 2, 3)
    print("Reshaped Tensor Shape:", result.shape)
    return result
