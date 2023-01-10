import os
import joblib
import yaml


def preprocess_data(dataset, config):
    X = dataset[config['features']]
    y = dataset[[config['outcome']['time'], config['outcome']['event']]].values
    y[:, 1] = 1 - y[:, 1]  # inverse to censored
    return X, y


def save(save_dir, model, c_index, params, data):
    os.makedirs(save_dir, exist_ok=True)

    model['model'].save(os.path.join(save_dir, "model.yaml"))
    if 'selector' in model:
        joblib.dump(model['selector'],  os.path.join(save_dir, "selector.pkl"))
    joblib.dump(data, os.path.join(save_dir, "data.pkl"))
    metadata = {
        'c_index': float(c_index),
        'params': dict(params),
    }
    with open(os.path.join(save_dir, "params.yaml"), 'w') as f:
        yaml.dump(metadata, f)
