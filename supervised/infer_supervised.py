"""
Inference wrapper matching RL decision API.
Provides a callable predict_action(state_vector) for service integration.
"""

import os, json, numpy as np, pandas as pd, joblib, torch
from typing import Any, Optional

ACTION_MAP = {-1: 0, 0: 2, 1: 4}

# --- Model Loading ---

def load_lightgbm(model_path):
    import lightgbm as lgb
    model = lgb.Booster(model_file=model_path)
    meta_path = model_path + ".meta.pkl"
    meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}
    return model, meta

def load_pytorch(model_path, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(model_path, map_location=device)
    return ckpt, device

# --- Inference ---

def predict_from_state_lightgbm(model, meta, state_df):
    feature_cols = meta.get("feature_cols", state_df.columns.tolist())
    X = state_df[feature_cols].values
    probs = model.predict(X)
    cls = int(np.argmax(probs, axis=1)[0]) - 1  # Convert back from [0,1,2] to [-1,0,1]
    return cls

def predict_from_state_pytorch(ckpt, device, state_tensor):
    from supervised.train_pytorch import SmallCNN
    model = SmallCNN(ckpt["n_features"], ckpt["window"], n_classes=3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    with torch.no_grad():
        out = model(state_tensor.to(device))
        cls = int(out.argmax(dim=1).cpu().numpy()[0]) - 1
    return cls

# --- Utilities ---

def state_to_df(state_vector, feature_cols):
    arr = np.array(state_vector)
    n_features = len(feature_cols)
    window = arr.size // n_features
    arr2 = arr.reshape(window, n_features)
    df = pd.DataFrame(arr2, columns=feature_cols)
    return df.iloc[-1:].reset_index(drop=True)

# --- Unified API ---

def load_model(model_path, model_type="lightgbm", feature_cols=None):
    """Load model and return model object and metadata."""
    if model_type == "lightgbm":
        model, meta = load_lightgbm(model_path)
        if not feature_cols:
            feature_cols = meta.get("feature_cols")
        return model, meta, feature_cols
    else:
        ckpt, device = load_pytorch(model_path)
        return ckpt, device, feature_cols

def predict_action(model, state_dict, model_type="lightgbm", feature_cols=None, meta=None, ckpt=None, device=None):
    """Predict action from state dictionary."""
    if model_type == "lightgbm":
        # Convert state dict to DataFrame
        state_df = pd.DataFrame([state_dict])
        cls = predict_from_state_lightgbm(model, meta, state_df)
        # Get probability distribution
        probs = model.predict(state_df[feature_cols].values)
        prob = float(np.max(probs))
    else:
        # Convert state dict to tensor
        state_vector = [state_dict.get(col, 0.0) for col in feature_cols]
        tensor = torch.tensor(state_vector, dtype=torch.float32).reshape(1, ckpt["n_features"], ckpt["window"])
        cls = predict_from_state_pytorch(ckpt, device, tensor)
        # For PyTorch, we'd need to get probabilities from the model
        prob = 0.8  # Placeholder - would need to modify PyTorch prediction
    
    action = ACTION_MAP.get(cls, 2)
    return {"action": action, "prob": prob, "class": cls}

class SupervisedInference:
    def __init__(self, model_path, model_type="lightgbm", feature_cols=None):
        self.model_type = model_type
        self.feature_cols = feature_cols
        if model_type == "lightgbm":
            self.model, self.meta = load_lightgbm(model_path)
            if not self.feature_cols:
                self.feature_cols = self.meta.get("feature_cols")
        else:
            self.ckpt, self.device = load_pytorch(model_path)

    def predict_action(self, state_vector):
        if self.model_type == "lightgbm":
            df = state_to_df(state_vector, self.feature_cols)
            cls = predict_from_state_lightgbm(self.model, self.meta, df)
            # Get probability
            probs = self.model.predict(df[self.feature_cols].values)
            prob = float(np.max(probs))
        else:
            tensor = torch.tensor(state_vector, dtype=torch.float32).reshape(1, self.ckpt["n_features"], self.ckpt["window"])
            cls = predict_from_state_pytorch(self.ckpt, self.device, tensor)
            prob = 0.8  # Placeholder
        
        action = ACTION_MAP.get(cls, 2)
        return {"action": action, "prob": prob, "class": cls}

# --- CLI Entry Point ---

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model_type", choices=["lightgbm", "pytorch"], default="lightgbm")
    p.add_argument("--state_json", required=False)
    p.add_argument("--feature_cols", required=False)
    args = p.parse_args()

    feature_cols = json.loads(open(args.feature_cols).read()) if args.feature_cols else None
    inf = SupervisedInference(args.model, args.model_type, feature_cols)

    state = json.loads(open(args.state_json).read()) if args.state_json else None
    action = inf.predict_action(state)
    print(json.dumps({"action": int(action)}))
