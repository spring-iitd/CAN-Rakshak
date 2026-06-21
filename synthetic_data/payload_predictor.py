import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


DATA_COLS = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']

class _ConditionalPayloadLSTM(nn.Module):

    def __init__(
        self,
        num_ids:     int,
        embed_dim:   int = 16,
        hidden_size: int = 64,
        num_layers:  int = 1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_ids, embed_dim)
        self.lstm = nn.LSTM(
            input_size=8 + embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head    = nn.Linear(hidden_size, 8)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, id_idx: torch.Tensor) -> torch.Tensor:
        emb    = self.embedding(id_idx)                        # (batch, embed_dim)
        emb    = emb.unsqueeze(1).expand(-1, x.size(1), -1)   # (batch, seq_len, embed_dim)
        x_cond = torch.cat([x, emb], dim=-1)                  # (batch, seq_len, 8+embed_dim)
        out, _ = self.lstm(x_cond)
        return self.sigmoid(self.head(out[:, -1, :]))

def _to_float_array(payloads: list) -> np.ndarray:
    """Convert list-of-list hex strings → (N, 8) float32 array in [0, 1]."""
    return np.array(
        [[int(b, 16) / 255.0 for b in p] for p in payloads],
        dtype=np.float32,
    )


def _to_hex_list(arr: np.ndarray) -> list:
    """Convert 1-D float array in [0, 1] → list of 8 lower-case hex strings."""
    ints = (np.clip(arr, 0.0, 1.0) * 255).round().astype(int)
    return [hex(int(v))[2:].zfill(2) for v in ints]


def _safe_torch_load(path: str) -> dict:
    """Load a torch checkpoint; handles both old and new PyTorch APIs."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


class PayloadPredictor:

    _HIDDEN_SIZE: int = 32
    _EMBED_DIM:   int = 8
    _NUM_LAYERS:  int = 1
    _BATCH_SIZE:  int = 256

    def __init__(
        self,
        seq_len:     int   = 5,
        epochs:      int   = 30,
        lr:          float = 1e-3,
        min_samples: int   = 20,
    ):
        self.seq_len     = seq_len
        self.epochs      = epochs
        self.lr          = lr
        self.min_samples = min_samples

        self._model:     _ConditionalPayloadLSTM = None
        self._id_to_idx: dict = {}   # can_id str → integer index in embedding table
        self._fallback:  dict = {}   # can_id → list[payload]  (under-sampled IDs)
        self._histories: dict = {}   # can_id → rolling context window for inference

    def fit(self, df) -> 'PayloadPredictor':
        trainable_ids = []
        for can_id in df['id'].unique():
            payloads = df[df['id'] == can_id][DATA_COLS].values.tolist()
            if len(payloads) >= self.min_samples + self.seq_len:
                trainable_ids.append(can_id)
            else:
                self._fallback[can_id] = payloads

        if not trainable_ids:
            print("  PayloadPredictor: no IDs had enough samples — all use fallback.")
            return self

        self._id_to_idx = {cid: i for i, cid in enumerate(trainable_ids)}

        all_X, all_y, all_id_idx = [], [], []

        for can_id in trainable_ids:
            payloads = df[df['id'] == can_id][DATA_COLS].values.tolist()
            floats   = _to_float_array(payloads)
            idx      = self._id_to_idx[can_id]

            for i in range(len(floats) - self.seq_len):
                all_X.append(floats[i: i + self.seq_len])
                all_y.append(floats[i + self.seq_len])
                all_id_idx.append(idx)

            self._histories[can_id] = payloads[-self.seq_len:]

        X_t   = torch.tensor(np.array(all_X),  dtype=torch.float32)  # (N, seq_len, 8)
        y_t   = torch.tensor(np.array(all_y),  dtype=torch.float32)  # (N, 8)
        ids_t = torch.tensor(all_id_idx,        dtype=torch.long)     # (N,)
        N     = len(all_X)

        num_ids     = len(trainable_ids)
        self._model = _ConditionalPayloadLSTM(
            num_ids, self._EMBED_DIM, self._HIDDEN_SIZE, self._NUM_LAYERS
        )
        opt     = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self._model.train()
        for epoch in range(self.epochs):
            perm  = torch.randperm(N)
            X_t   = X_t[perm]
            y_t   = y_t[perm]
            ids_t = ids_t[perm]

            epoch_loss = 0.0
            for start in range(0, N, self._BATCH_SIZE):
                end = min(start + self._BATCH_SIZE, N)
                opt.zero_grad()
                loss = loss_fn(self._model(X_t[start:end], ids_t[start:end]),
                               y_t[start:end])
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * (end - start)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    epoch {epoch + 1:>{len(str(self.epochs))}}/{self.epochs}"
                      f"  loss={epoch_loss / N:.5f}")

        self._model.eval()
        print(
            f"  PayloadPredictor: 1 model trained on {num_ids} IDs "
            f"({N:,} samples).  {len(self._fallback)} ID(s) use fallback sampling."
        )
        return self


    def predict(self, can_id: str) -> list:

        if can_id in self._id_to_idx and self._model is not None:
            history = self._histories[can_id]

            x = torch.tensor(
                [[[int(b, 16) / 255.0 for b in row] for row in history]],
                dtype=torch.float32,
            )  # (1, seq_len, 8)
            id_t = torch.tensor([self._id_to_idx[can_id]], dtype=torch.long)

            with torch.no_grad():
                pred = self._model(x, id_t)[0].numpy()  # (8,)

            payload = _to_hex_list(pred)

            self._histories[can_id] = history[1:] + [payload]
            return payload

        if can_id in self._fallback and self._fallback[can_id]:
            return list(random.choice(self._fallback[can_id]))

        return [hex(random.randint(0, 255))[2:].zfill(2) for _ in range(8)]



    def evaluate(self, df, val_ratio: float = 0.20) -> dict:

        train_frames = []
        test_data    = {}  

        for can_id in df['id'].unique():
            id_df    = df[df['id'] == can_id]
            n        = len(id_df)
            split_at = max(self.min_samples + self.seq_len, int(n * (1 - val_ratio)))
            if n - split_at < 1:
                continue
            payloads = id_df[DATA_COLS].values.tolist()
            train_frames.append(id_df.iloc[:split_at])
            test_data[can_id] = payloads[split_at:]

        if not train_frames:
            print("  [evaluate] No IDs have enough data for a train/val split.")
            return {}

        df_train = pd.concat(train_frames).reset_index(drop=True)

        print(f"  [evaluate] Training temporary model on {(1 - val_ratio) * 100:.0f}% split …")
        tmp = PayloadPredictor(self.seq_len, self.epochs, self.lr, self.min_samples)
        tmp.fit(df_train)

        per_id     = {}
        mae_preds  = []
        mae_naives = []

        for can_id, test_payloads in test_data.items():
            if can_id not in tmp._id_to_idx:
                continue

            test_floats = _to_float_array(test_payloads)   # (T, 8) in [0,1]
            preds       = np.zeros_like(test_floats)
            naives      = np.zeros_like(test_floats)

            # Seed naive baseline from the last training payload for this ID
            if can_id in tmp._histories and tmp._histories[can_id]:
                last_train_hex = tmp._histories[can_id][-1]
                last_real = np.array([int(b, 16) / 255.0 for b in last_train_hex])
            else:
                last_real = np.zeros(8)

            for t in range(len(test_floats)):
                pred_hex = tmp.predict(can_id)              # advances internal window
                preds[t] = np.array([int(b, 16) / 255.0 for b in pred_hex])
                naives[t] = last_real
                last_real = test_floats[t]                  # advance naive to real value

            mae_pred  = float(np.abs(preds  - test_floats).mean() * 255)
            mae_naive = float(np.abs(naives - test_floats).mean() * 255)
            per_id[can_id] = {
                'mae_pred':  round(mae_pred,  3),
                'mae_naive': round(mae_naive, 3),
                'n_steps':   len(test_floats),
            }
            mae_preds.append(mae_pred)
            mae_naives.append(mae_naive)

        mean_mae   = float(np.mean(mae_preds))  if mae_preds  else float('nan')
        mean_naive = float(np.mean(mae_naives)) if mae_naives else float('nan')

        # sep = "  " + "-" * 58
        # print()
        # print("  [evaluate] Payload prediction quality (MAE on 0–255 scale)")
        # print(sep)
        # print(f"  {'ID':<8}  {'Steps':>6}  {'MAE(pred)':>10}  {'MAE(naive)':>10}  {'Better':>6}")
        # print(sep)
        # for can_id, m in per_id.items():
        #     better = "pred " if m['mae_pred'] < m['mae_naive'] else "naive"
        #     print(f"  {can_id:<8}  {m['n_steps']:>6}  {m['mae_pred']:>10.2f}"
        #           f"  {m['mae_naive']:>10.2f}  {better:>6}")
        # print(sep)
        # print(f"  {'MEAN':<8}  {'':>6}  {mean_mae:>10.2f}  {mean_naive:>10.2f}")
        # print()
        # if mean_mae <= mean_naive:
        #     print("  [evaluate] PASS — LSTM predictions beat the naive baseline on average.")
        # else:
        #     print("  [evaluate] INFO — Naive repeat wins on step-wise MAE.")
        #     print("             This is expected for near-constant CAN signals.")
        #     print("             Distribution match (mean/std per ID) is the correct")
        #     print("             metric for augmentation; step-wise MAE is a lower bound.")
        # print()

        return {
            'per_id':     per_id,
            'mean_mae':   round(mean_mae,   3),
            'mean_naive': round(mean_naive, 3),
        }


    def save(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        torch.save(
            {
                # User-facing parameters (restored into __init__ on load)
                'config': dict(
                    seq_len=self.seq_len,
                    epochs=self.epochs,
                    lr=self.lr,
                    min_samples=self.min_samples,
                ),
                # Architecture constants (stored so the checkpoint is self-describing)
                'arch': dict(
                    hidden_size=self._HIDDEN_SIZE,
                    embed_dim=self._EMBED_DIM,
                    num_layers=self._NUM_LAYERS,
                ),
                'num_ids':     len(self._id_to_idx),
                'id_to_idx':   self._id_to_idx,
                'model_state': self._model.state_dict() if self._model is not None else None,
                'fallback':    self._fallback,
                'histories':   self._histories,
            },
            path,
        )
        print(f"  PayloadPredictor saved  → {path}")

    @classmethod
    def load(cls, path: str) -> 'PayloadPredictor':
        state    = _safe_torch_load(path)
        cfg      = state['config']
        instance = cls(**cfg)

        instance._id_to_idx = state['id_to_idx']
        instance._fallback  = state['fallback']
        instance._histories = state['histories']

        if state['model_state'] is not None:
            arch = state['arch']
            instance._model = _ConditionalPayloadLSTM(
                state['num_ids'],
                arch['embed_dim'],
                arch['hidden_size'],
                arch['num_layers'],
            )
            instance._model.load_state_dict(state['model_state'])
            instance._model.eval()

        print(
            f"  PayloadPredictor loaded ← {path}  "
            f"(1 model, {len(instance._id_to_idx)} IDs)"
        )
        return instance
