"""
Core augmentation logic for CAN bus attack datasets.

Provides two standalone functions:
  - generate_dos_dataset   : replaces '0000' DoS attack IDs with valid IDs
  - generate_spoof_dataset : injects spoofed packets between target-ID occurrences

Can be used directly or through the DoSAugmentor / SpoofAugmentor pipeline classes.

"""

import os
import pandas as pd
import random
import argparse


COLUMNS  = ['timestamp', 'id', 'dlc', 'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'label']
DATA_COLS = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']


def _get_unique_payloads_by_id(df):
    result = {}
    for can_id in df['id'].unique():
        result[can_id] = df[df['id'] == can_id][DATA_COLS].drop_duplicates().values.tolist()
    return result


def _random_payload():
    return [hex(random.randint(0, 255))[2:].zfill(2) for _ in range(8)]


def _build_predictor(df, cfg: dict):
    """
    Return a fitted :class:`~synthetic_data.payload_predictor.PayloadPredictor`.

    If *cfg* contains a ``model_path`` that already exists on disk the
    predictor is loaded from that file (no retraining).  Otherwise the
    predictor is trained on *df* and — when ``model_path`` is set — saved
    to that path so future runs can skip training.

    Parameters
    ----------
    df  : pd.DataFrame – CAN trace with columns id, d0..d7 (used for training)
    cfg : dict         – ``predictor`` sub-section from config.yaml

    Returns
    -------
    PayloadPredictor instance ready for :meth:`predict` calls
    """
    from synthetic_data.payload_predictor import PayloadPredictor

    path = cfg.get('model_path')

    run_eval = cfg.get('evaluate', False)

    if path and os.path.exists(path):
        predictor = PayloadPredictor.load(path)
        if run_eval:
            predictor.evaluate(df)
        return predictor

    predictor = PayloadPredictor(
        seq_len     = cfg.get('seq_len',     5),
        epochs      = cfg.get('epochs',      30),
        lr          = cfg.get('lr',          1e-3),
        min_samples = cfg.get('min_samples', 20),
    )
    predictor.fit(df)

    if run_eval:
        predictor.evaluate(df)

    if path:
        predictor.save(path)

    return predictor


def generate_dos_dataset(input_path, output_path, payload_mode='random', predictor_cfg=None):
    """
    Replace DoS attack packets (CAN ID '0000') with valid IDs sampled from
    IDs present in the dataset with integer value < the next legitimate ID in sequence.

    Scans rows backwards so that for each attack packet the reference upper-bound
    is the nearest legitimate CAN ID that follows it in the original trace.

    Args:
        input_path:    CSV with no header, columns: timestamp,id,dlc,d0..d7,label
        output_path:   path for the output CSV (no header)
        payload_mode:  'random'    — random hex bytes per byte field
                       'valid'     — randomly sampled real payload seen for that ID
                       'predicted' — LSTM-predicted next payload for the assigned ID
        predictor_cfg: dict of PayloadPredictor params (used only when
                       payload_mode='predicted').  Keys: seq_len, epochs,
                       lr, min_samples, model_path.
    """
    df = pd.read_csv(input_path, names=COLUMNS)

    valid_ids    = [h for h in df['id'].unique() if h != '0000']
    id_int_to_hex = {int(h, 16): h for h in valid_ids}
    sorted_ints  = sorted(id_int_to_hex.keys())
    index_of     = {v: i for i, v in enumerate(sorted_ints)}
    payload_dict = _get_unique_payloads_by_id(df) if payload_mode == 'valid' else None
    predictor    = _build_predictor(df, predictor_cfg or {}) if payload_mode == 'predicted' else None

    rows = df.to_dict('records')
    last_valid_id = None

    for i in range(len(rows) - 1, -1, -1):
        row    = rows[i]
        can_id = row['id']
        if can_id == '0000' and last_valid_id:
            lc = int(last_valid_id, 16)
            if index_of[lc] >= 1:
                # At least one valid ID exists below lc — pick from that range
                new_index = random.randint(0, index_of[lc] - 1)
            else:
                # lc is the smallest valid ID; no ID sits between 0000 and lc.
                # Fall back: pick any valid ID from the whole set
                new_index = random.randint(0, len(sorted_ints) - 1)
            new_id  = id_int_to_hex[sorted_ints[new_index]]
            if payload_mode == 'predicted':
                payload = predictor.predict(new_id)
            elif payload_mode == 'valid':
                payload = random.choice(payload_dict[new_id])
            else:
                payload = _random_payload()
            row['id'] = new_id
            for j, col in enumerate(DATA_COLS):
                row[col] = payload[j]
        else:
            last_valid_id = can_id

    result_df = pd.DataFrame(rows, columns=COLUMNS)
    result_df.to_csv(output_path, index=False, header=False)
    print(f"DoS synthetic dataset written to: {output_path}")


def generate_spoof_dataset(input_path, output_path, target_id, payload_mode='random',
                           max_injections=10, predictor_cfg=None):
    """
    Inject spoofed packets between consecutive BENIGN occurrences of target_id.

    For each pair of adjacent benign target_id packets at dataframe rows idx1 / idx2:

    * Bus capacity  = packets from any ID that appear between idx1 and idx2
                      in the original trace (idx2 - idx1 - 1).  This reflects
                      actual bus occupancy in that window.
    * N (injected)  = randint(1, max(1, min(capacity // 2, max_injections)))
                      — at most half the available capacity, hard-capped at
                      max_injections (default 10).
    * Payload       = one sample drawn from target_id's observed payloads (valid
                      mode), one random 8-byte vector (random mode), or the
                      LSTM-predicted next payload (predicted mode).  In predicted
                      mode each of the N injected packets advances the predictor's
                      history window, producing a coherent sequence rather than
                      N identical copies.
    * Timestamps    = evenly spaced inside (t1, t2); injected rows labelled 'T'.

    Args:
        input_path:    CSV with no header, columns: timestamp,id,dlc,d0..d7,label
        output_path:   path for the output CSV (with header)
        target_id:     hex CAN ID string to impersonate, e.g. '0164'
        payload_mode:  'random'    — random hex bytes per byte field
                       'valid'     — randomly sampled real payload seen for target_id
                       'predicted' — LSTM-predicted next payload for target_id
        max_injections: hard cap on N per gap (default 10)
        predictor_cfg: dict of PayloadPredictor params (used only when
                       payload_mode='predicted').  Keys: seq_len, epochs,
                       lr, min_samples, model_path.
    """
    df = pd.read_csv(input_path, names=COLUMNS)
    payload_dict = _get_unique_payloads_by_id(df) if payload_mode == 'valid' else None
    predictor    = _build_predictor(df, predictor_cfg or {}) if payload_mode == 'predicted' else None

    benign_mask    = (df['id'] == target_id) & (df['label'] != 'T')
    target_indices = df.index[benign_mask].tolist()
    injected_rows  = []

    for i in range(len(target_indices) - 1):
        idx1 = target_indices[i]
        idx2 = target_indices[i + 1]

        t1 = float(df.loc[idx1, 'timestamp'])
        t2 = float(df.loc[idx2, 'timestamp'])

        # Packets of any ID occupying the bus between the two boundary packets
        bus_capacity = idx2 - idx1 - 1
        n_max        = max(1, min(bus_capacity // 2, max_injections))
        num_injected = random.randint(1, n_max)
        gap = (t2 - t1) / (num_injected + 1)

        for j in range(num_injected):
            # In 'predicted' mode each packet gets its own prediction so the
            # injected sequence continues the natural signal trajectory.
            # In 'valid' / 'random' modes all N packets share one payload
            # (mimicking a single replayed message burst).
            if payload_mode == 'predicted':
                payload = predictor.predict(target_id)
            elif j == 0:
                payload = (
                    random.choice(payload_dict[target_id])
                    if payload_mode == 'valid'
                    else _random_payload()
                )

            injected_rows.append({
                'timestamp': round(t1 + gap * (j + 1), 6),
                'id':        target_id,
                'dlc':       8,
                **{DATA_COLS[k]: payload[k] for k in range(8)},
                'label':     'T',
            })

    result_df = pd.concat([df, pd.DataFrame(injected_rows)], ignore_index=True)
    result_df = result_df.sort_values('timestamp').reset_index(drop=True)
    result_df.to_csv(output_path, index=False)
    print(f"Spoof synthetic dataset written to: {output_path}")

