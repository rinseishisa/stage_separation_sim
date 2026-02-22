# src/stage_sep/dynamics.py
import numpy as np

def pack_state(s: dict) -> np.ndarray:
    """dict状態 -> 1本のベクトルへ"""
    return np.hstack([
        s["r1"], s["v1"], s["q1"], s["w1"],
        s["r2"], s["v2"], s["q2"], s["w2"],
    ]).astype(float)

def unpack_state(x: np.ndarray) -> dict:
    """1本のベクトル -> dict状態へ"""
    x = np.asarray(x, dtype=float)
    i = 0
    r1 = x[i:i+3]; i+=3
    v1 = x[i:i+3]; i+=3
    q1 = x[i:i+4]; i+=4
    w1 = x[i:i+3]; i+=3

    r2 = x[i:i+3]; i+=3
    v2 = x[i:i+3]; i+=3
    q2 = x[i:i+4]; i+=4
    w2 = x[i:i+3]; i+=3

    return {"r1": r1, "v1": v1, "q1": q1, "w1": w1,
            "r2": r2, "v2": v2, "q2": q2, "w2": w2}

def normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])  # 単位クォータニオン
    return q / n

def f(x: np.ndarray, t: float, sim_config: dict) -> np.ndarray:
    """
    状態微分 xdot = f(x,t)
    MVP: 重力のみ（一定g）で並進を解く
    """
    s = unpack_state(x)

    g = np.asarray(sim_config["gravity"], dtype=float) #:contentReference[oaicite:3]{index=3}

    # 並進
    r1_dot = s["v1"]
    v1_dot = g
    r2_dot = s["v2"]
    v2_dot = g

    #