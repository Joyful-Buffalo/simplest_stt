from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.JsonlASRDataset import JsonlASRDataset
from utils.load_func import load_proj_root, load_yaml
from utils.CharTokenizer import CharTokenizer
from utils.asr_collate_fn import asr_collate_fn


def ort_type_to_np(ort_type: str):
    t = ort_type.lower()
    if "float16" in t:
        return np.float16
    if "float" in t:
        return np.float32
    if "int64" in t:
        return np.int64
    if "int32" in t:
        return np.int32
    raise ValueError(f"暂不支持的 ONNX 输入类型: {ort_type}")


def get_single_feature_input(onnx_path: Path):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()

    feat_inputs = [x for x in inputs if len(x.shape) == 3 and "float" in x.type.lower()]
    if len(feat_inputs) != 1:
        detail = "\n".join([f"  name={x.name}, shape={x.shape}, type={x.type}" for x in inputs])
        raise RuntimeError(
            "期望 ONNX 只有 1 个 3 维浮点输入（即导出的 ASR 特征输入），实际输入为：\n"
            f"{detail}"
        )
    return sess, feat_inputs[0]


def build_dataset(proj_root: Path, yaml_config: dict):
    jsonl_paths = yaml_config["jsonl_paths"]
    test_key = "test_jsonl" if "test_jsonl" in jsonl_paths else "dev_jsonl"
    test_jsonl = proj_root / jsonl_paths[test_key]

    tokenizer = CharTokenizer(
        chars=None,
        jsonl_paths=[str(proj_root / v) for v in jsonl_paths.values()],
    )

    return JsonlASRDataset(
        jsonl_file_path=str(test_jsonl),
        tokenizer=tokenizer,
        config=yaml_config["dataset_config"],
    )


def center_crop(feat: np.ndarray, target_t: int) -> np.ndarray:
    start = (feat.shape[0] - target_t) // 2
    return feat[start:start + target_t]


def main():
    proj_root = Path(load_proj_root())

    yaml_path = proj_root / "config" / "baseline.yaml"
    onnx_path = proj_root / "model_fp32.onnx"
    calib_npz_path = proj_root / "calib_input.npz"

    yaml_config = load_yaml(yaml_path)
    feat_dim = int(yaml_config["dataset_config"]["fbank_config"]["feature_dim"])

    # 单个 npz 校准包必须选一个固定代表形状
    calib_t = 512
    calib_steps = 256

    dataset = build_dataset(proj_root, yaml_config)
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=asr_collate_fn,
        pin_memory=False,
    )

    sess, feat_input = get_single_feature_input(onnx_path)

    print("ONNX input:")
    print(f"  name={feat_input.name}, shape={feat_input.shape}, type={feat_input.type}")

    samples = []
    skipped_short = 0

    for batch in loader:
        feat = batch["features"][0].detach().cpu().numpy()   # [T, F]
        feat_len = int(batch["feature_lengths"][0].item())

        if feat.ndim != 2:
            raise RuntimeError(f"features 期望 [T, F]，实际 {feat.shape}")
        if feat.shape[1] != feat_dim:
            raise RuntimeError(f"feature_dim 不一致: 数据={feat.shape[1]}, 配置={feat_dim}")

        # 不能“补零到 calib_t 但又假装是真实长度”
        if feat_len < calib_t:
            skipped_short += 1
            continue
        feat = center_crop(feat[:feat_len], calib_t).astype(np.float32, copy=False)
        samples.append(feat)

        if len(samples) >= calib_steps:
            break

    if len(samples) < calib_steps:
        raise RuntimeError(
            f"可用校准样本不足。要求至少 {calib_steps} 条长度 >= {calib_t} 的样本，"
            f"实际只有 {len(samples)} 条，跳过短样本 {skipped_short} 条。"
            f"请减小 calib_t，或换更长的数据集。"
        )

    calib_arr = np.stack(samples, axis=0).astype(ort_type_to_np(feat_input.type), copy=False)

    dry_run = calib_arr[:1]
    _ = sess.run(None, {feat_input.name: dry_run})
    print("ORT dry-run passed.")

    np.savez(calib_npz_path, **{feat_input.name: calib_arr})

    print(f"\n已生成: {calib_npz_path}")
    print(f"  {feat_input.name}: shape={calib_arr.shape}, dtype={calib_arr.dtype}")


if __name__ == "__main__":
    main()