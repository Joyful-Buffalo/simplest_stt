import os
from pathlib import Path
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.CharTokenizer import CharTokenizer
from utils.load_func import load_proj_root, load_yaml
from models.ctc_conformer import CTCConformer


class OnnxExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super(OnnxExportWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        input_lengths = torch.full(
            (x.shape[0],),
            x.shape[1],
            dtype=torch.int64,
            device=x.device
        )
        y = self.model(x, input_lengths)

        if isinstance(y, (tuple, list)):
            logits = y[0]
            if len(y) > 1 and torch.is_tensor(y[1]):
                output_lengths = y[1].to(torch.int64)
            else:
                output_lengths = torch.full(
                    (x.shape[0],),
                    logits.shape[1],
                    dtype=torch.int64,
                    device=x.device
                )
        else:
            logits = y
            output_lengths = torch.full(
                (x.shape[0],),
                logits.shape[1],
                dtype=torch.int64,
                device=x.device
            )
        return logits, output_lengths


class ExportModel:
    def __init__(self, proj_root=load_proj_root()):
        self.yaml_config_path = Path(f"{proj_root}/config/baseline.yaml")
        self.yaml_config = load_yaml(self.yaml_config_path)

        self.dataset_config = self.yaml_config["dataset_config"]
        self.model_config = self.yaml_config["model_config"]

        self.char_tokenizer = CharTokenizer(
            chars=None,
            jsonl_paths=[os.path.join(proj_root, v) for _ , v in self.yaml_config["jsonl_paths"].items()],
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(os.path.join(proj_root, 'log', '20260226-204151', 'best_model.pth'))
        self.wrapper_model = OnnxExportWrapper(self.model).to(self.device)
        self.onnx_model_path = os.path.join(proj_root, 'model_fp32.onnx')

    def load_model(self, model_path='None'):
        origin_weight = torch.load(model_path, map_location=self.device)
        weight = {}
        for k, v in origin_weight.items():
            if k.startswith('_orig_mod.'):
                weight[k[10:]] = v
            else:
                weight[k] = v
        model = CTCConformer(
            input_dim=self.yaml_config['dataset_config']['fbank_config']['feature_dim'],
            vocab_size=self.char_tokenizer.vocab_size(),
            **self.yaml_config['model_config']['encoder']
        )
        model.load_state_dict(weight)
        return model.to(self.device).eval()
    
    def export_onnx(self):
        feat_dim = self.yaml_config['dataset_config']['fbank_config']['feature_dim']
        dummy_input = torch.randn(1, 100, feat_dim, device=self.device)
        dynamic_T = torch.export.Dim.AUTO
        with torch.inference_mode():
            torch.onnx.export(
                self.wrapper_model,
                (dummy_input,),
                self.onnx_model_path,
                input_names=["input"],
                output_names=["logits", "output_lengths"],
                opset_version=18,
                dynamo=True,
                dynamic_shapes=(
                    (None, dynamic_T, feat_dim), 
                ),
                optimize=True,
                verify=False,
                report=False,
                fallback=False,
                external_data=False
            )
    

if __name__ == "__main__":
    export_model = ExportModel()
    export_model.export_onnx()
