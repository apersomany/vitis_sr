
from basicsr.utils.download_util import load_file_from_url
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch
import torch_directml
import torch.onnx
    
load_file_from_url("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth", model_dir=".", file_name="model.pth")

class Model(RRDBNet):
    def __init__(self):
        super().__init__(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)

    def forward(self, x):
        x = x.half()
        x = x.div(255)
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)
        x = super().forward(x)
        x = x.squeeze()
        x = x.permute(1, 2, 0)
        x = x.mul(255)
        x = x.clamp(0, 255)
        x = x.byte()
        return x

model = Model()
model.load_state_dict(torch.load("model.pth")["params_ema"])
model.eval()
model.half()
model.to(torch_directml.device(0))

torch.onnx.export(
    model, 
    torch.randn(120, 720, 3).byte().to(torch_directml.device(0)),
    "model.onnx", 
    input_names=["i"],
    output_names=["o"],
    dynamic_axes={
        "i": {
            0: "ih",
            1: "iw"
        },
        "o": {
            0: "oh",
            1: "ow"
        }
    },
)