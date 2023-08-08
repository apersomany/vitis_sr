# vitis sr

Super resolution proxy for Vitis

## Limitations

- Only supports directml as backend



```sh
model.py # downloads and converts model to onnx
serve.py # runs the proxy (runs model.py if model.onnx is not present)
```