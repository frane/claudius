"""Make the browser variants of the ONNX export (transformers.js file layout).

    .venv-onnx/bin/python scripts/quantize_onnx.py out/onnx/fp32/model.onnx out/onnx/web/onnx

model.onnx (fp32), model_fp16.onnx, model_quantized.onnx (int8 weights),
model_q4.onnx (4-bit weights, block 32), model_q4f16.onnx (4-bit, fp16 rest).
"""
import os
import shutil
import sys

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from onnxruntime.transformers.float16 import convert_float_to_float16


src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
shutil.copy(src, f"{out}/model.onnx")

m = onnx.load(src)
onnx.save_model(convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True),
                f"{out}/model_fp16.onnx")
print("fp16 done", flush=True)

quantize_dynamic(src, f"{out}/model_quantized.onnx", weight_type=QuantType.QInt8,
                 extra_options={"MatMulConstBOnly": True})
print("int8 done", flush=True)

m = onnx.load(src)
q = MatMulNBitsQuantizer(m, block_size=32, is_symmetric=True)
q.process()
onnx.save_model(q.model.model, f"{out}/model_q4.onnx")
print("q4 done", flush=True)

m16 = convert_float_to_float16(onnx.load(f"{out}/model_q4.onnx"), keep_io_types=True, disable_shape_infer=True)
onnx.save_model(m16, f"{out}/model_q4f16.onnx")
print("q4f16 done", flush=True)
