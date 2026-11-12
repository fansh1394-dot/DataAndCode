# -*- coding: utf-8 -*-
"""把单输入 6ch ONNX 改造成 rgb(3) + ir(3) 双输入，便于 RKNN 量化。"""
import onnx
from onnx import TensorProto, helper

SRC = "/root/rivermind-fs/birddrone/deploy_nc3/model_raw.onnx"
DST = "/root/rivermind-fs/birddrone/deploy_nc3/model_2in.onnx"

m = onnx.load(SRC)
inp = m.graph.input[0]
name = inp.name
shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
print("orig input", name, shape)
assert shape[1] == 6

rgb = helper.make_tensor_value_info("rgb", TensorProto.FLOAT, [1, 3, shape[2], shape[3]])
ir = helper.make_tensor_value_info("ir", TensorProto.FLOAT, [1, 3, shape[2], shape[3]])
concat = helper.make_node("Concat", ["rgb", "ir"], [name], axis=1, name="mm_concat_in")

m.graph.input.remove(inp)
m.graph.input.extend([rgb, ir])
m.graph.node.insert(0, concat)
onnx.checker.check_model(m)
onnx.save(m, DST)
print("saved", DST)
