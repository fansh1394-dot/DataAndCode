# RTDETRMM YAML Layout

RTDETRMM configs are organized by task first, then backbone, then variant family:

```text
rtmm/
  detect/
    r18/{base,aifi,extraction,change,backbone}/
    r50/{base,aifi,extraction}/
  obb/
    r18/base/
```

## Header rules

New RTDETRMM YAML files should start with this project declaration and a short architecture summary:

```yaml
# YOLOMM|RTDETRMM 多模态项目,汤圆版权所有,仅限学术用途
# RTDETRMM-r18 Detect Base: <name>
# 思路: <why this architecture exists>
# 核心策略: <RGB/X branch depth, fusion levels, decoder input shape>
# 设计原则: <main trade-off>
# 适用场景: <recommended task/data setting>
# Decoder contract: keep RTDETRDecoder inputs as P3/P4/P5.
```

## Directory rules

- `task`: use `detect`, `obb`, `seg`, `pose`, `semseg`, or `cls`. Add a task directory only when a real YAML exists.
- `backbone`: use the RT-DETR backbone family such as `r18` or `r50`.
- `base`: standard task baselines such as `*-mm.yaml`, `*-mm-early.yaml`, `*-mm-mid.yaml`, and task heads such as `*-mm-mid-obb.yaml`.
- `aifi`: `*-mm-mid-aifi-*` encoder variants.
- `extraction`: `*-mm-mid-c2f-*` feature extraction variants.
- `change`: fusion or structural experiments such as `ssa`, `msaa`, `deyolo`, `fscf`, `cam`, `mrod`, `rd`, `fcm`, `hfp`, `pst`, and `unirgb-ir`.
- `backbone`: `*-mm-bb-*` backbone replacement configs.
- `head`, `neck`, and `sample`: reserved for future task-local variants that match the YOLOMM layout.

Do not keep duplicate YAML basenames in different directories. `check_yaml()` resolves bare filenames by recursive search, and duplicate names make calls such as `RTDETRMM("rtdetr-r18-mm-mid.yaml")` ambiguous.

## Detect r18 base variants

- `rtdetr-r18-mm.yaml`: RGB-only YAML that can enter the RTDETRMM route for single-modal compatibility.
- `rtdetr-r18-mm-early.yaml`: `Dual` input early fusion.
- `rtdetr-r18-mm-mid.yaml`: P3 fusion baseline, then single fused branch to P4/P5.
- `rtdetr-r18-mm-mid-p3p4.yaml`: P3/P4 fusion, then P5 generated from fused P4.
- `rtdetr-r18-mm-mid-p4p5.yaml`: P4/P5 fusion with RGB P3 shallow feature.
- `rtdetr-r18-mm-mid-p3p4p5.yaml`: full P3/P4/P5 fusion.
- `rtdetr-r18-mm-asymmetric.yaml`: RGB deep branch to P5 and X shallow branch to P4.
- `rtdetr-r18-mm-residual.yaml`: progressive P2->P3->P4->P5 residual fusion.
- `rtdetr-r18-mm-late-ref.yaml`: branch-local FPN/PAN first, late feature fusion before one decoder.
- `rtdetr-r18-mm-p2-inject.yaml`: P2 detail injection into P3 while keeping P3/P4/P5 decoder inputs.
