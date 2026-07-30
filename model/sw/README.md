# NPU Layer CSV Inputs

- Path rule (new): `model/sw/cnn/<npu>/<network>.csv`
- Derivation report per CSV: `model/sw/cnn/<npu>/<network>.md`
- One row = one function descriptor (multiple rows can share one `layer_id`).
- Recommended runtime path: `-scenario <npu>:<network>` (resolver maps to this tree).

Required columns:
- `layer_id,function_id,layer_name,layer_type,function_type`
- `i1_connect,i2_connect`
- `input_h,input_w,input_c`
- `kernel_h,kernel_w,kernel_c,kernel_count`
- `stride,padding,dilation`
- `tile_h,tile_w`
- `pe_type,k,g,G,q`
- `preprocess_type,preprocess_param0,preprocess_param1`

Optional columns:
- `input_addr,kernel_addr,output_addr` (used by explicit-address smoke/sanity scenarios)

Legacy flat files are removed. Use only `model/sw/cnn/<npu>/<network>.csv`.

These per-NPU CSVs are checked-in artifacts. The former generator
`generate_npu_model_csvs.py` was removed — it depended on a deleted GUI module
and expected a flat hw-JSON schema (`pe_type/k0/g0/...`) that never matched the
shipped specs. HW specs now follow `model/hw/npu/README.md`.

Example run:
```bash
build/sim/flexnpusim \
  -scenario midap:mobilenetv2 \
  -o result.csv
```

Equivalent explicit-path run:
```bash
build/sim/flexnpusim \
  -hw_conf model/hw/npu/midap.json \
  -network model/sw/cnn/midap/mobilenetv2.csv \
  -o result.csv
```

Quick smoke scenarios:
- `midap:tiny2layer`
- `midap:dynamic_tail`
- `nvdla-large:tiny2layer`
