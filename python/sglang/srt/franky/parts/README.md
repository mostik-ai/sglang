# Add a model to Franky

A model joins Franky through `FrankyMixin`. `python/sglang/srt/models/qwen3_5.py` is the reference.

**1. Add the mixin to the language model class, and name the form of its residual:**

```python
class MyModel(FrankyMixin, nn.Module):
    franky_output_cls = StreamOutput
```

The form is the `LayerOutput` that the model hands Franky after a layer. Each form has an injector in
`injector/`, and the Franky config names one injector per side (`injector.sandy`, `injector.ricky`). The injector
alone reads the residual for encoders and readers and adds a reader's result to it; parts never touch the residual
themselves. A side whose injector does not take the model's form is refused at start.

| Form | Residual | Injector | Read | Inject |
| --- | --- | --- | --- | --- |
| `StreamOutput` | one stream that `residual_batch` keeps in `forward_batch` | `residual` | the stream's sum | add to the layer's output |
| `AttentionBankOutput` | attention residuals (Kimi K3): a bank of block sums and the running sum | `attention_bank` | the mix the next layer reads, as DSpark's capture | add to every row of that mix |
| `HyperConnectionOutput` | manifold hyper-connections (DeepSeek V4): several streams | `hyper_connection` | the mean of the streams, as DSpark's capture | add to every stream |

**2. In the layer loop, right after the layer runs,** call the method of the model's form:

```python
hidden_states = self.franky_after_layer(
    layer_idx=layer_idx, hidden_states=hidden_states, forward_batch=forward_batch
)
```

`franky_after_attention_bank_layer` also takes `next_read`, which says how the consumer after a layer mixes the bank
(`KimiK3LinearModel._next_attention_bank_read`). `franky_after_hyper_connection_layer` takes the complete stream state
and the rows it holds: DeepSeek V4.1 runs its layers past the tail switch only on each request's last rows, and a
reader there reads the taps of those rows.

**KV.** Taps live at the token's KV slot, so Ricky needs an allocator whose slots last as long as a token's KV: the
paged one, or the sliding-window one with a full pool, whose full-pool slots hold the taps (DeepSeek V4).

**3. For Sandy under PP, at the stage edges:**

```python
# After reading the previous stage's tensors.
self.franky_read_pp(pp_proxy_tensors=pp_proxy_tensors)

# Before returning tensors to the next stage.
self.franky_write_pp(pp_proxy_tensors=pp_proxy_tensors)
```

**What Franky reads from the model:** the class with the mixin has `start_layer`, `end_layer` (as `make_layers` gives them) and `config.hidden_size`. The top model class has `lm_head` and `logits_processor`, as SGLang models do; on Sandy they are replaced, because Sandy never samples.

A model used only as Ricky needs steps 1 and 2. A residual of another form needs its own `LayerOutput`, injector and
`FrankyMixin` method; encoders and readers only ever see one hidden per row.
