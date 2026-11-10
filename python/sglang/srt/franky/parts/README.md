# Add a model to Franky

A model joins Franky through `FrankyMixin`. `python/sglang/srt/models/qwen3_5.py` is the reference.

**1. Add the mixin to the language model class:**

```python
class MyModel(FrankyMixin, nn.Module):
```

**2. In the layer loop, right after the layer runs.** Use the form that matches what the layer returns.

Layers that return `hidden_states`:

```python
hidden_states = self.franky_after_layer(
    layer_idx=layer_idx, hidden_states=hidden_states, forward_batch=forward_batch
)
```

Layers that return `(hidden_states, residual)`:

```python
hidden_states, residual = self.franky_after_layer_with_residual(
    layer_idx=layer_idx,
    hidden_states=hidden_states,
    residual=residual,
    forward_batch=forward_batch,
)
```

**3. For Sandy under PP, at the stage edges:**

```python
# After reading the previous stage's tensors.
self.franky_read_pp(pp_proxy_tensors=pp_proxy_tensors)

# Before returning tensors to the next stage.
self.franky_write_pp(pp_proxy_tensors=pp_proxy_tensors)
```

**What Franky reads from the model:** the class with the mixin has `start_layer`, `end_layer` (as `make_layers` gives them) and `config.hidden_size`. The top model class has `lm_head` and `logits_processor`, as SGLang models do; on Sandy they are replaced, because Sandy never samples.

A model used only as Ricky needs steps 1 and 2. A layer output of another shape needs one more `after_layer` form in `FrankyMixin` and `FrankyParts`; parts themselves only see the residual as a tensor.
