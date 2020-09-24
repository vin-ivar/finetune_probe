import tensorboard

def log_lca(model, writer, step):
    lca = {}
    for k, v in model.named_parameters():
        if not v.requires_grad or type(v.grad) == type(None):
            continue

        lca[k] = (v.data - model._saved_params[k]) * v.grad

    self._saved_params = {k: v.data.clone() for k, v in self.named_parameters() if v.requires_grad}

    for k, v in lca.items():
        if v.mean().item() == 0:
            continue

        if 'LayerNorm' in k:
            continue

        # k_path = '.'.join(k.split('.')[5:])
        if 'query' in k or 'key' in k or 'value' in k:
            embed_size = v.size(0) // self.num_heads
            v = v.view(self.num_heads, embed_size, -1)
            for n in range(self.num_heads):
                mean, sum, numel = v[n].mean().item(), v[n].sum().item(), v[n].numel()
                self.writer.add_scalar(k + f'_head_{n}/mean', mean, self.step_counter)
                self.writer.add_scalar(k + f'_head_{n}/sum', sum, self.step_counter)
                self.writer.add_scalar(k + f'_head_{n}/numel', numel, self.step_counter)
            # continue

        mean, sum, numel = v.mean().item(), v.sum().item(), v.numel()
        self.writer.add_scalar(f'{k}/mean', mean, self.step_counter)
        self.writer.add_scalar(f'{k}/sum', sum, self.step_counter)
        self.writer.add_scalar(f'{k}/numel', numel, self.step_counter)
