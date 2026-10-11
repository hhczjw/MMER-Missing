"""Post-initialization model-only interventions. Empty changes are a strict no-op.

Native trainer, tokenization, kernels, data loader and RNG are not replaced.
GLCE branch removal keeps the original shortcut and second normalization.
Self K uses the existing two source-specific B projections on the target input;
self V uses the native target in_proj slice (not newly initialized weights).
"""
import inspect
import textwrap
import types


def cell_source(source, mode):
    if mode not in ('self','key_self','value_self'):
        raise ValueError(mode)
    replacements = {}
    if mode in ('self','key_self'):
        replacements.update({
            'B0 = self.b_projs[s0_key](u_src0)': 'B0 = self.b_projs[s0_key](u_tgt)',
            'B1 = self.b_projs[s1_key](u_src1)': 'B1 = self.b_projs[s1_key](u_tgt)'})
    if mode in ('self','value_self'):
        replacements['x = w0 * V0 + w1 * V1'] = 'x = x_default'
    for before,after in replacements.items():
        if source.count(before)!=1:
            raise RuntimeError('Native cell source changed: '+before)
        source = source.replace(before,after)
    return source


def apply(model, changes):
    if not changes:
        return []
    import torch
    touched = []
    def zero_branch(module, inputs, output):
        return torch.zeros_like(output)
    for li,layer in enumerate(model.layers):
        if changes.get('uniform'):
            for name,net in layer.weight_nets.items():
                net.requires_grad_(False)
                net.register_forward_hook(zero_branch)  # native softmax(zeros) => 1/2,1/2
                touched.append(f'layers.{li}.weight_nets.{name}')
        mode = changes.get('source_mode')
        if mode:
            for name,cell in layer.cells.items():
                if cell.v_self_ratio != 0:
                    raise RuntimeError('Self-V definition prespecified for native v_self_ratio=0')
                source = cell_source(textwrap.dedent(inspect.getsource(type(cell).forward)),mode)
                scope = dict(type(cell).forward.__globals__)
                exec(compile(source,'<native-cell-'+mode+'>','exec'),scope)
                cell.forward = types.MethodType(scope['forward'],cell)
                if mode in ('self','value_self'):
                    cell.v_projs.requires_grad_(False)
                touched.append(f'layers.{li}.cells.{name}: '+mode)
        if changes.get('gcmn') is False:
            if layer.gcmn is None:
                raise RuntimeError('Native baseline GCMN is unexpectedly absent')
            layer.gcmn = None
            touched.append(f'layers.{li}.gcmn')
    for name in ('ism_text','ism_audio','ism_video'):
        for bi,block in enumerate(getattr(model,name).layers):
            for change,branch in [('global','global_extractor'),('local','local_extractor')]:
                if changes.get(change) is False:
                    module = getattr(block,branch)
                    module.requires_grad_(False)
                    module.register_forward_hook(zero_branch)
                    touched.append(f'{name}.layers.{bi}.{branch}')
            if changes.get('bssm') is False:
                if not block.use_bssm_gate:
                    raise RuntimeError('Native baseline BSSM is unexpectedly disabled')
                block.use_bssm_gate = False
                block.bssm_gate.requires_grad_(False)
                touched.append(f'{name}.layers.{bi}.bssm_gate')
    return touched


def verify_gradients(model, changes):
    """Observe post-clip first-step gradients; no RNG or tensor mutation."""
    import torch
    rows = {n:float(p.grad.detach().norm()) for n,p in model.named_parameters()
            if p.grad is not None and any(k in n for k in (
                'proj_audio','weight_nets','b_projs','v_projs','global_extractor',
                'local_extractor','bssm_gate','gcmn','sub_fc'))}
    if not rows or not all(torch.isfinite(torch.tensor(v)).item() for v in rows.values()):
        raise RuntimeError('Missing/nonfinite gradient evidence')
    if not any('proj_audio' in n and v>0 for n,v in rows.items()):
        raise RuntimeError('Native audio projection has no gradient')
    for change,stem in [('global','global_extractor'),('local','local_extractor'),('bssm','bssm_gate'),('gcmn','gcmn')]:
        if changes.get(change) is False and any(stem in n for n in rows):
            raise RuntimeError('Disabled branch still has gradients: '+stem)
    if changes.get('uniform') and any('weight_nets' in n for n in rows):
        raise RuntimeError('Uniform routing still trains the routing network')
    if changes.get('source_mode') in ('self','value_self') and any('v_projs' in n for n in rows):
        raise RuntimeError('Unused source V projections still have gradients')
    if any('sub_fc' in n for n in rows):
        raise RuntimeError('Unexpected unimodal supervision; expected M-only')
    return rows
