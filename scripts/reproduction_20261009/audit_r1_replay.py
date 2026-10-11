"""CPU-only evidence audit; creates an exclusive report without changing experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/p0B_legacy_replay_20261008'
RUN = ROOT / 'results/p0B_legacy_replay_20261008/run_01'


def parse_log(path):
    rows = {}
    epoch = None
    for line in path.read_text().splitlines():
        match = re.search(r'\[Train\] Epoch (\d+) \| loss=([\d.]+) \| lr=(.*)', line)
        if match:
            epoch = int(match[1])
            rows[epoch] = dict(loss=float(match[2]), lr=match[3])
        if '[valid]' in line and epoch is not None:
            rows[epoch].update({k: float(v) for k, v in re.findall(
                r'(MAE|Corr|Acc2|Acc7|F1_has0)=([\d.]+)', line)})
    return rows


def permutations(persistent):
    # Isolate the loader mechanism. No claim that the artificial RNG consumption
    # reproduces model initialization or historical dropout calls.
    torch.random.default_generator.manual_seed(2024)
    train = DataLoader(range(1284), batch_size=4, shuffle=True, num_workers=4,
                       persistent_workers=persistent)
    valid = DataLoader(range(229), batch_size=4, shuffle=False, num_workers=4,
                       persistent_workers=persistent)
    rows = []
    for epoch in range(2):
        indices = torch.cat(list(train))
        list(valid)
        rows.append(dict(epoch=epoch + 1, first_indices=indices[:24].tolist(),
                         sha256=hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
                         indices=indices.tolist()))
    del train, valid
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=False)
    historical = parse_log(BASE / 'historical_evidence/MSA_MOSI_20260918_233755.log')
    current = parse_log(RUN / 'console.log')
    manifest = json.loads((BASE / 'recovery_manifest.json').read_text())
    old_path = Path(manifest['preserved_best']['checkpoint'])
    new_path = RUN / 'checkpoints/MOSI_p0B_seed2024_best_Acc2.pt'
    metadata = {}
    shapes = []
    for name, path in [('historical', old_path), ('replay', new_path)]:
        payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        state = payload['model']
        shapes.append({k: list(v.shape) for k, v in state.items()})
        metadata[name] = dict(path=str(path), keys=list(payload),
            tensor_count=len(state), elements=sum(v.numel() for v in state.values()),
            metadata={k: v for k, v in payload.items()
                      if k != 'model' and isinstance(v, (dict, str, int, float, bool, type(None)))})
        del state, payload
    orders = {str(p): permutations(p) for p in [False, True]}
    order_comparison = []
    for a, b in zip(orders['False'], orders['True']):
        order_comparison.append(dict(epoch=a['epoch'],
            identical=a['sha256'] == b['sha256'],
            changed_positions=sum(x != y for x, y in zip(a['indices'], b['indices']))))
        del a['indices'], b['indices']
    old_args = json.loads((BASE / 'historical_args.json').read_text())
    text = (RUN / 'console.log').read_text()
    new_args, _ = json.JSONDecoder().raw_decode(text.split('Args: ', 1)[1])
    deltas = {k: dict(historical=old_args.get(k), replay=new_args.get(k))
              for k in sorted(set(old_args) | set(new_args)) if old_args.get(k) != new_args.get(k)}
    report = dict(cuda_initialized=torch.cuda.is_initialized(), checkpoint_metadata=metadata,
        state_dict_shapes_equal=shapes[0] == shapes[1], argument_differences=deltas,
        learning_rates_equal=all(historical[e]['lr'] == current[e]['lr']
                                for e in current if e in historical),
        first_epochs={str(e): dict(historical=historical[e], replay=current[e]) for e in [1, 2, 3]},
        loader_mechanism=dict(orders=orders, comparisons=order_comparison,
            scope='Synthetic index dataset, global CPU RNG; causal mechanism only, historical flag unknown'),
        conclusion='Hyperparameters and checkpoint shape compatibility do not establish training trajectory equivalence.')
    assert not report['cuda_initialized']
    (cli.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k not in ['checkpoint_metadata', 'loader_mechanism']}, ensure_ascii=False))
    print('LOADER', order_comparison)


if __name__ == '__main__':
    main()
