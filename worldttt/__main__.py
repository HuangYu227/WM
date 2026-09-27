"""python -m worldttt --help (help and doctor work without SANA/CUDA imports)."""
import argparse
import importlib.util
import json
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='WorldTTT v1: ordinary causal SANA with episode adaptation')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor', help='Report installed runtime and CUDA availability')
    for name in ('train', 'grail-train', 'grail-experiment', 'infer', 'evaluate', 'fixture', 'check-cache', 'query-eval', 'check-data'):
        cmd = commands.add_parser(name)
        cmd.add_argument('--settings', required=True, help='JSON run settings')
        if name != 'check-data':
            cmd.add_argument('--output', required=True)
        if name in {'train', 'grail-train', 'grail-experiment', 'infer', 'evaluate', 'query-eval'}:
            cmd.add_argument('--adapter', required=name == 'grail-experiment')
        if name == 'grail-experiment':
            cmd.add_argument('--cases', help='Camera-selected windows from worldttt.grail_long prepare; overrides --samples')
            cmd.add_argument('--seeds', nargs='+', type=int, help='Paired query seeds; clips and seeds are not independent scenes')
            cmd.add_argument('--split', choices=['val', 'test'], default='val')
            cmd.add_argument('--samples', type=int, default=4, help='One clip from each of this many scenes')
            cmd.add_argument('--frames', type=int, help='Exact latent-frame count; at least 10')
            cmd.add_argument('--histories', nargs='+', choices=['real', 'generated'], default=['real'])
            cmd.add_argument('--variants', nargs='+', choices=['ridge', 'no_read', 'prototype', 'shuffle_value'],
                             default=['ridge', 'no_read', 'prototype', 'shuffle_value'])
        if name == 'infer':
            cmd.add_argument('--resume', help='Complete chunk-boundary rollout.pt checkpoint')
            cmd.add_argument('--case', required=True)
            cmd.add_argument('--mode', choices=['off', 'frozen', 'kv_ttt', 'noise_ttt',
                                                'sap_frozen', 'sap_online', 'sap_no_read',
                                                'sap_no_commit', 'sap_shuffle_text',
                                                'binding_off', 'binding_frozen',
                                                'binding_online'], default='noise_ttt')
            cmd.add_argument('--ablation', choices=['reset', 'no_protection', 'shuffle'])
        if name == 'evaluate':
            cmd.add_argument('--resume', action='store_true', help='Reuse verified complete cases; restart interrupted cases')
            cmd.add_argument('--cases', required=True, help='JSONL with id/image/prompt/camera/intrinsics/seed')
            cmd.add_argument('--ablations', action='store_true')
            cmd.add_argument('--video-metrics', action='store_true')
            cmd.add_argument('--modes', nargs='+', help='Run only these comparison modes')
        if name in {'check-cache', 'query-eval'}:
            cmd.add_argument('--fixture', required=True)
        if name == 'query-eval':
            cmd.add_argument('--histories', nargs='+', choices=['real', 'generated'], default=['real', 'generated'])
            cmd.add_argument('--modes', nargs='+', default=['off', 'frozen_kv', 'kv_ttt'])
            cmd.add_argument('--fixtures', nargs='*', default=[], help='Additional held-out test fixtures')
        if name == 'check-cache':
            cmd.add_argument('--diagnose', action='store_true', help='Record FP32 per-layer cache errors')
        if name in {'fixture', 'check-data'}:
            cmd.add_argument('--split', choices=['train', 'val', 'test'], default='val')
        if name == 'fixture':
            cmd.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'doctor':
        import torch
        print(json.dumps(dict(python=platform.python_version(), platform=platform.platform(),
            torch=torch.__version__, cuda=torch.version.cuda, available=torch.cuda.is_available(),
            dependencies={m: importlib.util.find_spec(m) is not None for m in
                          ('diffusers', 'transformers', 'pyrallis', 'triton', 'pytest', 'numpy')}), indent=2))
        return
    settings = json.loads(Path(args.settings).read_text(encoding='utf-8'))
    if args.command == 'grail-train':
        from .grail_train import train
        train(settings, args.output, args.adapter)
    elif args.command == 'grail-experiment':
        from .grail_experiment import run_experiment
        run_experiment(settings, args.adapter, args.output, split=args.split, samples=args.samples,
                       frames=args.frames, histories=args.histories, variants=args.variants,
                       case_file=args.cases, seeds=args.seeds)
    elif args.command == 'train':
        from .check_cache import require_gate
        from .train import train
        require_gate(settings)
        train(settings, args.output, args.adapter)
    elif args.command == 'infer':
        from .infer import infer
        infer(settings, json.loads(Path(args.case).read_text(encoding='utf-8')), args.output,
              args.mode, args.adapter, args.ablation, args.resume)
    elif args.command == 'evaluate':
        from .evaluate import evaluate
        evaluate(args.settings, args.cases, args.output, args.adapter, args.ablations, args.video_metrics, args.modes,
                 resume=args.resume)
    elif args.command == 'check-cache':
        from .check_cache import check_cache
        check_cache(settings, args.fixture, args.output, diagnose=args.diagnose)
    elif args.command == 'query-eval':
        from .evaluate import query_evaluate
        query_evaluate(settings, [args.fixture, *args.fixtures], args.adapter, args.output,
                       args.histories, args.modes)
    else:
        from .data import EpisodeDataset
        data = EpisodeDataset(settings['data'], settings['manifest'], args.split, frames=settings.get('frames', 10))
        if args.command == 'check-data':
            for i in range(len(data)):
                data[i]
            print(f'Validated {len(data)} episodes in {args.split}')
        else:
            import torch
            from .sana import load_config, make_fixture, make_pipeline
            pipe = make_pipeline(load_config(settings['sana_config']), settings['base_checkpoint'], 'cuda', training=True)
            fixture = make_fixture(data[args.index], pipe, 'cuda')
            torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in fixture.items()}, args.output)


if __name__ == '__main__':
    main()
