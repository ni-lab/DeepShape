#!/usr/bin/env python3
"""Run public pretrained DeepShape inference, DeepLIFT, and an optional training step."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REVISION = '8fb6e29695b9728f45e8f97c74ca59197f9cf589'
CHECKPOINT_URL = 'https://huggingface.co/ryankeivanfar/DeepShape/resolve/' + REVISION + '/deepshape.pth.tar'
CHECKPOINT_SHA256 = '663b7ae1df67f9497fbed6dad9670b056722a2fdd1f2162520f2825fba11c738'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(destination):
    destination = Path(destination)
    if destination.exists():
        if sha256(destination) != CHECKPOINT_SHA256:
            raise ValueError('Checkpoint checksum differs from the published weights: ' + str(destination))
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as target:
            temporary = Path(target.name)
            with urllib.request.urlopen(CHECKPOINT_URL, timeout=60) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    target.write(chunk)
        if sha256(temporary) != CHECKPOINT_SHA256:
            raise ValueError('Downloaded checkpoint failed SHA-256 verification.')
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def load_model_class():
    path = ROOT / 'model/deepshape_no_normalize.py'
    spec = importlib.util.spec_from_file_location('deepshape_inference', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DeepShape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'downloads/deepshape.pth.tar')
    parser.add_argument('--output', type=Path, default=ROOT/'quickstart-output')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--skip-attributions', action='store_true')
    parser.add_argument('--train-step', action='store_true', help='Also verify one CPU optimizer step with a one-target demonstration model.')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(1337)
    fixture = ROOT/'examples/quickstart'
    metadata = json.loads((fixture/'manifest.json').read_text())
    for name, expected in metadata['files_sha256'].items():
        if sha256(fixture/name) != expected:
            raise ValueError('Example file checksum mismatch: ' + name)
    features = (ROOT/'data/distinct_features.txt').read_text().splitlines()
    if len(features) != 919 or sha256(ROOT/'data/distinct_features.txt') != metadata['features_sha256']:
        raise ValueError('The example requires the published, ordered 919-feature list.')
    with np.load(fixture/'inputs.npz') as data:
        inputs = data['encodings']
        references = data['reference_encodings']
        labels = data['labels']
    assert inputs.shape == (16, 9, 1000) and np.isfinite(inputs).all()
    checkpoint = download_checkpoint(args.checkpoint)
    weights = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
    weights = {name[7:] if name.startswith('module.') else name: value for name,value in weights.items()}
    model = load_model_class()(sequence_length=1000, n_targets=919)
    model.load_state_dict(weights, strict=True)
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(inputs)).numpy()
    assert predictions.shape == (16, 919) and np.isfinite(predictions).all()
    assert np.all((predictions >= 0) & (predictions <= 1))
    reference = np.load(fixture/'reference_predictions.npy')
    np.testing.assert_allclose(predictions, reference, rtol=2e-4, atol=2e-5)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output/'predictions.npy', predictions)
    result = dict(python=__import__('platform').python_version(), torch=torch.__version__, numpy=np.__version__,
                  device='cpu', checkpoint_sha256=sha256(checkpoint), prediction_shape=list(predictions.shape),
                  reference_prediction_max_absolute_error=float(np.max(np.abs(predictions-reference))),
                  target_index=metadata['target_index'], target_name=metadata['target_name'],
                  channel_order=metadata['channels'])
    if not args.skip_attributions:
        from captum.attr import DeepLiftShap
        explanation = DeepLiftShap(model)
        arrays = []
        for sequence in inputs[:2]:
            attribution = explanation.attribute(torch.from_numpy(sequence[None]),
                                                torch.from_numpy(references), target=metadata['target_index'])
            arrays.append(attribution.detach().numpy()[0])
        attributions = np.stack(arrays)
        assert attributions.shape == (2,9,1000) and np.isfinite(attributions).all()
        # Compatible with interpret/run_tfmodisco.py.
        np.savez_compressed(args.output/'encodings.npz', inputs[:2])
        np.savez_compressed(args.output/'attributions.npz', attributions)
        result['attribution_shape'] = list(attributions.shape)
    if args.train_step:
        training_model = load_model_class()(sequence_length=1000, n_targets=1)
        training_model.train()
        optimizer = torch.optim.SGD(training_model.parameters(), lr=.08, momentum=.9, weight_decay=1e-6)
        chosen = [0,1,8,9]
        y = torch.from_numpy(labels[chosen, metadata['target_index'], None])
        assert y.sum() == 2
        before = training_model.classifier[3].weight.detach().clone()
        optimizer.zero_grad()
        output = training_model(torch.from_numpy(inputs[chosen]))
        loss = torch.nn.functional.binary_cross_entropy(output, y)
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in training_model.parameters() if p.grad is not None)
        optimizer.step()
        assert not torch.equal(before, training_model.classifier[3].weight)
        result['training_step_loss'] = float(loss.detach())
        result['training_step_scope'] = 'One CPU optimizer step on example inputs; not benchmark training.'
    (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
