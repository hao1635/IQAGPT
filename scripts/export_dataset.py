"""Build a deterministic CT-IQA archive without local paths or DICOM metadata."""
import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


def add_bytes(archive, name, data):
    info = tarfile.TarInfo('CT-IQA/' + name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    archive.addfile(info, io.BytesIO(data))


def export(root, output, data_card):
    root, output = Path(root), Path(output)
    files = {name: root / name for name in ['iqagpt_caption.json', 'labels.json']}
    # Preserve the alternate labels under their original name; do not silently merge them.
    for name in ['iqagpt_labels.json', 'report3.json']:
        if (root / name).exists():
            files['annotation_versions/' + name] = root / name
    captions = json.loads((root / 'iqagpt_caption.json').read_text())
    for split, count in [('train', 800), ('test', 200)]:
        if len(captions[split]) != count:
            raise ValueError('Unexpected dataset size')
        for row in captions[split]:
            image_id = row['image_id']
            if Path(image_id).name != image_id or not image_id.startswith(split + '-'):
                raise ValueError('Invalid image ID')
            relative = split + '/npy/' + image_id + '.npy'
            files[relative] = root / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with output.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as archive:
                for relative, path in sorted(files.items()):
                    data = path.read_bytes()
                    manifest.append(hashlib.sha256(data).hexdigest() + '  ' + relative)
                    add_bytes(archive, relative, data)
                add_bytes(archive, 'DATASET_CARD.md', Path(data_card).read_bytes())
                add_bytes(archive, 'checksums.sha256', ('\n'.join(manifest) + '\n').encode())
    checksum = hashlib.sha256()
    with output.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    digest = checksum.hexdigest()
    output.with_suffix(output.suffix + '.sha256').write_text(digest + '  ' + output.name + '\n')
    print(json.dumps({'file': output.name, 'bytes': output.stat().st_size, 'sha256': digest, 'source_files': len(files)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--data-card', default='docs/DATASET_CARD.md')
    args = p.parse_args()
    export(args.data_root, args.output, args.data_card)
