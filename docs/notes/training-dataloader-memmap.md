# Training dataloader: memmap, not np.load

`MultipvDataset` uses memory-mapped `.npy` files now; required because
merged datasets (~145 GB uncompressed) don't fit in any sensible GPU
instance's RAM.

`experiments/distill-soft/src/distill_soft/train_supervised.py::MultipvDataset`
used to `np.load(npz)[k]` for each array, materializing them fully in
RAM. That worked for the 4000-game datasets the trainer was originally
written for (~250 MB) but **OOMs** on the salvage-era datasets:

| Dataset | Positions | `states.npy` uncompressed | Fits in g6/g5/g4dn (16 GB RAM)? |
|---|---|---|---|
| us-east d15 | 5.07M | ~25 GB | NO |
| eu d10 | 30.66M | ~145 GB | NO (also doesn't fit on a 256 GB box) |

**Fix (May 2026):** the loader now defaults to `mmap=True`. On first
construction:

1. `_extract_npz_to_memmap_dir(npz_path, npz_path + "_extracted")`
   streams the `.npz` (a zip of `.npy` files) to a directory of plain
   `.npy` files, using `zipfile.ZipFile.open()` + chunked read. Peak
   memory is the deflate buffer (a few MB), not the array.
2. Each array is then `np.load(<npy>, mmap_mode="r")`. Only the
   batch-sized slice the trainer touches is paged in by the OS.

The `mmap=False` path is preserved for tiny test fixtures.

**Disk requirement** for the extracted files:

- us-east d15: ~25 GB → `cluster-train-us.yaml` `volumeSize: 100` is fine.
- eu d10: ~145 GB → `cluster-train-eu.yaml` `volumeSize: 250` (300 with margin).
- Experiment C (full 30M): bump to 300 GB on the bare-EC2 launcher's volume.

**Why:** The merged datasets are 100–600× bigger than what the
original loader was designed for. Loading them fully would require a
GPU instance with 150+ GB RAM, which doesn't exist outside the largest
x2/r5 families. Memmapping is the right architecture regardless —
sequential access through the dataset epoch-by-epoch is exactly the
access pattern OS page caches handle well.

**How to apply:** If anyone reports OOM during data loading in a
training run, the diagnosis is almost certainly that something
downstream (a third-party tool, a hand-written loader, etc.) bypassed
`MultipvDataset` and called `np.load(npz)` directly. Either route them
through the dataset class or apply the same
`_extract_npz_to_memmap_dir` + `mmap_mode='r'` pattern.
