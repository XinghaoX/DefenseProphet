# DefenseProphet Inference

DefenseProphet supports three inference entry points:

- `encoded`: run inference from precomputed protein embedding `.pkl` files.
- `fasta`: encode protein FASTA files, then run inference.
- `genome`: call Prodigal on genome FASTA files, encode predicted proteins, then run inference.

## Install

Create the Linux inference environment:

```bash
conda env create -f environment.yml
conda activate defenseprophet-infer
```

The environment uses Python 3.12, CUDA 12.4 PyTorch, Transformers, pandas, tqdm, safetensors, and Prodigal. The key Python package versions follow the model training environment while keeping training-only packages out of the inference environment.

The default PyTorch wheel is `torch==2.6.0+cu124`. If your Linux system requires a different CUDA runtime, update the PyTorch wheel and index URL before creating the environment.

Verify the installation:

```bash
python -c "import torch, transformers, pandas, tqdm, safetensors; from transformers import DebertaV2ForTokenClassification; print(torch.__version__, torch.cuda.is_available()); print(transformers.__version__)"
prodigal -v
```

## Model Weights

The model weights are hosted at [XinghaoX/DefenseProphet](https://huggingface.co/XinghaoX/DefenseProphet). From the project root, download them into the existing `models/` directories:

```bash
hf download XinghaoX/DefenseProphet \
  debert_binary_150M/model.safetensors \
  debert_multi_150M/model.safetensors \
  --local-dir models
```

After downloading, the model files should be arranged as follows:

```text
models/
  debert_binary_150M/
    config.json
    model.safetensors
  debert_multi_150M/
    config.json
    model.safetensors
```

The small `config.json` files are included in this repository. The downloaded `model.safetensors` files contain the model weights.

## Example Data

The `examples/` folder contains one GTDB example in the three input forms accepted by the inference script:

- `examples/GB_GCA_000008085.1_protein.pkl`: precomputed protein embeddings for `encoded` mode.
- `examples/GB_GCA_000008085.1_protein.faa`: protein FASTA for `fasta` mode.
- `examples/GCF_000008085.1_ASM808v1_genomic.fna`: genome FASTA for `genome` mode.

The repository includes `label_mapping_dicts.pkl` in the project root. It is used by default to convert multi-class prediction IDs to defense gene labels.

The `fasta` and `genome` modes use ESM2-150M protein embeddings. Provide the local ESM2-150M model directory with `--embedding-model`, for example `/path/to/esm2_t30_150M_UR50D`.

## Usage Examples

Infer from the precomputed example embedding file:

```bash
python infer_fasta.py \
  --mode encoded \
  --input examples/GB_GCA_000008085.1_protein.pkl \
  --output-csv-dir results/example_encoded
```

Encode the example protein FASTA, then infer:

```bash
python infer_fasta.py \
  --mode fasta \
  --input examples/GB_GCA_000008085.1_protein.faa \
  --embedding-model /path/to/esm2_t30_150M_UR50D \
  --embedding-cache-dir results/encoded_pkl/example_fasta \
  --output-csv-dir results/example_fasta
```

Predict proteins from the example genome FASTA with Prodigal, encode them, then infer:

```bash
python infer_fasta.py \
  --mode genome \
  --input examples/GCF_000008085.1_ASM808v1_genomic.fna \
  --embedding-model /path/to/esm2_t30_150M_UR50D \
  --prodigal-output-dir results/prodigal_output/example_genome \
  --embedding-cache-dir results/encoded_pkl/example_genome \
  --output-csv-dir results/example_genome
```

For `fasta` and `genome` modes, input files must contain FASTA-formatted sequences. File extensions are not used for filtering.

## Parameters

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `--mode` | No | `fasta` | Input workflow. Use `encoded` for existing `.pkl` embeddings, `fasta` for protein FASTA, or `genome` for genome FASTA through Prodigal. |
| `--input` | Yes | None | Input file or folder. In `encoded` mode, folders are filtered to `.pkl` files. In `fasta` and `genome` modes, files must contain FASTA-formatted sequences and extensions are not used for filtering. |
| `--embedding-model` | For `fasta` and `genome` | None | Local ESM2-150M model directory used by `AutoTokenizer` and `AutoModel` to encode protein sequences. |
| `--binary-model` | No | `./models/debert_binary_150M` | Binary token-classification model path. Predicts `other` or `defense`. |
| `--multi-model` | No | `./models/debert_multi_150M` | Multi-class token-classification model path. Predicts the defense gene class. |
| `--label-encoder` | No | `./label_mapping_dicts.pkl` | Pickle file containing `label2id` and `id2label`. Converts multi-class prediction IDs to defense gene labels. |
| `--embedding-cache-dir` | No | `./results/encoded_pkl` | Output/cache directory for encoded `.pkl` files generated from FASTA input. Existing `.pkl` files are reused. |
| `--prodigal-output-dir` | Only used by `genome` | `./results/prodigal_output` | Directory for Prodigal outputs: predicted protein `.faa`, nucleotide `.ffn`, and annotation `.gff` files. |
| `--prodigal-bin` | Only used by `genome` | `prodigal` | Prodigal executable name or full path. |
| `--prodigal-mode` | Only used by `genome` | `meta` | Prodigal mode. Use `meta` for metagenomic or draft genomes, `single` for complete single genomes. |
| `--output-csv-dir` | No | `./results/predictions` | Directory for prediction CSV files. |
| `--device` | No | `cuda:0` if available, else `cpu` | PyTorch device used for encoding and inference. |
| `--toks-per-batch` | No | `4096` | Approximate token budget per protein embedding batch. Lower this value if GPU memory is limited. |
| `--window-size` | No | `64` | Number of genes per inference window. Should match model training. |
| `--stride` | No | `32` | Sliding-window stride. Should match model training. |
| `--same-contig` | No | Off | Treat each input `.pkl` or FASTA-derived file as one contig instead of splitting by gene ID prefix. |

Each output CSV contains one row per gene with binary prediction, multi-class prediction, confidence scores, contig ID, gene ID, and description.
