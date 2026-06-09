import argparse
import os
import pickle as pk
import shutil
import subprocess


PKL_EXTENSION = ".pkl"
EXPECTED_EMBEDDING_SIZE = 640


def collect_input_files(input_path, extension=None):
    if os.path.isfile(input_path):
        return [input_path]

    input_files = []
    for name in os.listdir(input_path):
        path = os.path.join(input_path, name)
        if not os.path.isfile(path):
            continue
        if extension is None or name.lower().endswith(extension):
            input_files.append(path)
    return sorted(input_files)


def load_id2label(label_encoder_path):
    with open(label_encoder_path, "rb") as f:
        label_encoder = pk.load(f)
    return label_encoder["id2label"]


def run_prodigal(genome_files, output_dir, prodigal_bin, prodigal_mode):
    if shutil.which(prodigal_bin) is None:
        raise FileNotFoundError(
            f"Prodigal executable not found: {prodigal_bin}. "
            "Install Prodigal or pass its full path with --prodigal-bin."
        )

    os.makedirs(output_dir, exist_ok=True)

    protein_files = []
    for genome_path in genome_files:
        base_name = os.path.splitext(os.path.basename(genome_path))[0]
        protein_path = os.path.join(output_dir, f"{base_name}.faa")
        nucleotide_path = os.path.join(output_dir, f"{base_name}.ffn")
        annotation_path = os.path.join(output_dir, f"{base_name}.gff")

        if not os.path.exists(protein_path):
            cmd = [
                prodigal_bin,
                "-i",
                genome_path,
                "-a",
                protein_path,
                "-d",
                nucleotide_path,
                "-o",
                annotation_path,
                "-p",
                prodigal_mode,
                "-q",
            ]
            subprocess.run(cmd, check=True)

        protein_files.append(protein_path)

    return protein_files


def encode_fastas(fasta_files, embedding_model_path, output_dir, device, toks_per_batch):
    if embedding_model_path is None:
        raise ValueError("--embedding-model is required for fasta/genome modes and must point to ESM2-150M.")

    from transformers import AutoModel, AutoTokenizer

    from src.infer import encode_fasta_to_pkl_dynamic

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
    embedding_model = AutoModel.from_pretrained(embedding_model_path).to(device)
    hidden_size = getattr(embedding_model.config, "hidden_size", None)
    if hidden_size != EXPECTED_EMBEDDING_SIZE:
        raise ValueError(
            "DefenseProphet expects ESM2-150M protein embeddings with "
            f"hidden_size={EXPECTED_EMBEDDING_SIZE}, but {embedding_model_path} "
            f"has hidden_size={hidden_size}."
        )
    embedding_model.eval()

    pkl_files = []
    for fasta_path in fasta_files:
        base_name = os.path.splitext(os.path.basename(fasta_path))[0]
        output_pkl_path = os.path.join(output_dir, f"{base_name}.pkl")
        pkl_path = encode_fasta_to_pkl_dynamic(
            fasta_path=fasta_path,
            output_pkl_path=output_pkl_path,
            model=embedding_model,
            tokenizer=tokenizer,
            device=device,
            toks_per_batch=toks_per_batch,
        )
        if pkl_path is not None:
            pkl_files.append(pkl_path)

    if not pkl_files:
        raise RuntimeError("FASTA encoding failed for all inputs. Check error_log.txt for details.")

    return pkl_files


def main():
    parser = argparse.ArgumentParser(
        description="Run DefenseProphet inference from encoded pkl, protein FASTA, or genome FASTA input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["encoded", "fasta", "genome"],
        default="fasta",
        help=(
            "encoded: infer from existing pkl embeddings; "
            "fasta: encode protein FASTA then infer; "
            "genome: call Prodigal, encode predicted proteins, then infer."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input file or folder. For fasta/genome modes, files must be FASTA-format; "
            "file extensions are not used for filtering."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Required for fasta/genome modes. ESM2-150M model path/name for protein encoding.",
    )
    parser.add_argument("--binary-model", default="./models/debert_binary_150M")
    parser.add_argument("--multi-model", default="./models/debert_multi_150M")
    parser.add_argument("--label-encoder", default="./label_mapping_dicts.pkl", help="Label mapping pickle path.")
    parser.add_argument("--embedding-cache-dir", default="./results/encoded_pkl")
    parser.add_argument("--prodigal-output-dir", default="./results/prodigal_output")
    parser.add_argument("--prodigal-bin", default="prodigal")
    parser.add_argument("--prodigal-mode", choices=["single", "meta"], default="meta")
    parser.add_argument("--output-csv-dir", default="./results/predictions")
    parser.add_argument("--device", default=None, help="Defaults to cuda:0 when available, otherwise cpu.")
    parser.add_argument("--toks-per-batch", type=int, default=4096)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument(
        "--same-contig",
        action="store_true",
        help="Treat each FASTA file as one contig instead of splitting by gene_id prefix.",
    )
    args = parser.parse_args()

    import torch
    from transformers import DebertaV2ForTokenClassification

    from src.infer import run_pipeline

    device_name = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    if args.mode == "encoded":
        pkl_files = collect_input_files(args.input, extension=PKL_EXTENSION)
        if not pkl_files:
            raise FileNotFoundError(f"No encoded .pkl files found under: {args.input}")
    else:
        input_files = collect_input_files(args.input)
        if not input_files:
            raise FileNotFoundError(f"No input files found under: {args.input}")

        if args.mode == "genome":
            fasta_files = run_prodigal(
                genome_files=input_files,
                output_dir=args.prodigal_output_dir,
                prodigal_bin=args.prodigal_bin,
                prodigal_mode=args.prodigal_mode,
            )
        else:
            fasta_files = input_files

        pkl_files = encode_fastas(
            fasta_files=fasta_files,
            embedding_model_path=args.embedding_model,
            output_dir=args.embedding_cache_dir,
            device=device,
            toks_per_batch=args.toks_per_batch,
        )

    binary_model = DebertaV2ForTokenClassification.from_pretrained(args.binary_model)
    multi_model = DebertaV2ForTokenClassification.from_pretrained(args.multi_model)
    id2label = load_id2label(args.label_encoder)

    run_pipeline(
        files_to_process=pkl_files,
        binary_model=binary_model,
        multi_model=multi_model,
        output_csv_dir=args.output_csv_dir,
        id2label=id2label,
        binary_id2label={0: "other", 1: "defense"},
        window_size=args.window_size,
        stride=args.stride,
        device=str(device),
        is_same_contig=args.same_contig,
    )


if __name__ == "__main__":
    main()
