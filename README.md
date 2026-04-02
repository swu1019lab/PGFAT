# PGFAT: Pan-Gene Family Analysis Toolkit

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-BSD--3--Clause-green)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
[![DOI](https://zenodo.org/badge/1157870577.svg)](https://doi.org/10.5281/zenodo.18696454)

---

## 🌟 Key Features

*   **🔍 Comprehensive Identification**: Detect gene family members using **HMM**, **BLAST**, or a hybrid approach across multiple genomes.
*   **📊 Pan-Genome Classification**: Automatically classify members into **Core**, **Softcore**, **Shell**, and **Specific** based on presence/absence variation (PAV).
*   **🧬 Evolutionary Insights**:
    *   **Ka/Ks Analysis**: Quantify selection pressure differentiation between core and non-core members.
    *   **TE Landscape**: Analyze Transposable Element insertion densities around gene bodies.
    *   **Duplication History**: Distinguish between ancient Whole Genome Duplication (WGD) and recent tandem duplications.
*   **🌱 Phenotype Association**: Link genetic variations (CNV & PAV) to agronomic traits using statistical methods.
*   **📈 Rich Visualization**: Generate publication-ready figures including phylogenetic trees, heatmaps, volcano plots, and evolutionary rate distributions.

---

## 🛠️ Installation & Setup

PGFAT is distributed as a compiled binary extension for performance and security. No standard package installation is required.

1. **Download the Release**:
   Extract the provided PGFAT release archive and navigate into the directory.
   ```bash
   git clone https://github.com/swu1019lab/PGFAT.git
   cd PGFAT
   ```

2. **Install Python Dependencies**:
   Ensure you have Python 3.8+ installed, then install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Usage**:
   ```bash
   python run_pgfat.py -h
   ```

**Dependencies:**
- Python 3.8+
- pandas, numpy, scipy
- matplotlib
- biopython
- tqdm, tomli

**External Tools (must be in PATH):**
- hmmsearch (HMMER)
- mafft
- orthofinder
- meme
- yn00 (PAML)
- pal2nal.pl
- diamond

---

## 🚀 Quick Start

### 1. Basic Gene Family Search

Identify members of the "MYB" family across genomes using an HMM profile:

```bash
python run_pgfat.py -i ./proteins_dir -o ./output_dir -f MYB myb.hmm
```

*   `./proteins_dir`: Directory containing one protein FASTA file per genome (e.g., `GenomeA.pep`, `GenomeB.pep`).
*   `./output_dir`: Directory where results and figures will be saved.
*   `myb.hmm`: HMM profile for the gene family (from Pfam etc.).

### 2. Run with bundled example data (CIPK)

Use the built-in example dataset to quickly test PGFAT with the CIPK gene family:

```bash
python run_pgfat.py -i example/data/proteins -o results_cipk -f CIPK example/data/PK-PF00069.hmm,example/data/CIPK-PF03822.hmm
```

This command uses two HMM profiles (comma-separated) for stricter family identification.

### 3. Advanced Analysis with Evolutionary & Phenotypic Data

Enable Ka/Ks calculation, TE analysis, Phlyogeny and Phenotype association:

```bash
python run_pgfat.py -i ./proteins \
      -o ./results \
      -f MYB myb.hmm \
      --cds ./cds_sequences \
      --loc ./gene_locations \
      --tree species_tree.nwk \
      --phe traits.csv \
      --config config.toml \
      --threads 8
```

---

## 📖 Usage Guide

### Command Line Arguments

| Argument | Description | Required |
| :--- | :--- | :--- |
| `-i`, `--input` | Directory containing protein sequences (one `.fa`/`.pep` file per genome). | **Yes** |
| `-o`, `--output` | Main output directory. | **Yes** |
| `-f`, `--family` | Evaluation pairs: `Name File` (e.g., `-f MYB myb.hmm`). Can be used multiple times. | **Yes** |
| `-t`, `--threads` | Number of CPU threads to use (default: 4). | No |
| `-c`, `--config` | Path to `config.toml` for advanced parameters and plotting settings. | No |
| `--method` | Search method: `hmm` (default), `blast`, `hmm_blast`, or `blast_hmm`. | No |
| `--cds` | Directory containing CDS sequences (required for **Ka/Ks** analysis). | No |
| `--loc` | Directory containing gene location files (BED6 format) (required for **TE/Synteny**). | No |
| `--tree` | Path to species phylogenetic tree (Newick format). | No |
| `--phe` | Phenotype data in CSV format (Columns: `Genome,Trait1,Trait2...`). | No |
| `--force` | Force overwrite of existing results. | No |

### Configuration (`config.toml`)

The `config.toml` file controls detailed analysis parameters and visualization aesthetics.

**Example Snippet:**
```toml
[family.search]
hmm_evalue = 1e-5

[analysis.pheno_assoc]
cnv_assoc = true
pav_assoc = true
cnv_method = "spearman"

[plot.volcano]
figsize = [4, 4]
colors = {CNV = '#E64B35', PAV = '#4DBBD5'}
```
*See the provided `config.toml` template for all available options.*

---

## 📂 Input File Formats

1.  **Protein Directory (`-i`)**:
    *   `Genome1.pep`, `Genome2.pep`, ...
2.  **CDS Directory (`--cds`)** (Optional):
    *   Expected if calculating Ka/Ks. Must match protein IDs.
3.  **Location Directory (`--loc`)** (Optional):
    *   Files (e.g., `.bed`) containing gene coordinates.
    *   Format (BED6): `chrom start end gene_id score strand`
4.  **Phenotype File (`--phe`)** (Optional):
    *   CSV format:
        ```csv
        Genome,Yield,Height
        Genome1,500,120
        Genome2,450,115
        ```

---

## 📄 License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

---

## ✉️ Contact

*   **Author**: Xiaodong Li
*   **Email**: lxd1997xy@163.com
