
# Installation
To get started running this code run within your environment.
```
pip install -r requirements.txt
conda create alignment -f env.yaml
```
A virtualenv is recommended for installation.

In all of the following scripts, the data paths may have to be altered, in particular the `DATADIR` environmental variable.  See scripts below


# Training
Scripts to train roberta and elmo are found under the train folder.  The file paths within these scripts will have to be altered according to the specifications in the local file system.
Note that the roberta download will require one to provide formatted data directories in order to extract the embeddings.  Scripts to format the input datasets can be found under `train/roberta-train`.  Note that this will require the download of the uniref90 database and the pfam database via

```
wget ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz
wget ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam30.0/Pfam-A.full.gz
```

# Preprocessing
The following scripts will extract embeddings from the pretrained models.

```
sbatch blast.sh
sbatch extract_elmo.sh
sbatch extract_attention.sh
```


# Domain prediction

The domain prediction (i.e residue similarity) scripts can be run as follows.
```
sbatch elmo_domain.sh
sbatch attn_domain.sh
```

# Protein distance

The protein similarity benchmarks can be run as follows
```
sbatch elmo_dist.sh
sbatch attn_dist.sh
```

Once the protein distance and domain prediction benchmarks are done, all of the results can be
directly read out through

```
sh score_distances.sh
```

# Alignment
The alignment scripts can be run as follows.

```
sbatch elmo_align.sh
sbatch attn_align.sh
```

Once the alignments have been completed, the results can be read out as follows

```
sh score_alignments.sh
```

# Analysis notebooks

The notebooks to generate the figures can be found under the `ipynb` folder.
`blast-distance-evaluation.ipynb` evaluates the protein distances from blast.
`pairwise-blast-domain-format.ipynb` formats the blast output to a format readable for the alignment benchmark.
`pairwise-domain-benchmark.ipynb` is the alignment benchmark.

# Analysis scripts
The tsne plot was generated under `scripts/lm_embeddings.py`.  The file paths may have to change according to the local file paths set above.