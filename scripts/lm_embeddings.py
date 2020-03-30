from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pdf')
import seaborn as sns
import numpy as np
import glob


def load_pfam(fname, prots):
    pfam2uniprot = {}
    fRead = open(fname, 'r')
    fRead.readline()
    for line in fRead:
        splitted = line.strip().split()
        if len(splitted) == 4:
            uniprot = splitted[0]
            pfam_ids = splitted[3]
            pfam_ids = pfam_ids.split(';')[:-1]
            if uniprot in prots:
                for pfam_id in pfam_ids:
                    if pfam_id not in pfam2uniprot:
                        pfam2uniprot[pfam_id] = set()
                    pfam2uniprot[pfam_id].add(uniprot)
    fRead.close()

    return pfam2uniprot


def load_elmo_embed(prot2path, prots):
    X = np.zeros((len(prots), 512), dtype=float)
    for i, prot in enumerate(prots):
        embedd = np.load(prot2path[prot])
        X[i] = embedd['embed'].sum(axis=0)
        if i % 1000 == 0:
            print (i, '/', len(prots))
    return X


def load_attn_embed(prot2path, prots):
    X = np.zeros((len(prots), 768), dtype=float)
    for i, prot in enumerate(prots):
        embedd = np.load(prot2path[prot])
        embedd = embedd['embed'][1:-1]
        X[i] = embedd.sum(axis=0)
        if i % 1000 == 0:
            print (i, '/', len(prots))
    return X


def load_npz(path):
    fn = glob.glob(path)
    prot2path = {f.split('/')[-1].split('.')[0]: f for f in fn}
    return prot2path

if __name__ == "__main__":
    elmo_prot2path = load_npz('/mnt/ceph/users/jmorton/icml-final-submission/results/embeddings/elmo/epoch5/*.npz')
    attn_prot2path = load_npz('/mnt/ceph/users/jmorton/icml-final-submission/results/embeddings/attn/epochuniref90/*.npz')
    pfam2uniprot = load_pfam('/mnt/ceph/users/vgligorijevic/Jamie-attention/Pfam_benchmarking/uniprot-filtered-reviewed_yes.tab', set(elmo_prot2path.keys()))

    uniprot_list = []
    classes = []
    for pfam in pfam2uniprot:
        family = list(pfam2uniprot[pfam])
        if len(family) >= 100 and len(family) < 105:
            for prot in family:
                if prot not in uniprot_list:
                    uniprot_list.append(prot)
                    classes.append(pfam)

    print ("### Number of classes = ", len(set(classes)))
    print ("### Number of proteins = ", len(uniprot_list))
    print ("\n\n")


    print ("### Extracting ELMO embeddings....")
    X_elmo = load_elmo_embed(elmo_prot2path, uniprot_list)
    np.savez_compressed('elmo_embedd.npz', X=X_elmo)
    print ("\n\n")
    print ("### Extracting ROBERTA embeddings....")
    X_attn = load_attn_embed(attn_prot2path, uniprot_list)
    np.savez_compressed('roberta_embedd.npz', X=X_attn)
    print ("\n\n")

    #DIR = '/mnt/ceph/users/vgligorijevic/Jamie-attention/attention-alignment/scripts/'
    X_elmo = np.load(f'elmo_embedd.npz')
    X_elmo = X_elmo['X']
    X_attn = np.load(f'roberta_embedd.npz')
    X_attn = X_attn['X']

    print ("### tSNE for ELMO...")
    X_elmo_tsne = TSNE(n_components=2).fit_transform(X_elmo)

    print ("### tSNE for BERT...")
    X_attn_tsne = TSNE(n_components=2).fit_transform(X_attn)


    plt.figure(figsize=(10, 5))
    plt.subplot(121, aspect='equal', adjustable='box')
    sns.scatterplot(x=X_elmo_tsne[:, 0], y=X_elmo_tsne[:, 1], hue=classes, legend=False, s=15)
    plt.xlabel('tSNE-1', fontsize=14)
    plt.ylabel('tSNE-2', fontsize=14)
    plt.title('ELMo', fontsize=14)
    plt.grid(linestyle=':', alpha=0.5)

    plt.subplot(122, aspect='equal', adjustable='box')
    g = sns.scatterplot(x=X_attn_tsne[:, 0], y=X_attn_tsne[:, 1], hue=classes, s=15)
    plt.xlabel('tSNE-1', fontsize=14)
    plt.ylabel('tSNE-2', fontsize=14)
    plt.title('RoBERTa', fontsize=14)
    plt.grid(linestyle=':', alpha=0.5)
    g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), fontsize=8, ncol=1)
    plt.tight_layout()
    plt.savefig('pfam_embedding.pdf', dpi=300)
