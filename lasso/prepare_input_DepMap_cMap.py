import pandas as pd
import numpy as np
import argparse
import os

def prepare_input(ont, y, c, outf_conn):
    '''
    
    :param ont: 
    :param sig: 
    :param outf_conn: 
    :param outf_sig: 
    :param file_exist: 
    :return: 
    '''
    df = pd.read_csv(ont, ',', header=None,comment='#')
    df[0] = df[0].astype(str)
    df[2] = df[2] + len(pd.unique(df[1]))
    df[3] = df[3] + len(pd.unique(df[1])) + len(pd.unique(df[2]))
    df[4] = df[4] + len(pd.unique(df[1])) + len(pd.unique(df[2])) + len(pd.unique(df[3]))
    df[5] = df[5] + len(pd.unique(df[1])) + len(pd.unique(df[2])) + len(pd.unique(df[3])) + len(pd.unique(df[4]))

    df1 = df[[1,0]]
    df1['term']="gene"
    df1.columns = [0,1,'term']

    df2= df[[2,1]]
    df2['term']="default"
    df2.columns = [0,1,'term']

    df3= df[[3,2]]
    df3['term']="default"
    df3.columns = [0,1,'term']

    df4= df[[4,3]]
    df4['term']="default"
    df4.columns = [0,1,'term']

    df5= df[[5,4]]
    df5['term']="default"
    df5.columns = [0,1,'term']

    df = pd.concat([df1,df2,df3,df4,df5])

    df.to_csv('ont.csv', index=False)

    df_terms = df.loc[df['term'] != 'gene', :]
    df_genes = df.loc[df['term'] == 'gene', :]

    genes = sorted(set(df_genes[1].tolist()))
    terms = sorted(set(df[0].tolist() + df_terms[1].tolist()))
    

    genes_idx = {genes[i]:i for i in range(len(genes))}
    terms_idx = {terms[i]:i for i in range(len(terms))}


    mat_terms = np.eye(len(terms))
    mat_gene2term = np.zeros((len(genes), len(terms)))
    for i, row in df_terms.iterrows():
        mat_terms[terms_idx[row[1]], terms_idx[row[0]]] = 1
    for i, row in df_genes.iterrows():
        mat_gene2term[genes_idx[row[1]], terms_idx[row[0]]] = 1

    # propagate
    while True:
        mat_gene2term_new = np.dot(mat_gene2term, mat_terms)
        if np.sum(mat_gene2term_new > 0) == np.sum(mat_gene2term > 0):
            mat_gene2term = mat_gene2term_new
            break
        else:
            mat_gene2term = mat_gene2term_new

    with open(outf_conn, 'w') as ofh:
        for i in range(len(genes)):
            ofh.write('{}\t{}\n'.format(i, i))
        row, col = np.where(mat_gene2term > 0)
        for i in range(len(row)):
            ofh.write('{}\t{}\n'.format(row[i], len(genes) + col[i]))

    print("len(genes):",len(genes))

    with open(os.path.join(os.getcwd(), 'data', 'genes.txt'), 'w') as fh:
        for i in range(len(genes)):
            fh.write(genes[i] + '\n')

    terms = [str(x) for x in terms]
    print("len(terms):",len(terms))
    print("terms:",terms)

    with open(os.path.join(os.getcwd(), 'data', 'terms.txt'), 'w') as fh:
        for i in range(len(terms)):
            fh.write(terms[i] + '\n')
  
    signal_all = pd.read_csv(os.path.join(os.getcwd(), 'data', 'DepMap_cMap', f'{y}.csv'),index_col=0)
    #print(signal_all)
    signal = {}
    with open(os.path.join(os.getcwd(), 'data', 'DepMap_cMap', 'all_signal.txt'), 'w') as ofh:
        c = int(c)
        #print("c:",c)
        for j in range(signal_all.shape[0]):
            g = signal_all.index[j]
            #print("g:",g)
            s = signal_all.iloc[j,c]
            s = float(s)
            #print("s:",s)
            signal[g] = s
        #print(signal)
        for g in genes:
            if g in signal:
                ofh.write('{}\n'.format(signal[g]))
            else:
                ofh.write('0.0\n')

if __name__ == "__main__":
    par = argparse.ArgumentParser()
    par.add_argument('--ont', required=True, help = 'an ontology file')
    par.add_argument('--y', required=True, help = 'a text file for the signal on gene (leaf nodes)')
    par.add_argument('--out', required=True, help = 'output prefix')
    args = par.parse_args()

    prepare_input(args.ont, args.y, args.c, args.out+ '_conn.txt')
