#!/bin/bash

  for y in depmap
  do
    for c in {1..47}
    do
      echo -n "$y," >> hotspots.txt
      echo -n "$c," >> hotspots.txt
      python ./lasso/prepare_input_DepMap_cMap.py --ont ./hier.csv --y $y --c $c --out ./HGNN_depmap
      #python ./HiSig/prepare_input.py --ont ./results/all_output.txt --sig ./data/stouffer_z_brca.txt --out ./results/all
      #R -f ./HiSig/R/glmnet.R --args ./HGNN_mutation_conn.txt ./HGNN_mutation_signal.txt ./HGNN_mutation_ms_impact 10

      R -f ./HiSig/R/glmnet.R --args ./data/HGNN_depmap_conn.txt ./data/all_signal.txt ./data/all_ms_impact 50
      python ./HiSig/parse.py --ont_conn ./data/HGNN_depmap_conn.txt --rout ./data/all_ms_impact.impact-w-rand.tsv --terms ./data/terms.txt --genes ./data/genes.txt --signal ./data/all_signal.txt --out ./hotspots.tsv >> hotspots.txt
      echo ""  >> hotspots.txt
    done
  done
  for y in cmap
  do
    for c in {1..334}
    do
      #echo -n "$d," >> hotspots.txt
      echo -n "$y," >> hotspots.txt
      echo -n "$c," >> hotspots.txt
      
      python ./lasso/prepare_input_DepMap_cMap.py --ont ../hier.csv --y $y --c $c --out ./HGNN_cmap
      R -f ./HiSig/R/glmnet.R --args ./data/HGNN_cmap_conn.txt ./data/all_signal.txt ./data/all_ms_impact 50
      python ./HiSig/parse.py --ont_conn ./data/HGNN_cmap_conn.txt --rout ./data/all_ms_impact.impact-w-rand.tsv --terms ./data/terms.txt --genes ./data/genes.txt --signal ./data/all_signal.txt --out ./hotspots.tsv >> hotspots.txt
      echo ""  >> hotspots.txt
    done
  done


