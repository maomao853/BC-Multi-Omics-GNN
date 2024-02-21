
python ./HiSig/prepare_input.py --ont ./hier.csv --sig ./data/survival.txt --out ./HGNN_survival
R -f ./HiSig/R/glmnet.R --args ./data/HGNN_survival_conn.txt ./data/HGNN_survival_signal.txt ./data/HGNN_survival_ms_impact 10
python ./HiSig/parse.py --ont_conn ./data/HGNN_survival_conn.txt --rout ./data/HGNN_survvial_ms_impact.impact-w-rand.tsv --terms ./data/terms.txt --genes ./data/genes.txt --signal ./data/HGNN_survival_signal.txt --out ./HGNN_survival_ms_impact_summary.tsv


python ./HiSig/prepare_input.py --ont ./hier.csv --sig ./data/mutation.txt --out ./HGNN_mutation
R -f ./HiSig/R/glmnet.R --args ./data/HGNN_mutation_conn.txt ./data/HGNN_mutation_signal.txt ./data/HGNN_mutation_ms_impact 10
python ./HiSig/parse.py --ont_conn ./data/HGNN_mutation_conn.txt --rout ./data/HGNN_mutation_ms_impact.impact-w-rand.tsv --terms ./data/terms.txt --genes ./data/genes.txt --signal ./data/HGNN_mutation_signal.txt --out ./HGNN_mutation_ms_impact_summary.tsv