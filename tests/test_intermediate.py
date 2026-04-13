import pandas as pd
from indra_cogex.client import Neo4jClient
from indra_cogex.analysis.intermediate_pathway_analysis import intermediate_pathway_analysis
from indra_cogex.analysis.intermediate_pathway_analysis import read_gene_set, parse_gene_list

client = Neo4jClient()

upstream_df = pd.read_csv("resources/kinomescan_test.csv")
downstream_df = pd.read_csv("resources/phosphoproteomics_significant_gene_list.csv")

print(upstream_df.head())
print(downstream_df.head())

result = intermediate_pathway_analysis(
    client=client,
    upstream_df=upstream_df,
    downstream_df=downstream_df,
    upstream_gene_col="kinase",
    upstream_metric_col="pct_control_1000",
    downstream_gene_col="GeneSymbol",
    downstream_metric_col="log2FC",
    upstream_relationship_types=["Phosphorylation"],
    downstream_relationship_types=["Activation", "Inhibition", "IncreaseAmount", "DecreaseAmount"],
    minimum_belief=0.5,
    minimum_evidence_count=1,
)

print("\nUpstream scores sample:")
upstream_genes, upstream_scores = read_gene_set(
    df=upstream_df,
    gene_col="kinase",
    metric_col="pct_control_1000",
)
print(list(upstream_scores.items())[:5])
print("\nSample upstream gene from pathways:")
print(result['pathways']['upstream_gene'].head())

print(f"Intermediates: {len(result['intermediates'])}")
print(result['intermediates'].head(10))

print(f"\nPathways: {len(result['pathways'])}")
print(result['pathways'].head(10))

print(f"\nNetwork: {result['network'].number_of_nodes()} nodes, {result['network'].number_of_edges()} edges")