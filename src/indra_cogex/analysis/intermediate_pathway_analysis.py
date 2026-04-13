"""Intermediate pathway analysis connecting two gene sets through
enriched intermediate regulators using INDRA CoGEx."""

from typing import Collection, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from indra_cogex.client.enrichment.discrete import indra_intermediate_ora
from indra_cogex.client.enrichment.utils import (
    get_entity_to_regulators,
    get_entity_to_targets,
)
from indra_cogex.client.neo4j_client import Neo4jClient, autoclient
from indra_cogex.analysis.gene_analysis import parse_gene_list

def read_gene_set(
    df: pd.DataFrame,
    gene_col: str,
    metric_col: str,
) -> Tuple[List[str], Dict[str, float]]:
    """Read a gene set DataFrame and return gene list and scaled metric scores.

    Groups by gene name, takes the minimum metric value per gene, and
    scales to [0, 1] using min-max normalization.

    Parameters
    ----------
    df :
        DataFrame containing gene set data.
    gene_col :
        Column name for gene symbols.
    metric_col :
        Column name for the ranking metric values.

    Returns
    -------
    :
        A tuple of a list of gene symbols and a dict mapping gene
        symbols to scaled metric scores in [0, 1].
    """

    gene_effects = df.groupby(gene_col).agg({metric_col: "min"}).reset_index()
    gene_effects.columns = ["gene", "metric"]

    scaler = MinMaxScaler()
    gene_effects["scaled_metric"] = scaler.fit_transform(gene_effects[["metric"]].abs())

    gene_set, _ = parse_gene_list(gene_effects["gene"].tolist())
    
    gene_list = list(gene_set.values())
    scores = {
        resolved_name: gene_effects.loc[
            gene_effects["gene"] == original_name, "scaled_metric"
        ].values[0]
        for original_name, resolved_name in zip(
            gene_effects["gene"].tolist(), gene_set.values()
        )
    }
    return gene_list, scores

def assemble_pathways(
    upstream_genes: List[str],
    downstream_genes: List[str],
    intermediates_df: pd.DataFrame,
    entity_to_regulators: dict,
    entity_to_targets: dict,
    upstream_scores: Dict[str, float],
    downstream_scores: Dict[str, float],
) -> pd.DataFrame:
    """Assemble and score three-step paths through enriched intermediates.

    Parameters
    ----------
    upstream_genes :
        List of upstream gene symbols.
    downstream_genes :
        List of downstream gene symbols.
    intermediates_df :
        DataFrame of enriched intermediates from indra_intermediate_ora.
    entity_to_regulators :
        Mapping from get_entity_to_regulators.
    entity_to_targets :
        Mapping from get_entity_to_targets.
    upstream_scores :
        Dict mapping upstream gene symbols to scaled scores in [0, 1].
    downstream_scores :
        Dict mapping downstream gene symbols to scaled scores in [0, 1].

    Returns
    -------
    :
        DataFrame of scored three-step paths.
    """
    upstream_gene_set, _ = parse_gene_list(upstream_genes)
    upstream_hgnc_ids = set(upstream_gene_set.keys())
    downstream_gene_set, _ = parse_gene_list(downstream_genes)
    downstream_hgnc_ids = set(downstream_gene_set.keys())

    pathways = []
    for _, row in intermediates_df.iterrows():
        intermediate_curie = row["curie"]
        intermediate_name = row["name"]
        intermediate_regulators = entity_to_regulators.get((intermediate_curie,intermediate_name), set())
        intermediate_targets = entity_to_targets.get((intermediate_curie,intermediate_name), set())
        genes_upstream = upstream_hgnc_ids.intersection(intermediate_regulators)
        genes_downstream = downstream_hgnc_ids.intersection(intermediate_targets)

        for upstream_hgnc_id in genes_upstream:
            upstream_name = upstream_gene_set.get(upstream_hgnc_id, upstream_hgnc_id)
            for downstream_hgnc_id in genes_downstream:
                downstream_name = downstream_gene_set.get(downstream_hgnc_id, downstream_hgnc_id)

                pathways.append({
                    "upstream_gene": upstream_name,
                    "intermediate": intermediate_name,
                    "downstream_gene": downstream_name,
                    "path": f"{upstream_name} -> {intermediate_name} -> {downstream_name}",
                    "p_combined": row["p_combined"],
                    "q_combined": row["q_combined"],
                    "mlp_combined": row["mlp_combined"],
                    "upstream_score": upstream_scores.get(upstream_name, 0.0),
                    "downstream_score": downstream_scores.get(downstream_name, 0.0),
                })
        
    if not pathways:
        return pd.DataFrame()
    
    pathways_df = pd.DataFrame(pathways)
    scaler = MinMaxScaler()

    pathways_df["norm_enrichment"] = scaler.fit_transform(pathways_df[["mlp_combined"]])

    pathways_df["pathway_score"] = (
        pathways_df["norm_enrichment"] *
        pathways_df["upstream_score"] *
        pathways_df["downstream_score"]
    )

    return pathways_df.sort_values("pathway_score", ascending=False).reset_index(drop=True)
        

def build_network(pathways_df: pd.DataFrame) -> nx.DiGraph:
    """Build a NetworkX directed graph from scored pathways.

    Parameters
    ----------
    pathways_df :
        DataFrame of scored three-step paths from assemble_pathways.

    Returns
    -------
    :
        Directed graph with edge weights reflecting average pathway score.
    """

    G = nx.DiGraph()
    edge_weights = {}

    for _, row in pathways_df.iterrows():
        upstream = row["upstream_gene"]
        intermediate = row["intermediate"]
        downstream = row["downstream_gene"]
        score = row["pathway_score"]

        for edge in [(upstream, intermediate), (intermediate, downstream)]:
            if edge not in edge_weights:
                edge_weights[edge] = []
                edge_weights[edge].append(score)
    
    for (source, target), scores in edge_weights.items():
        G.add_edge(source, target, weight=np.mean(scores))

    return G

def build_network_visjs(
        pathways_df: pd.DataFrame,
        network: nx.DiGraph,
        intermediates_df: pd.DataFrame,
        upstream_scores: Dict[str, float],
        downstream_scores: Dict[str, float],
        top_n: int = 100,
) -> dict:
    """Build vis.js nodes and edges from pathway results.

    Parameters
    ----------
    pathways_df :
        DataFrame of scored three-step paths from assemble_pathways.
    network :
        NetworkX DiGraph from build_network.
    intermediates_df :
        DataFrame of enriched intermediates from indra_intermediate_ora.
    upstream_scores :
        Dict mapping upstream gene symbols to scaled scores in [0, 1].
    downstream_scores :
        Dict mapping downstream gene symbols to scaled scores in [0, 1].
    top_n : 
        Maximum number of pathways to include in the network visualization.
        If pathways_df has more than top_n unique intermediates, only the
        top_n pathways by pathway_score.

    Returns
    -------
    :
        Dictionary containing nodes and edges in vis.js format.
    """
    pathways_df = pathways_df.head(top_n)

    from indra_cogex.analysis.intermediate_pathway_analysis import build_network
    network = build_network(pathways_df)

    upstream_genes = set(pathways_df["upstream_gene"].unique())
    intermediates = set(pathways_df["intermediate"].unique())

    intermediates_lookup = {
        row["name"]: row
        for _, row in intermediates_df.iterrows()
    }

    nodes = []
    for node in network.nodes():
        if node in intermediates:
            color, shape, size = "#FF8C00", "ellipse", 45
            row = intermediates_lookup.get(node, {})
            details = {
                "type": "Intermediate",
                "q_combined": float(row["q_combined"]) if "q_combined" in row else None,
                "p_down": float(row["p_down"]) if "p_down" in row else None,
                "p_up": float(row["p_up"]) if "p_up" in row else None,
            }
            title = f"{node} (Intermediate)"
        elif node in upstream_genes:
            color, shape, size = "#4CAF50", "box", 35
            details = {
                "type": "Upstream",
                "metric_score": float(upstream_scores.get(node, 1.0)),
            }
            title = f"{node} (Upstream)"
        else:
            color, shape, size = "#2196F3", "box", 35
            details = {
                "type": "Downstream",
                "metric_score": float(downstream_scores.get(node, 1.0)),
            }
            title = f"{node} (Downstream)"

        nodes.append({
            "id": node,
            "label": node,
            "title": title,
            "color": {"background": color, "border": "#37474F"},
            "shape": shape,
            "size": size,
            "font": {"size": 22, "color": "#000000", "face": "arial"},
            "borderWidth": 2,
            "details": details,
        })

    edges = []
    for i, (source, target, data) in enumerate(network.edges(data=True)):
        weight = data.get("weight", 0)
        width = max(1.0, weight * 10)
        edges.append({
            "id": f"e{i}",
            "from": source,
            "to": target,
            "width": width,
            "title": f"Pathway score: {weight:.3f}",
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
            "color": {"color": "rgba(100,100,100,0.6)"},
            "details": {
                "source": source,
                "target": target,
                "pathway_score": float(weight),
            },
        })

    return {"nodes": nodes, "edges": edges}

@autoclient()
def intermediate_pathway_analysis(
    upstream_df: pd.DataFrame,
    downstream_df: pd.DataFrame,
    upstream_gene_col: str,
    upstream_metric_col: str,
    downstream_gene_col: str,
    downstream_metric_col: str,
    upstream_relationship_types: Optional[List[str]] = None,
    downstream_relationship_types: Optional[List[str]] = None,
    background_gene_ids: Optional[Collection[str]] = None,
    minimum_evidence_count: Optional[int] = 1,
    minimum_belief: Optional[float] = 0.0,
    pathway_score_threshold: Optional[float] = None,
    method: Optional[str] = "fdr_bh",
    alpha: Optional[float] = 0.05,
    keep_insignificant: bool = False,
    *,
    client: Neo4jClient,
) -> Dict[str, object]:
    """Reconstruct pathways connecting two gene sets through enriched
    intermediate regulators using INDRA CoGEx.

    Corresponding web-form based analysis can be found at:
    https://discovery.indra.bio/gene/pathway

    Parameters
    ----------
    upstream_df :
        DataFrame containing upstream gene data.
    downstream_df :
        DataFrame containing downstream gene data.
    upstream_gene_col :
        Column name for gene symbols in upstream_df.
    upstream_metric_col :
        Column name for the ranking metric in upstream_df. Values will
        be scaled to [0, 1] using min-max normalization.
    downstream_gene_col :
        Column name for gene symbols in downstream_df.
    downstream_metric_col :
        Column name for the ranking metric in downstream_df. Values
        will be scaled to [0, 1] using min-max normalization.
    upstream_relationship_types :
        Optional list of relationship types to filter the upstream ORA,
        e.g. ['Phosphorylation']. If None, uses the SQLite cache.
    downstream_relationship_types :
        Optional list of relationship types to filter the downstream ORA,
        e.g. ['Activation', 'Inhibition', 'IncreaseAmount', 'DecreaseAmount'].
        If None, uses the SQLite cache.
    background_gene_ids :
        Background gene set for ORA. Defaults to all human genes.
    minimum_evidence_count :
        Minimum number of evidences to consider a causal relationship.
    minimum_belief :
        Minimum belief to consider a causal relationship.
    pathway_score_threshold :
        Optional minimum pathway score filter.
    method :
        Multiple testing correction method, by default 'fdr_bh'.
    alpha :
        Significance threshold, by default 0.05.
    keep_insignificant :
        Whether to retain insignificant intermediates, by default False.
    client :
        Neo4jClient, managed automatically by the autoclient decorator.

    Returns
    -------
    :
        Dictionary with keys 'intermediates', 'pathways', 'network',
        'upstream_scores', and 'downstream_scores'.
    """

    upstream_genes, upstream_scores = read_gene_set(
        df=upstream_df,
        gene_col=upstream_gene_col,
        metric_col=upstream_metric_col,
    )

    downstream_genes, downstream_scores = read_gene_set(
        df=downstream_df,
        gene_col=downstream_gene_col,
        metric_col=downstream_metric_col,
    )

    intermediates_df = indra_intermediate_ora(
        client=client,
        upstream_gene_ids=upstream_genes,
        downstream_gene_ids=downstream_genes,
        background_gene_ids=background_gene_ids,
        minimum_evidence_count=minimum_evidence_count,
        minimum_belief=minimum_belief,
        method=method,
        alpha=alpha,
        keep_insignificant=keep_insignificant,
        upstream_relationship_types=upstream_relationship_types,
        downstream_relationship_types=downstream_relationship_types,
    )

    if intermediates_df.empty:
        return {
            "intermediates": intermediates_df,
            "pathways": pd.DataFrame(),
            "network": nx.DiGraph(),
            "upstream_scores": upstream_scores,
            "downstream_scores": downstream_scores,
        }
    
    entity_to_regulators = get_entity_to_regulators(
        client=client,
        minimum_evidence_count=minimum_evidence_count,
        minimum_belief=minimum_belief,
        relationship_types=upstream_relationship_types,
    )

    entity_to_targets = get_entity_to_targets(
        client=client,
        minimum_evidence_count=minimum_evidence_count,
        minimum_belief=minimum_belief,
        relationship_types=downstream_relationship_types,
    )

    pathways_df = assemble_pathways(
        upstream_genes=upstream_genes,
        downstream_genes=downstream_genes,
        intermediates_df=intermediates_df,
        entity_to_regulators=entity_to_regulators,
        entity_to_targets=entity_to_targets,
        upstream_scores=upstream_scores,
        downstream_scores=downstream_scores,
    )

    if pathways_df.empty:
        return {
            "intermediates": intermediates_df,
            "pathways": pd.DataFrame(),
            "network": nx.DiGraph(),
            "upstream_scores": upstream_scores,
            "downstream_scores": downstream_scores,
        }
    
    if pathway_score_threshold is not None:
        pathways_df = pathways_df[
            pathways_df["pathway_score"] >= pathway_score_threshold
        ]

    G = build_network(pathways_df)

    return {
        "intermediates": intermediates_df,
        "pathways": pathways_df,
        "network": G,
        "upstream_scores": upstream_scores,
        "downstream_scores": downstream_scores,
    }


