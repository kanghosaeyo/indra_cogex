from indra_cogex.client import Neo4jClient
from indra_cogex.client.enrichment.discrete import indra_intermediate_ora

client = Neo4jClient()

upstream_genes = [
    "MINK1", "MAP3K19", "PIP4K2C", "ANKK1", "CDPK1", "MAP3K10", "RIPK4",
    "MAP4K4", "TNIK", "RIOK3", "MAP4K2", "ULK3", "PIKFYVE", "CSNK1E",
    "PAK7", "DYRK1A", "MYLK", "PIK3CG", "BMP2K", "MARK2", "SGK1",
    "AURKA", "PIP5K1C", "RIOK1", "MAP2K2", "MAP2K5", "MYLK2", "MAP3K7",
    "STK17B", "DCLK3"
]

downstream_genes = [
    "SRGAP2", "BAIAP2", "EIF5", "FAM171B", "NEO1", "TANC2", "VASP", "ITSN1",
    "SORBS1", "LIMCH1", "REPS1", "CLASP1", "CTNND1", "NUMBL", "PLEKHA5",
    "AFDN", "DVL1", "PARD3", "ARVCF", "FMNL2", "SH3KBP1", "DLL1", "ERBB2",
    "SLC39A10", "PSD3", "MAP4K4", "PLEKHA7", "GAB2", "ARHGAP35", "DVL3",
    "PCDH10", "BAALC", "CCDC88A", "CLASP2", "TLN2", "ERBIN", "ADGRB3",
    "CTNND2", "PAK4", "MYO10", "VCL", "ARHGAP5", "TNIK", "RASSF8",
    "CDC42BPB", "NHSL2", "PAG1", "MSN", "C1orf43", "AMOT", "DLG1",
    "CSNK2A2", "ADGRL3", "FAT1", "PLXND1", "FAS", "FGD3", "LIFR"
]

result = indra_intermediate_ora(
    client=client,
    upstream_gene_ids=upstream_genes,
    downstream_gene_ids=downstream_genes,
    minimum_belief=0.5,
    minimum_evidence_count=1,
)

print(result.head(20))