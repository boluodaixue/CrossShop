# RAG data and reproducibility

The full local CategoryInsight release is maintained outside this public checkout because its catalog-derived provenance is not redistributed. The public repository contains representative authored documentation only. Runtime retrieval uses the OpenSearch Knowledge Index; Markdown files are build inputs, not the online source of truth.

Knowledge documents are classified as `catalog_attribute_summary`, `catalog_style_proxy`, `historical_price_range`, `curated_selection_guide`, or `official_source_summary`. Official-source documents are Chinese summaries with a checked-on date, not copied official text or a live compliance promise.

Build an index with `scripts/index/build_knowledge_opensearch.py` and pass an explicit source root and publish alias. Keep private/full corpora outside Git and use a separate alias from the public demo alias.
