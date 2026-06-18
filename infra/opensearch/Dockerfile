ARG OPENSEARCH_VERSION=2.19.1
FROM opensearchproject/opensearch:${OPENSEARCH_VERSION}

# H1 uses the standard analyzers.  The dedicated image pins the server version
# so product-index builds cannot silently move to a different OpenSearch release.
