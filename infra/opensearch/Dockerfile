ARG OPENSEARCH_VERSION=2.19.1
FROM opensearchproject/opensearch:${OPENSEARCH_VERSION}

ARG OPENSEARCH_VERSION
RUN /usr/share/opensearch/bin/opensearch-plugin install --batch \
    "https://get.infini.cloud/opensearch/analysis-ik/${OPENSEARCH_VERSION}"
