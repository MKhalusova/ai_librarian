import os
from dotenv import load_dotenv

from unstructured.ingest.v2.pipeline.pipeline import Pipeline
from unstructured.ingest.v2.interfaces import ProcessorConfig
from unstructured.ingest.v2.processes.connectors.local import (
    LocalIndexerConfig,
    LocalDownloaderConfig,
    LocalConnectionConfig,
)
from unstructured.ingest.v2.processes.connectors.mongodb import (
    MongoDBConnectionConfig,
    MongoDBUploadStagerConfig,
    MongoDBUploaderConfig,
    MongoDBAccessConfig
)

from unstructured.ingest.v2.processes.partitioner import PartitionerConfig
from unstructured.ingest.v2.processes.chunker import ChunkerConfig
from unstructured.ingest.v2.processes.embedder import EmbedderConfig

if __name__ == "__main__":
    load_dotenv()

    Pipeline.from_configs(
        context=ProcessorConfig(
            verbose=True,
            tqdm=True,
            num_processes=20
        ),

        indexer_config=LocalIndexerConfig(input_path=os.getenv("BOOKS_PATH"), recursive=False),
        downloader_config=LocalDownloaderConfig(),
        source_connection_config=LocalConnectionConfig(),

        partitioner_config=PartitionerConfig(
            partition_by_api=True,
            api_key=os.getenv("UNSTRUCTURED_API_KEY"),
            partition_endpoint=os.getenv("UNSTRUCTURED_URL"),
            strategy="fast"
        ),

        chunker_config=ChunkerConfig(
            chunking_strategy="by_title",
            chunk_max_characters=512,
            chunk_multipage_sections=True,
            chunk_combine_text_under_n_chars=250,
        ),

        embedder_config=EmbedderConfig(
            embedding_provider="langchain-huggingface",
            embedding_model_name=os.getenv("EMBEDDING_MODEL"),
        ),

        destination_connection_config=MongoDBConnectionConfig(
            access_config=MongoDBAccessConfig(uri=os.getenv("MONGODB_URI")),
            collection="unstructured-demo",
            database="books",
        ),
        stager_config=MongoDBUploadStagerConfig(),
        uploader_config=MongoDBUploaderConfig(batch_size=10)
    ).run()