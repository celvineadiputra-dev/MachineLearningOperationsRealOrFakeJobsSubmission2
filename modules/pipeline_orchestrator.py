"""
Pipeline Orchestrator
"""
from typing import Text
from absl import logging

from tfx.orchestration import metadata, pipeline


def init_pipeline_orchestrator(
        components,
        pipeline_root: Text,
        pipeline_name: str,
        metadata_path: Text,
) -> pipeline.Pipeline:
    """
    Initializes a pipeline orchestrator with the given components, pipeline root, pipeline name,
    and metadata path.

    Args:
        components (List[Component]): The list of components to be included in the pipeline.
        pipeline_root (Text): The root directory of the pipeline.
        pipeline_name (str): The name of the pipeline.
        metadata_path (Text): The path to the metadata file.

    Returns:
        pipeline.Pipeline: The initialized pipeline object.
    """
    logging.info(f"Pipeline root set to : {pipeline_root}")

    beam_args = [
        "--direct_running_mode=multi_processing",
        "----direct_num_workers=0",
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path,
        ),
        eam_pipeline_args=beam_args,
    )
