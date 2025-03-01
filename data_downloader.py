from bing_image_downloader import downloader
from queries_for_download import queries
from pathlib import Path

path_name = "raw_dataset"

dir = Path(path_name)
dir.mkdir(parents=True, exist_ok=True)

for query in queries:
    downloader.download(query, limit=100, output_dir=path_name, adult_filter_off=False)
