from bing_image_downloader import downloader
from queries_for_download import queries

for query in queries:
    downloader.download(query, limit=100, output_dir='dataset', adult_filter_off=False)
