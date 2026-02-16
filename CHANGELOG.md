# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.3.2] - 2025-02-16

### Added
- `update_reference_features`, a quick way to update your reference feature dict

### Changed
- Made `pool_predictions`, `insert_new_id`, and `find_neighbors` public. 
- Lazy imports for `verify_pytorch()` and `get_best_device()`

## [0.3.1] - 2025-02-11

### Changed
- Bug fixes: Clean up a couple of fixes with `identify.predict_ids`
- Updated documentation to reflect new versioning

## [0.3.0] - 2025-02-11

### Added
- Added `identify.predict_ids` to predict ids of animals in query images given a 
  reference set
- Added `experimental` module to visualize Pyseter output

## [0.2.3] - 2025-02-11

### Changed
- Bug fix: Continue extracting features even if file is corrupted

## [0.2.2] - 2025-12-14

### Changed
- Updated `load_features` to accept a path, not a folder.

## [0.2.1] - 2025-09-26

### Changed
- Fixed a bug in rate_distinctiveness

## [0.2.0] - 2025-09-26

### Added
- Automatic model downloading from Hugging Face Hub
- Model caching system for one-time downloads

### Changed
- Model weights now downloaded automatically instead of downloaded separately 
- User no longer provides `model_path` argument to `FeatureExtractor`
- Added `huggingface-hub>=0.16.0` as a dependency

### Technical
- Models cached in `~/.cache/huggingface/hub/` directory

## [0.1.0] - 2025-09-16

### Added
- Initial release 
- Extract feature vectors from images with AnyDorsal
- Grade individuals in images based on distinctiveness
- Cluster individuals based on similarity with NetworkX or sklearn's AgglomerativeClustering
- Sort images into folders by proposed ID then encounter