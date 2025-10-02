# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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