# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.2.0]

### Added

- CHANGELOG.md
- documentation on ffmpeg dependency
- recommendations on whispe model usage for english and other languages
- `--workers` flag that allows multithread processing of transcription

### Fixed

- ML models used in verbatim will now be sent to the correct device with the
following preference: `mps`(for series M apple processors) -> `cuda` -> `cpu`
- errors will no longer be shown twice when thrown by spinners
- wav is now the default format that audio chunks will be saved in

### Removed

- poetry as package manager

## [v0.1.0]

### Added

- verbatim
