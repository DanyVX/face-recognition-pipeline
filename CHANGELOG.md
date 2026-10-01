# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Repository skeleton, tooling (uv, ruff, mypy, pytest, pre-commit), CI, and settings/environment
  logging (M0).
- Image loader with format sniffing, bomb protection, EXIF/alpha/palette/CMYK/16-bit handling and
  bounded downscaling with coordinate mapping (M1).
- `FaceDetector` protocol and SCRFD backend (boxes, 5 landmarks, confidence), multi-face policy,
  checksum-verified model downloader (M1).
