# Arch Repository Generator

[![Update Arch Repository](https://github.com/bordeux/arch-repo/actions/workflows/update-repo.yml/badge.svg)](https://github.com/bordeux/arch-repo/actions/workflows/update-repo.yml)

Generate an Arch Linux Pacman repository from GitHub releases containing binary tarballs. Downloads `.tar.gz` files, extracts binaries, and repackages as `.pkg.tar.zst`.

**No changes required to your release process** - works with standard binary releases.

## How It Works

1. Define projects in `projects.yaml` with asset patterns for each architecture
2. GitHub Actions downloads tarballs, extracts binaries, builds `.pkg.tar.zst` packages
3. Generates Pacman database with `repo-add`
4. Deploys to GitHub Pages
5. Users install with `pacman`

## Quick Start

### 1. Configure Projects

Edit `projects.yaml`:

```yaml
settings:
  name: bordeux
  baseurl: https://bordeux.github.io/arch-repo
  architectures:
    - x86_64
    - aarch64
  packager: "Your Name <email@example.com>"

projects:
  - repo: bordeux/tmpltool
    name: tmpltool
    description: "A fast template rendering tool"
    license: MIT
    binary_name: tmpltool          # Name of binary in the tarball
    install_path: /usr/bin         # Where to install
    asset_patterns:
      x86_64: "linux.*amd64.*\\.tar\\.gz$"
      aarch64: "linux.*arm64.*\\.tar\\.gz$"
```

### 2. Enable GitHub Pages

1. Go to repository **Settings** > **Pages**
2. Set **Source** to "Deploy from a branch"
3. Select **gh-pages** branch

### 3. (Optional) Set Up GPG Signing

1. Add secret `GPG_PRIVATE_KEY`
2. Add variable `GPG_KEY_ID`

### 4. Trigger the Workflow

Go to **Actions** > **Update Arch Repository** > **Run workflow**

## Using the Repository

Add to `/etc/pacman.conf`:

```ini
[bordeux]
SigLevel = Optional TrustAll
Server = https://bordeux.github.io/arch-repo/$arch
```

Or with signature verification:

```ini
[bordeux]
SigLevel = Required
Server = https://bordeux.github.io/arch-repo/$arch
```

Then install:

```bash
# Import key (if signed)
sudo pacman-key --add <(curl -s https://bordeux.github.io/arch-repo/keys/bordeux.gpg)
sudo pacman-key --lsign-key <KEY_ID>

# Install
sudo pacman -Sy tmpltool
```

## Configuration Reference

### projects.yaml

```yaml
settings:
  name: bordeux                              # Repository name
  baseurl: https://bordeux.github.io/arch-repo
  architectures:
    - x86_64
    - aarch64
  description: "Bordeux Arch Packages"
  packager: "Name <email@example.com>"       # Packager info

projects:
  - repo: owner/repo                         # GitHub repository
    name: pkgname                            # Package name
    description: "Package description"
    license: MIT                             # License
    keep_versions: 1                         # Past versions to keep
    binary_name: myapp                       # Binary name in tarball
    install_path: /usr/bin                   # Installation path
    asset_patterns:                          # Regex patterns for assets
      x86_64: "linux.*amd64.*\\.tar\\.gz$"
      aarch64: "linux.*arm64.*\\.tar\\.gz$"
```

### Command Line

```bash
# Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate repository locally (requires pacman tools)
python scripts/generate_repo.py --output repo

# Process specific project
python scripts/generate_repo.py --project bordeux/tmpltool

# Dry run
python scripts/generate_repo.py --dry-run

# List configured projects
python scripts/generate_repo.py --list

# With GPG signing
python scripts/generate_repo.py --gpg-key YOUR_KEY_ID
```

## Repository Structure

```
arch-repo/
├── x86_64/
│   ├── bordeux.db              (symlink)
│   ├── bordeux.db.tar.gz       (package database)
│   ├── bordeux.db.tar.gz.sig   (signature, if signed)
│   ├── bordeux.files           (symlink)
│   ├── bordeux.files.tar.gz    (file listings)
│   └── *.pkg.tar.zst           (packages)
├── aarch64/
│   └── ...
├── keys/
│   └── bordeux.gpg             (GPG public key)
└── packages.json               (manifest)
```

## Requirements

- Python 3.9+
- PyYAML
- `repo-add` (from pacman - available in Arch container)
- `bsdtar` or `tar` + `zstd`
- GPG (optional, for signing)

The GitHub Actions workflow runs in an `archlinux:latest` container where all tools are available.

## How Repackaging Works

1. **Download**: Fetches `tmpltool-linux-amd64.tar.gz` from GitHub release
2. **Extract**: Finds `tmpltool` binary inside the tarball
3. **Package**: Creates `tmpltool-1.2.2-1-x86_64.pkg.tar.zst` with:
   - `.PKGINFO` - Package metadata
   - `.MTREE` - File checksums
   - `usr/bin/tmpltool` - The binary
4. **Index**: Runs `repo-add` to update the database

## License

MIT