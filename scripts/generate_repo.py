#!/usr/bin/env python3
"""
Arch Repository Generator

Generates an Arch Linux Pacman repository from GitHub releases containing binary tarballs.
Downloads tar.gz/zip files, extracts binaries, and repackages as .pkg.tar.zst.

Supports incremental updates - can update a single project while preserving others.
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml


# Architecture mapping from common names to Arch names
ARCH_MAP = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "x64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}

# Manifest file to track packages by project
MANIFEST_FILE = "packages.json"


@dataclass
class PacmanPackage:
    """Represents a Pacman package."""
    pkgname: str
    pkgver: str
    pkgrel: int
    arch: str
    pkgdesc: str
    url: str
    license: str
    packager: str
    builddate: int
    size: int
    # For tracking
    project_repo: str = ""
    source_url: str = ""
    filename: str = ""
    sha256: str = ""


@dataclass
class Release:
    """Represents a GitHub release."""
    tag: str
    version: str
    major_minor: str
    packages: list[PacmanPackage] = field(default_factory=list)


@dataclass
class Project:
    """Project configuration from projects.yaml."""
    repo: str
    name: str = ""
    description: str = ""
    license: str = "MIT"
    keep_versions: int = 0
    binary_name: str = ""
    install_path: str = "/usr/bin"
    asset_patterns: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = self.repo.split("/")[-1]
        if not self.binary_name:
            self.binary_name = self.name


@dataclass
class RepoSettings:
    """Repository settings from projects.yaml."""
    name: str = "custom"
    baseurl: str = ""
    architectures: list[str] = field(default_factory=lambda: ["x86_64", "aarch64"])
    description: str = "Custom Arch Packages"
    packager: str = "Unknown Packager <unknown@example.com>"


class GitHubAPI:
    """GitHub API client for fetching releases."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.base_url = "https://api.github.com"

    def _request(self, endpoint: str) -> dict | list:
        """Make an authenticated request to GitHub API."""
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "arch-repo-generator",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        req = Request(url, headers=headers)
        try:
            with urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode())
        except HTTPError as e:
            if e.code == 404:
                raise RuntimeError(f"Repository or release not found: {endpoint}")
            elif e.code == 403:
                raise RuntimeError(
                    f"Rate limit exceeded. Set GITHUB_TOKEN env var for higher limits."
                )
            raise

    def get_releases(self, repo: str, per_page: int = 30) -> list[dict]:
        """Get releases for a repository."""
        return self._request(f"repos/{repo}/releases?per_page={per_page}")

    def get_repo(self, repo: str) -> dict:
        """Get repository information including description."""
        return self._request(f"repos/{repo}")


def extract_version(tag: str) -> str:
    """Extract version number from tag (removes 'v' prefix)."""
    return tag.lstrip("vV")


def extract_major_minor(version: str) -> str:
    """Extract major.minor from version string."""
    parts = version.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return version


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_file(url: str, dest: Path, token: Optional[str] = None) -> None:
    """Download a file from URL to destination."""
    headers = {"User-Agent": "arch-repo-generator"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = Request(url, headers=headers)
    with urlopen(req, timeout=300) as response:
        with open(dest, "wb") as f:
            shutil.copyfileobj(response, f)


def extract_binary(archive_path: Path, binary_name: str, dest_dir: Path) -> Optional[Path]:
    """Extract binary from archive (tar.gz, zip, etc.)."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        if archive_path.suffix == ".zip" or archive_path.name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                for name in zf.namelist():
                    basename = os.path.basename(name)
                    if basename == binary_name or basename == f"{binary_name}.exe":
                        # Extract to dest
                        extracted = dest_dir / binary_name
                        with zf.open(name) as src, open(extracted, 'wb') as dst:
                            dst.write(src.read())
                        os.chmod(extracted, 0o755)
                        return extracted
        else:
            # Assume tar.gz, tar.xz, etc.
            with tarfile.open(archive_path, 'r:*') as tf:
                for member in tf.getmembers():
                    basename = os.path.basename(member.name)
                    if basename == binary_name and member.isfile():
                        # Extract to dest
                        extracted = dest_dir / binary_name
                        with tf.extractfile(member) as src, open(extracted, 'wb') as dst:
                            dst.write(src.read())
                        os.chmod(extracted, 0o755)
                        return extracted
    except Exception as e:
        print(f"      Error extracting: {e}")

    return None


def create_pkginfo(pkg: PacmanPackage, installed_size: int) -> str:
    """Generate .PKGINFO content."""
    return f"""# Generated by arch-repo-generator
pkgname = {pkg.pkgname}
pkgbase = {pkg.pkgname}
pkgver = {pkg.pkgver}-{pkg.pkgrel}
pkgdesc = {pkg.pkgdesc}
url = {pkg.url}
builddate = {pkg.builddate}
packager = {pkg.packager}
size = {installed_size}
arch = {pkg.arch}
license = {pkg.license}
"""


def create_mtree(files: list[tuple[str, Path]], builddate: int) -> bytes:
    """Generate .MTREE content (gzip compressed as required by pacman).

    Format follows mtree(5) specification used by makepkg.
    """
    lines = [
        "#mtree",
        "/set type=file uid=0 gid=0 mode=644",
    ]

    # Track directories we've already added
    added_dirs = set()

    for install_path, local_path in files:
        # Remove leading slash for mtree
        mtree_path = install_path.lstrip("/")
        stat = local_path.stat()
        sha256 = compute_sha256(local_path)

        # Add directory entries
        parts = mtree_path.split("/")
        for i in range(len(parts) - 1):
            dir_path = "/".join(parts[:i+1])
            if dir_path not in added_dirs:
                lines.append(f"./{dir_path} time={builddate}.0 type=dir mode=755")
                added_dirs.add(dir_path)

        # Set mode for executables then add file entry
        lines.append("/set mode=755")
        lines.append(f"./{mtree_path} time={builddate}.0 size={stat.st_size} sha256digest={sha256}")

    # Ensure trailing newline
    content = "\n".join(lines) + "\n"
    # Pacman requires .MTREE to be gzip compressed
    return gzip.compress(content.encode("utf-8"))


def build_package(
    pkg: PacmanPackage,
    binary_path: Path,
    install_path: str,
    output_dir: Path,
) -> Optional[Path]:
    """Build a .pkg.tar.zst package."""
    pkg_filename = f"{pkg.pkgname}-{pkg.pkgver}-{pkg.pkgrel}-{pkg.arch}.pkg.tar.zst"
    pkg_path = output_dir / pkg_filename

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create package structure with all parent directories
        install_dir = tmppath / install_path.lstrip("/")
        install_dir.mkdir(parents=True, exist_ok=True)

        # Copy binary
        dest_binary = install_dir / binary_path.name
        shutil.copy2(binary_path, dest_binary)
        os.chmod(dest_binary, 0o755)

        installed_size = dest_binary.stat().st_size

        # Create .PKGINFO
        pkginfo_content = create_pkginfo(pkg, installed_size)
        pkginfo_path = tmppath / ".PKGINFO"
        pkginfo_path.write_text(pkginfo_content)

        # Create .MTREE (gzip compressed as required by pacman)
        full_install_path = f"{install_path}/{binary_path.name}"
        mtree_content = create_mtree([(full_install_path, dest_binary)], pkg.builddate)
        mtree_path = tmppath / ".MTREE"
        mtree_path.write_bytes(mtree_content)

        # Get the top-level directory (e.g., "usr" from "usr/bin")
        # This ensures all parent dirs are included in the archive
        top_level_dir = install_path.lstrip("/").split("/")[0]

        # Build tar.zst using bsdtar (preferred) or tar + zstd
        try:
            # Try bsdtar first (handles .PKGINFO ordering correctly)
            result = subprocess.run(
                [
                    "bsdtar", "-cf", str(pkg_path),
                    "--zstd",
                    "-C", str(tmppath),
                    ".PKGINFO", ".MTREE",
                    top_level_dir
                ],
                capture_output=True,
                timeout=120,
            )
            if result.returncode == 0:
                return pkg_path
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Fallback: use tar + zstd
        try:
            tar_path = tmppath / "pkg.tar"
            result = subprocess.run(
                [
                    "tar", "-cf", str(tar_path),
                    "-C", str(tmppath),
                    ".PKGINFO", ".MTREE",
                    top_level_dir
                ],
                capture_output=True,
                timeout=120,
            )
            if result.returncode == 0:
                result = subprocess.run(
                    ["zstd", "-q", "-f", str(tar_path), "-o", str(pkg_path)],
                    capture_output=True,
                    timeout=120,
                )
                if result.returncode == 0:
                    return pkg_path
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Last fallback: tar.gz instead of tar.zst
        try:
            pkg_filename_gz = f"{pkg.pkgname}-{pkg.pkgver}-{pkg.pkgrel}-{pkg.arch}.pkg.tar.gz"
            pkg_path_gz = output_dir / pkg_filename_gz
            result = subprocess.run(
                [
                    "tar", "-czf", str(pkg_path_gz),
                    "-C", str(tmppath),
                    ".PKGINFO", ".MTREE",
                    top_level_dir
                ],
                capture_output=True,
                timeout=120,
            )
            if result.returncode == 0:
                pkg.filename = pkg_filename_gz
                return pkg_path_gz
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"      Error creating package: {e}")

    return None


def find_matching_asset(
    release_data: dict,
    pattern: str,
) -> Optional[dict]:
    """Find asset matching the pattern."""
    for asset in release_data.get("assets", []):
        name = asset.get("name", "")
        if re.search(pattern, name, re.IGNORECASE):
            return {
                "name": name,
                "url": asset.get("browser_download_url", ""),
                "size": asset.get("size", 0),
            }
    return None


def fetch_releases(
    github: GitHubAPI,
    project: Project,
    settings: RepoSettings,
) -> list[Release]:
    """Fetch and process releases for a project."""
    releases_data = github.get_releases(project.repo)

    # Fetch description from GitHub if not provided
    description = project.description
    if not description:
        try:
            repo_info = github.get_repo(project.repo)
            description = repo_info.get("description") or f"{project.name} from GitHub"
        except Exception:
            description = f"{project.name} from GitHub"

    # Group releases by major.minor version
    releases_by_minor: dict[str, dict] = {}

    for release_data in releases_data:
        # Skip pre-releases and drafts
        if release_data.get("prerelease") or release_data.get("draft"):
            continue

        tag = release_data["tag_name"]
        version = extract_version(tag)
        major_minor = extract_major_minor(version)

        # Keep only the first (latest) release for each major.minor
        if major_minor not in releases_by_minor:
            releases_by_minor[major_minor] = release_data

    # Sort by version (newest first)
    sorted_versions = sorted(
        releases_by_minor.keys(),
        key=lambda v: [int(x) if x.isdigit() else 0 for x in v.split(".")],
        reverse=True,
    )

    # Determine how many versions to keep
    if project.keep_versions > 0:
        versions_to_keep = sorted_versions[: project.keep_versions + 1]
    else:
        versions_to_keep = sorted_versions[:1]

    releases = []
    for major_minor in versions_to_keep:
        release_data = releases_by_minor[major_minor]
        tag = release_data["tag_name"]
        version = extract_version(tag)

        release = Release(
            tag=tag,
            version=version,
            major_minor=major_minor,
        )

        # Find assets for each architecture
        for arch in settings.architectures:
            pattern = project.asset_patterns.get(arch)
            if not pattern:
                continue

            asset = find_matching_asset(release_data, pattern)
            if asset:
                pkg = PacmanPackage(
                    pkgname=project.name,
                    pkgver=version,
                    pkgrel=1,
                    arch=arch,
                    pkgdesc=description,
                    url=f"https://github.com/{project.repo}",
                    license=project.license,
                    packager=settings.packager,
                    builddate=int(datetime.now().timestamp()),
                    size=asset["size"],
                    project_repo=project.repo,
                    source_url=asset["url"],
                    filename=f"{project.name}-{version}-1-{arch}.pkg.tar.zst",
                )
                release.packages.append(pkg)

        if release.packages:
            releases.append(release)

    return releases


def run_repo_add(db_path: Path, pkg_files: list[Path]) -> bool:
    """Run repo-add to update the package database."""
    if not pkg_files:
        return False

    try:
        cmd = ["repo-add", str(db_path)] + [str(f) for f in pkg_files]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return True
        print(f"    repo-add failed: {result.stderr}")
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"    repo-add not available: {e}")

    return False


def sign_database(db_path: Path, gpg_key: Optional[str] = None) -> bool:
    """Sign the package database with GPG."""
    try:
        gpg_cmd = ["gpg", "--batch", "--yes", "--detach-sign"]
        if gpg_key:
            gpg_cmd.extend(["--local-user", gpg_key])
        gpg_cmd.append(str(db_path))

        subprocess.run(gpg_cmd, check=True, capture_output=True, timeout=60)
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"    GPG signing failed: {e}")
        return False


def export_public_key(output_path: Path, gpg_key: Optional[str] = None) -> bool:
    """Export GPG public key."""
    try:
        gpg_cmd = ["gpg", "--armor", "--export"]
        if gpg_key:
            gpg_cmd.append(gpg_key)

        result = subprocess.run(gpg_cmd, capture_output=True, timeout=60)
        if result.returncode == 0 and result.stdout:
            output_path.write_bytes(result.stdout)
            return True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return False


def load_manifest(output_dir: Path) -> dict:
    """Load existing packages manifest."""
    manifest_path = output_dir / MANIFEST_FILE
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {"packages": []}


def save_manifest(output_dir: Path, packages: list[PacmanPackage]) -> None:
    """Save packages manifest."""
    manifest_path = output_dir / MANIFEST_FILE
    data = {
        "packages": [asdict(pkg) for pkg in packages],
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=2)


def cleanup_old_packages(
    output_dir: Path,
    all_packages: list[PacmanPackage],
    architectures: list[str],
) -> list[Path]:
    """Remove package files that are no longer in the manifest."""
    removed = []
    current_filenames = {pkg.filename for pkg in all_packages}

    for arch in architectures:
        arch_dir = output_dir / arch
        if arch_dir.exists():
            for pkg_file in arch_dir.glob("*.pkg.tar.*"):
                if pkg_file.name not in current_filenames:
                    pkg_file.unlink()
                    removed.append(pkg_file)

    return removed


def load_config(config_path: Path) -> tuple[RepoSettings, list[Project]]:
    """Load configuration from projects.yaml."""
    with open(config_path) as f:
        data = yaml.safe_load(f)

    settings_data = data.get("settings", {})
    settings = RepoSettings(
        name=settings_data.get("name", "custom"),
        baseurl=settings_data.get("baseurl", ""),
        architectures=settings_data.get("architectures", ["x86_64", "aarch64"]),
        description=settings_data.get("description", "Custom Arch Packages"),
        packager=settings_data.get("packager", "Unknown <unknown@example.com>"),
    )

    projects = []
    for proj_data in data.get("projects", []):
        projects.append(Project(
            repo=proj_data["repo"],
            name=proj_data.get("name", ""),
            description=proj_data.get("description", ""),
            license=proj_data.get("license", "MIT"),
            keep_versions=proj_data.get("keep_versions", 0),
            binary_name=proj_data.get("binary_name", ""),
            install_path=proj_data.get("install_path", "/usr/bin"),
            asset_patterns=proj_data.get("asset_patterns", {}),
        ))

    return settings, projects


def main():
    parser = argparse.ArgumentParser(
        description="Generate Arch Pacman repository from GitHub releases"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("projects.yaml"),
        help="Path to projects.yaml config file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("repo"),
        help="Output directory for repository",
    )
    parser.add_argument(
        "--project", "-p",
        type=str,
        help="Process only specific project (name or owner/repo)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List configured projects and exit",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without downloading",
    )
    parser.add_argument(
        "--gpg-key", "-k",
        type=str,
        help="GPG key ID for signing (optional)",
    )
    parser.add_argument(
        "--no-sign",
        action="store_true",
        help="Skip GPG signing",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild packages (with --project: only that project, without: all)",
    )

    args = parser.parse_args()

    # Load configuration
    settings, all_projects = load_config(args.config)

    # List mode
    if args.list:
        print("Configured projects:")
        for proj in all_projects:
            print(f"  - {proj.repo} (name: {proj.name}, keep_versions: {proj.keep_versions})")
        return

    # Determine which projects to process
    if args.project:
        projects_to_process = [
            p for p in all_projects
            if p.name == args.project or p.repo == args.project
        ]
        if not projects_to_process:
            print(f"Error: Project '{args.project}' not found in config")
            return 1
    else:
        projects_to_process = all_projects

    # Initialize
    github = GitHubAPI()
    output_dir = args.output

    # Load existing manifest
    manifest = load_manifest(output_dir)
    existing_packages = [
        PacmanPackage(**pkg_data) for pkg_data in manifest.get("packages", [])
    ]

    # Filter out packages from projects we're about to update
    projects_to_update = {p.repo for p in projects_to_process}
    preserved_packages = [
        pkg for pkg in existing_packages
        if pkg.project_repo not in projects_to_update
    ]

    # Create architecture directories
    if not args.dry_run:
        for arch in settings.architectures:
            (output_dir / arch).mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(projects_to_process)} project(s)...")
    if preserved_packages:
        print(f"Preserving {len(preserved_packages)} package(s) from other projects")

    # If rebuild mode, clear existing packages for projects being updated
    if args.rebuild and not args.dry_run:
        # Build set of filenames to remove (only for projects being updated)
        files_to_remove = {
            pkg.filename for pkg in existing_packages
            if pkg.project_repo in projects_to_update
        }

        if args.project:
            print(f"\n[Rebuild mode: clearing packages for {args.project}]")
        else:
            print("\n[Rebuild mode: clearing all packages]")

        for arch in settings.architectures:
            arch_dir = output_dir / arch
            if arch_dir.exists():
                for pkg_file in arch_dir.glob("*.pkg.tar.*"):
                    # Only remove if it belongs to projects being updated
                    if pkg_file.name in files_to_remove or not args.project:
                        pkg_file.unlink()
                        print(f"  Removed {pkg_file}")
                # Remove database files (will be regenerated)
                for db_file in arch_dir.glob("*.db*"):
                    db_file.unlink()
                for files_file in arch_dir.glob("*.files*"):
                    files_file.unlink()

        # Update preserved packages (keep packages from other projects)
        if args.project:
            # Keep packages from projects NOT being rebuilt
            preserved_packages = [
                pkg for pkg in existing_packages
                if pkg.project_repo not in projects_to_update
            ]
        else:
            preserved_packages = []

    new_packages: list[PacmanPackage] = []

    for project in projects_to_process:
        print(f"\n{'='*60}")
        print(f"Project: {project.repo}")
        print(f"{'='*60}")

        try:
            releases = fetch_releases(github, project, settings)

            if not releases:
                print(f"  No matching releases found")
                continue

            for release in releases:
                print(f"\n  Release: {release.tag} (version {release.version})")

                for pkg in release.packages:
                    print(f"    - {pkg.arch}: {pkg.source_url.split('/')[-1]}")

                    if args.dry_run:
                        continue

                    arch_dir = output_dir / pkg.arch
                    pkg_path = arch_dir / pkg.filename

                    # Skip if package already exists
                    if pkg_path.exists():
                        print(f"      Package exists, skipping")
                        pkg.sha256 = compute_sha256(pkg_path)
                        new_packages.append(pkg)
                        continue

                    # Download and extract
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmppath = Path(tmpdir)
                        archive_path = tmppath / pkg.source_url.split("/")[-1]

                        print(f"      Downloading...")
                        download_file(pkg.source_url, archive_path, github.token)

                        print(f"      Extracting binary...")
                        binary_path = extract_binary(
                            archive_path,
                            project.binary_name,
                            tmppath / "extracted"
                        )

                        if not binary_path:
                            print(f"      Error: Could not find binary '{project.binary_name}'")
                            continue

                        print(f"      Building package...")
                        built_pkg = build_package(
                            pkg,
                            binary_path,
                            project.install_path,
                            arch_dir,
                        )

                        if built_pkg:
                            pkg.filename = built_pkg.name
                            pkg.sha256 = compute_sha256(built_pkg)
                            pkg.size = built_pkg.stat().st_size
                            new_packages.append(pkg)
                            print(f"      Created: {built_pkg.name}")
                        else:
                            print(f"      Error: Failed to build package")

        except Exception as e:
            print(f"  Error processing {project.repo}: {e}")
            continue

    if args.dry_run:
        print("\n[Dry run - no files written]")
        return

    # Combine packages
    all_packages = preserved_packages + new_packages

    # Cleanup old packages
    removed = cleanup_old_packages(output_dir, all_packages, settings.architectures)
    if removed:
        print(f"\nRemoved {len(removed)} old package(s)")

    # Save manifest
    save_manifest(output_dir, all_packages)

    print(f"\n{'='*60}")
    print("Generating repository database...")
    print(f"{'='*60}")

    # Generate database for each architecture
    for arch in settings.architectures:
        arch_dir = output_dir / arch
        arch_packages = [pkg for pkg in all_packages if pkg.arch == arch]
        pkg_files = [arch_dir / pkg.filename for pkg in arch_packages if (arch_dir / pkg.filename).exists()]

        if pkg_files:
            print(f"\n  {arch}: {len(pkg_files)} package(s)")
            db_path = arch_dir / f"{settings.name}.db.tar.gz"

            if run_repo_add(db_path, pkg_files):
                print(f"    Created {settings.name}.db.tar.gz")

                # Sign database
                if not args.no_sign and args.gpg_key:
                    if sign_database(db_path, args.gpg_key):
                        print(f"    Signed database")
            else:
                print(f"    Warning: Could not create database")
        else:
            print(f"\n  {arch}: no packages")

    # Export public key
    if not args.no_sign and args.gpg_key:
        keys_dir = output_dir / "keys"
        keys_dir.mkdir(exist_ok=True)
        key_path = keys_dir / f"{settings.name}.gpg"
        if export_public_key(key_path, args.gpg_key):
            print(f"\n  Exported public key to keys/{settings.name}.gpg")

    print(f"\nRepository generated in: {output_dir}")
    print(f"Total packages: {len(all_packages)}")


if __name__ == "__main__":
    exit(main() or 0)