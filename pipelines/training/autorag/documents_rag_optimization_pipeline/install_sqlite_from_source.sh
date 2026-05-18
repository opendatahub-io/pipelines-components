#!/usr/bin/env bash
# Build SQLite from the sqlite-autoconf tarball and replace the system libsqlite3 shared libraries.
#
# Tarball resolution:
#   1. Find sqlite-autoconf-3510300.tar.gz under ${HERMETO_GENERIC_DEPS} (Hermeto generic prefetch;
#      see artifacts.lock.yaml).
#   2. Otherwise download from sqlite.org (networked builds, e.g. local Containerfile).
#
# This script is intentionally separate from seed_docling_models.py (Docling OCI artifacts).

set -euo pipefail

readonly SQLITE_DIR_NAME="sqlite-autoconf-3510300"
readonly TARBALL_NAME="${SQLITE_DIR_NAME}.tar.gz"
readonly SQLITE_YEAR="2026"
readonly DOWNLOAD_URL="https://sqlite.org/${SQLITE_YEAR}/${TARBALL_NAME}"
readonly BUILD_ROOT="/tmp/sqlite-build-${SQLITE_DIR_NAME}"


resolve_system_sqlite_libdir() {
  local f
  if command -v ldconfig >/dev/null 2>&1; then
    f="$(ldconfig -p 2>/dev/null | awk '/libsqlite3\.so\.0 \(/{print $NF; exit}' || true)"
    if [[ -n "${f}" && -e "${f}" ]]; then
      dirname "$(readlink -f "${f}")"
      return 0
    fi
  fi
  for d in /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib/ppc64le-linux-gnu /usr/lib/s390x-linux-gnu; do
    if [[ -e "${d}/libsqlite3.so.0" ]]; then
      echo "${d}"
      return 0
    fi
  done
  echo "install_sqlite_from_source.sh: error: could not locate system libsqlite3.so.0" >&2
  return 1
}

find_tarball_in_hermeto() {
  local root="${HERMETO_GENERIC_DEPS:-}"
  [[ -n "${root}" && -d "${root}" ]] || return 1
  find "${root}" -name "${TARBALL_NAME}" -type f 2>/dev/null | head -1
}

fetch_tarball_to() {
  local dest="$1"
  local found
  found="$(find_tarball_in_hermeto || true)"
  if [[ -n "${found}" ]]; then
    echo "Using SQLite tarball from Hermeto: ${found}"
    cp -a "${found}" "${dest}"
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "install_sqlite_from_source.sh: error: no ${TARBALL_NAME} under HERMETO_GENERIC_DEPS and curl is not installed" >&2
    return 1
  fi
  echo "Downloading ${DOWNLOAD_URL}"
  curl -fsSL "${DOWNLOAD_URL}" -o "${dest}"
}

find_built_libdir() {
  if [[ -d /usr/local/lib64 ]] && compgen -G "/usr/local/lib64/libsqlite3.so*" >/dev/null 2>&1; then
    echo "/usr/local/lib64"
    return 0
  fi
  if compgen -G "/usr/local/lib/libsqlite3.so*" >/dev/null 2>&1; then
    echo "/usr/local/lib"
    return 0
  fi
  local one
  one="$(find /usr/local -name 'libsqlite3.so*' 2>/dev/null | head -1 || true)"
  if [[ -n "${one}" ]]; then
    dirname "${one}"
    return 0
  fi
  return 1
}

main() {


  local work_tar="/tmp/${TARBALL_NAME}"
  fetch_tarball_to "${work_tar}"

  rm -rf "${BUILD_ROOT}"
  mkdir -p "${BUILD_ROOT}"
  tar -xzf "${work_tar}" -C "${BUILD_ROOT}"
  cd "${BUILD_ROOT}/${SQLITE_DIR_NAME}"

  ./configure --prefix=/usr/local --enable-shared --disable-static
  make -j"$(nproc)"
  make install

  local system_libdir built_libdir
  system_libdir="$(resolve_system_sqlite_libdir)"
  built_libdir="$(find_built_libdir)" || {
    echo "install_sqlite_from_source.sh: error: could not find built libsqlite3 under /usr/local" >&2
    return 1
  }

  mkdir -p /opt/.sqlite-system-backup
  shopt -s nullglob
  for f in "${system_libdir}"/libsqlite3.so*; do
    [[ -e "${f}" ]] || continue
    cp -a "${f}" /opt/.sqlite-system-backup/
  done
  cp -a "${built_libdir}"/libsqlite3.so* "${system_libdir}/"
  ln -sf "${system_libdir}/libsqlite3.so.3.51.3" "${system_libdir}/libsqlite3.so"
  ln -sf "${system_libdir}/libsqlite3.so.3.51.3" "${system_libdir}/libsqlite3.so.0"

  rm -rf "${BUILD_ROOT}" "${work_tar}"
  cd /
}

main "$@"
