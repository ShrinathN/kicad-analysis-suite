#!/usr/bin/env bash
# Install the KiCad IPC plugin into the user plugin directory.
#   ./install.sh            copy (default)
#   ./install.sh --link     symlink, for development
# Detects Flatpak KiCad (~/.var/app/org.kicad.KiCad) and native installs.
# Override the destination with KICAD_PLUGIN_DIR=/path/to/plugins.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
ver="${KICAD_VERSION:-10.0}"
flatpak_data="$HOME/.var/app/org.kicad.KiCad/data/kicad"
if [[ -z "${KICAD_PLUGIN_DIR:-}" ]]; then
    for base in "$flatpak_data" "$HOME/.local/share/kicad" "$HOME/.local/share/KiCad"; do
        if [[ -d "$base/$ver" ]]; then KICAD_PLUGIN_DIR="$base/$ver/plugins"; break; fi
    done
    KICAD_PLUGIN_DIR="${KICAD_PLUGIN_DIR:-$HOME/.local/share/kicad/$ver/plugins}"
fi
dest="$KICAD_PLUGIN_DIR/kicad-analysis-suite"
mkdir -p "$KICAD_PLUGIN_DIR"
# Installs made before the rename used this folder (and the identifier io.github.shinu.pcb-current-sim);
# remove it so KiCad does not load the plugin twice.
if [[ -e "$KICAD_PLUGIN_DIR/pcb_current_sim/plugin.json" ]]; then
    rm -rf "$KICAD_PLUGIN_DIR/pcb_current_sim"
    echo "Removed the old install at $KICAD_PLUGIN_DIR/pcb_current_sim"
fi
rm -rf "$dest"
mkdir -p "$dest"
files=(plugin.json requirements.txt _bootstrap.py action.py home.py icon-24.png icon-48.png icon-home-24.png icon-home-48.png)
if [[ "${1:-}" == "--link" ]]; then
    for f in "${files[@]}"; do ln -s "$here/$f" "$dest/$f"; done
    ln -s "$here/../pcb_current_sim" "$dest/pcb_current_sim"
else
    for f in "${files[@]}"; do cp "$here/$f" "$dest/"; done
    cp -r "$here/../pcb_current_sim" "$dest/"
    find "$dest" -name __pycache__ -prune -exec rm -rf {} +
fi
echo "Installed to $dest"
echo "In KiCad: Preferences > Plugins > tick \"Enable KiCad API\", then restart the PCB editor."
echo "KiCad builds the plugin's Python environment (numpy, scipy, matplotlib, kicad-python, ...)"
echo "on first load; that needs network access and can take a minute before the button appears."
if [[ "$KICAD_PLUGIN_DIR" == "$flatpak_data"* ]]; then
    echo "Flatpak KiCad detected: the plugin uses the wxPython GUI bundled with KiCad."
else
    echo "Native KiCad: the system Python needs wxPython or tkinter (e.g. sudo apt install python3-tk)."
fi
