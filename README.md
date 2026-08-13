# Advanced Penumbra Item Converter

A [Dalamud](https://github.com/goatcorp/Dalamud) plugin for FFXIV that retargets [Penumbra](https://github.com/xivdev/Penumbra) modded items from one item to another while preserving advanced mod options and toggles.

## Requirements

- FFXIV with [XIVLauncher](https://github.com/goatcorp/FFXIVQuickLauncher) and Dalamud installed
- [Penumbra](https://github.com/xivdev/Penumbra) plugin installed and enabled

## Installation

1. Open Dalamud settings → **Experimental** → **Custom Plugin Repositories**
2. Add the following URL and click **Save**:
   ```
   https://raw.githubusercontent.com/link-0402/AdvancedPenumbraItemConverter/main/repo.json
   ```
3. Search for **Advanced Penumbra Item Converter** in the plugin installer and install it.

## Usage

Open the window with `/apic`. Usage should be fairly self explanatory.
I recommend making a backup of the source mod if you plan to convert in-place in case anything goes wrong.

1. Pick your mod from the Penumbra mod browser, or paste the mod folder path directly.
2. The plugin will auto-detect the source item ID from the mod's files.
3. Search for the target item by name and select it.
4. For accessories, choose the output slot if you want to change it.
5. Optionally enable **Create new mod** to copy the assets into a new Penumbra mod instead of modifying the original.
6. Click **Preview Changes**. You may now manually disable certain files from being affected on the preview tab if necessary.
7. 

## License

[AGPL-3.0-or-later](https://www.gnu.org/licenses/agpl-3.0.en.html)
