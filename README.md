# Advanced Penumbra Item Converter

Download the latest release here: https://github.com/link-0402/Advanced-Penumbra-Item-Converter/releases/tag/2.0

A Python GUI tool for converting FFXIV items within a Penumbra mod pack with advanced options from one item to another while preserving said options.

! Note that this tool is unable to read game data of any sort, so it will be unable to write metadata changes that were not explicitly set on the modded item beforehand.
Example: A pair of short boots without explicit meta is moved to a set of long boots. As a result, the knees will likely get hidden. You can manually fix this ingame.
Similarly, some models may not show ingame if the new item uses different racial versions, for example if it lacks a female version, the old male item will be shown. You can easily fix this with manual metadata edits in the “Meta Manipulations” tab in Penumbra. 
I’ve basically hit the limits of what I can do here as an external tool without access to game data. Similar limitations apply for items with variants using different material sets (-2 as an ending to it’s item code for instance), you may need to set the material set of the new item to match that of the old item.

## Features

- **Smart Asset Chain Discovery**: Automatically traces dependencies from models through materials to textures
- **Interactive Preview System**: See all planned changes before applying them with a detailed checklist UI
- **Selective Conversion**: Choose exactly which files, directories, and JSON entries to modify
- **Race-Specific Filtering**: Target specific character races and optionally clean up unused race assets
- **Slot-Aware Processing**: Intelligently filters operations by equipment slot (head, body, hands, etc.)
- **Model Selection Dialog**: Pick which .mdl files to include when multiple candidates exist
- **JSON Metadata Updates**: Updates Penumbra mod configuration files with slot-aware logic
  - Handles `Files` and `FileSwaps` dictionaries
  - Updates `PrimaryId` and `SetId` fields
  - Preserves structure while replacing item IDs
- **Binary Path Replacement**: Updates hardcoded paths in .mtrl and .mdl files
- **Diff Highlighting**: Visual comparison of old vs new filenames with color-coded changes
- **Post-Conversion Cleanup**: Optional removal of orphaned assets from other races
- **Slot Restriction Override**: Option to find all files with old item ID regardless of slot

## Requirements

- None, I think.

## Usage

See Guide.pdf document for detailed instructions.

## File Structure Requirements

Your Penumbra mod directory must contain:
```
your_mod_folder/
├── default_mod.json    (required)
├── meta.json          (required)
├── [other mod files]
```

The tool will scan all subdirectories for relevant files based on:
- Item ID patterns (e.g., `e0001`, `e9999`)
- Slot identifiers (e.g., `_met`, `_top`, `_glv`)
- Race codes (e.g., `c0101`, `c0201`)

## Slot Keys Reference

| Slot Name | Key | Description |
|-----------|-----|-------------|
| Head | `met` | Helmets, hats, masks |
| Body | `top` | Chest armor, robes |
| Hands | `glv` | Gloves, gauntlets |
| Legs | `dwn` | Pants, skirts |
| Feet | `sho` | Boots, shoes |
| Earring | `ear` | Earrings |
| Neck | `nek` | Necklaces |
| Wrists | `wrs` | Bracelets |
| Ring Right | `rir` | Right ring |
| Ring Left | `ril` | Left ring |

## Race Codes Reference

| Race | Code | Description |
|------|------|-------------|
| Midlander Male | `0101` | Hyur Midlander ♂ |
| Midlander Female | `0201` | Hyur Midlander ♀ |
| Highlander Male | `0301` | Hyur Highlander ♂ |
| Highlander Female | `0401` | Hyur Highlander ♀ |
| Elezen Male | `0501` | Elezen ♂ |
| Elezen Female | `0601` | Elezen ♀ |
| Miqo'te Male | `0701` | Miqo'te ♂ |
| Miqo'te Female | `0801` | Miqo'te ♀ |
| Roegadyn Male | `0901` | Roegadyn ♂ |
| Roegadyn Female | `1001` | Roegadyn ♀ |
| Lalafell Male | `1101` | Lalafell ♂ |
| Lalafell Female | `1201` | Lalafell ♀ |
| Au Ra Male | `1301` | Au Ra ♂ |
| Au Ra Female | `1401` | Au Ra ♀ |
| Hrothgar Male | `1501` | Hrothgar ♂ |
| Hrothgar Female | `1601` | Hrothgar ♀ |
| Viera Male | `1701` | Viera ♂ |
| Viera Female | `1801` | Viera ♀ |

## Troubleshooting

**"Missing required file(s)" error:**
- Ensure the selected directory is a valid Penumbra mod folder
- Check that both `default_mod.json` and `meta.json` exist in the root

**No models found:**
- Verify the original item ID is correct
- Try enabling "Ignore slot restrictions" checkbox
- Check that model files use standard naming conventions (e.g., `e####_slot.mdl`)
- Manually check if model files exist matching your item ID

## Best Practices

1. **Always make backups** of your mod folder before conversion
2. **Normalize the Mod through Penumbra first** to drastically increase chances of successful conversion
3. **Use the preview feature** to verify all changes before applying
4. **Test in-game** after each conversion to verify results
5. **Clean up other race assets** after conversion to reduce mod size (unless the mod uses race-specific headgear / accessories or is for both M/F)

## Known Limitations

- Only works with Penumbra mod format
- Requires valid JSON structure in mod configuration files
- Cannot undo conversions (backup before converting)
- Race detection depends on standard filename conventions (c####)
- Some edge cases with non-standard mod structures may need manual adjustment

## Example Use Cases

**Converting a body piece from item 6132 to 6140 for female Midlander:**
1. Set Original ID: `6132`
2. Set Target ID: `6140`
3. Select Slot: `Body (top)`
4. Select Race: `Midlander Female (0201)`
5. Review preview and confirm

**Converting all files for an item regardless of slot:**
1. Enter item IDs
2. Check "Ignore slot restrictions"
3. Select all models in the dialog
4. Review comprehensive preview
5. Confirm conversion

## License

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

## Credits

**Author**: Luci  
**Version**: 2.0.0

### Links
- **XMA Profile**: [XIV Mod Archive](https://www.xivmodarchive.com/user/124593)
- **X/Twitter**: [@Luci__xiv](https://x.com/Luci__xiv)
- **Support**: [Ko-fi](https://ko-fi.com/Luci_xiv)

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

---

**Note**: This tool is not affiliated with or endorsed by Square Enix or the Penumbra development team. Use at your own risk and always maintain backups of your mods.
