using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using AdvancedPenumbraItemConverter.Models;
using Dalamud.Plugin.Services;
using Lumina.Excel.Sheets;

namespace AdvancedPenumbraItemConverter.Services;

// ─────────────────────────────────────────────────────────────────────────────
// DTOs
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>
/// One variant entry from a game IMC (Item Metadata Change) file.
/// All field values are exactly as stored in the raw 6-byte entry.
/// </summary>
public sealed record ImcVariantEntry(
    /// <summary>1-based variant index (matches Penumbra's Variant field).</summary>
    int    Variant,
    /// <summary>Material-variant index; maps to the v000N material folder.</summary>
    byte   MaterialId,
    byte   DecalId,
    /// <summary>10-bit attribute bitmask (bits 0-9 of the packed ushort).</summary>
    ushort AttributeMask,
    /// <summary>6-bit sound ID (bits 10-15 of the packed ushort).</summary>
    byte   SoundId,
    byte   VfxId,
    byte   MaterialAnimationId
);

/// <summary>An equipment/accessory item ID + slot detected inside a mod directory.</summary>
public sealed class DetectedItem
{
    /// <summary>The equipment slot this item occupies.</summary>
    public EquipSlot Slot         { get; init; }

    /// <summary>Four-digit zero-padded model ID found in the mod paths (e.g. "0164").</summary>
    public string ModelIdPadded   { get; init; } = string.Empty;

    /// <summary>Variant index resolved from the Item sheet (bits 16-31 of ModelMain).</summary>
    public ushort Variant         { get; init; }

    /// <summary>Full display ID including variant suffix, e.g. "0585-2".</summary>
    public string ModelIdDisplay  => $"{ModelIdPadded}-{Variant}";

    /// <summary>Whether this is an accessory (prefix 'a') rather than equipment (prefix 'e').</summary>
    public bool   IsAccessory     { get; init; }

    /// <summary>Game item name resolved from the Item sheet, or "Unknown (ID {n})" if not found.</summary>
    public string ItemName        { get; init; } = string.Empty;
}

/// <summary>A searchable game item from the Item excel sheet.</summary>
public sealed class GameItem
{
    public uint      RowId        { get; init; }
    public string    Name         { get; init; } = string.Empty;

    /// <summary>Four-digit zero-padded model ID (lower 16 bits of ModelMain).</summary>
    public ushort    ModelId      { get; init; }

    /// <summary>Variant index (bits 16-31 of ModelMain, 1-based as stored by the game).</summary>
    public ushort    Variant      { get; init; }

    /// <summary>Four-digit zero-padded string for display / comparison.</summary>
    public string    ModelIdPadded => ModelId.ToString("D4");

    /// <summary>Full display ID including variant suffix, e.g. "0585-2".</summary>
    public string    ModelIdDisplay => $"{ModelIdPadded}-{Variant}";

    public EquipSlot Slot         { get; init; }
    public bool      IsAccessory  { get; init; }
}

// ─────────────────────────────────────────────────────────────────────────────
// Service
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>
/// Provides two capabilities:
/// <list type="bullet">
/// <item>Scan a Penumbra mod directory and detect which game items (slot + model ID) it
///       already replaces, resolved to human-readable names from game data.</item>
/// <item>Search the game's Item sheet by name to let the user pick a target item.</item>
/// </list>
/// </summary>
public sealed class GameDataService
{
    private readonly IDataManager _data;
    private readonly IPluginLog   _log;

    // Lazily-built searchable list of equipment/accessory items
    private List<GameItem>? _itemCache;

    // Matches tokens like e0164_top, a0123_ear — anywhere in a file path or JSON text.
    // Negative lookbehind only excludes preceding *letters* (not digits) so that
    // race-coded filenames like c0201e0164_top.mdl are matched correctly.
    private static readonly Regex TokenRx = new(
        @"(?<![a-zA-Z])([ea])(\d{4})_(met|top|glv|dwn|sho|ear|nek|wrs|rir|ril)(?![a-zA-Z0-9])",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    public GameDataService(IDataManager data, IPluginLog log)
    {
        _data = data;
        _log  = log;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Item search
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Returns up to <paramref name="max"/> equipment/accessory game items whose name contains
    /// <paramref name="query"/>. Prefix matches are listed before substring matches.
    /// </summary>
    public List<GameItem> SearchItems(string query, int max = 100)
    {
        if (string.IsNullOrWhiteSpace(query)) return new();

        var q     = query.Trim();
        var cache = GetItemCache();

        var starts   = cache
            .Where(i => i.Name.StartsWith(q, StringComparison.OrdinalIgnoreCase))
            .OrderBy(i => i.Name, StringComparer.OrdinalIgnoreCase);

        var contains = cache
            .Where(i => !i.Name.StartsWith(q, StringComparison.OrdinalIgnoreCase)
                     &&  i.Name.Contains(q,  StringComparison.OrdinalIgnoreCase))
            .OrderBy(i => i.Name, StringComparer.OrdinalIgnoreCase);

        return starts.Concat(contains).Take(max).ToList();
    }

    /// <summary>
    /// Like <see cref="SearchItems(string,int)"/> but restricted to a specific slot.
    /// </summary>
    public List<GameItem> SearchItems(string query, EquipSlot slot, int max = 100)
    {
        if (string.IsNullOrWhiteSpace(query)) return new();

        var q     = query.Trim();
        var cache = GetItemCache().Where(i => i.Slot == slot);

        var starts   = cache
            .Where(i => i.Name.StartsWith(q, StringComparison.OrdinalIgnoreCase))
            .OrderBy(i => i.Name, StringComparer.OrdinalIgnoreCase);

        var contains = cache
            .Where(i => !i.Name.StartsWith(q, StringComparison.OrdinalIgnoreCase)
                     &&  i.Name.Contains(q,  StringComparison.OrdinalIgnoreCase))
            .OrderBy(i => i.Name, StringComparer.OrdinalIgnoreCase);

        return starts.Concat(contains).Take(max).ToList();
    }

    /// <summary>
    /// Returns the number of items in the cache (useful for diagnostics / status display).
    /// Triggers a cache build if needed.
    /// </summary>
    public int CacheCount => GetItemCache().Count;

    /// <summary>
    /// Returns all equipment/accessory items for <paramref name="slot"/>, sorted by name.
    /// This is the full unfiltered list used to populate the target item picker.
    /// </summary>
    public List<GameItem> GetAllItemsForSlot(EquipSlot slot)
        => GetItemCache()
            .Where(i => i.Slot == slot)
            .OrderBy(i => i.Name, StringComparer.OrdinalIgnoreCase)
            .ToList();

    // ─────────────────────────────────────────────────────────────────────────
    // Mod scan
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Scans <paramref name="modDir"/> recursively for file paths and JSON content that
    /// contain equipment/accessory item tokens (e.g. <c>e0164_top</c>, <c>a0123_ear</c>).
    /// Each distinct (slot, modelId) combination is resolved to a game item name and
    /// returned as a <see cref="DetectedItem"/>.
    /// </summary>
    public List<DetectedItem> ScanModForItems(string modDir)
    {
        if (!Directory.Exists(modDir)) return new();

        // Key: (slot, 4-digit-id), Value: isAccessory
        var found = new Dictionary<(EquipSlot, string), bool>();

        try
        {
            foreach (var file in Directory.EnumerateFiles(modDir, "*", SearchOption.AllDirectories))
            {
                // Scan file name
                var name = Path.GetFileName(file);
                foreach (Match m in TokenRx.Matches(name))
                    RecordToken(m, found);

                // Also scan JSON file content (game paths inside Files/FileSwaps keys/values)
                if (string.Equals(Path.GetExtension(file), ".json",
                        StringComparison.OrdinalIgnoreCase))
                {
                    try
                    {
                        var text = File.ReadAllText(file);
                        foreach (Match m in TokenRx.Matches(text))
                            RecordToken(m, found);
                    }
                    catch { /* skip unreadable */ }
                }
            }
        }
        catch (Exception ex)
        {
            _log.Warning(ex, "[APIC] ScanModForItems failed for {0}", modDir);
        }

        // Resolve names from game data
        var cache  = GetItemCache();
        var result = new List<DetectedItem>();

        foreach (var ((slot, id), isAcc) in found
                     .OrderBy(k => (int)k.Key.Item1)
                     .ThenBy(k => k.Key.Item2))
        {
            if (!ushort.TryParse(id, out var modelId)) continue;

            var match     = cache.FirstOrDefault(i => i.ModelId == modelId && i.Slot == slot);
            var itemName  = match?.Name ?? $"Unknown (ID {modelId})";

            result.Add(new DetectedItem
            {
                Slot          = slot,
                ModelIdPadded = id,
                Variant       = match?.Variant ?? 0,
                IsAccessory   = isAcc,
                ItemName      = itemName,
            });
        }

        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ─────────────────────────────────────────────────────────────────────────

    private static void RecordToken(Match m, Dictionary<(EquipSlot, string), bool> found)
    {
        var prefix  = char.ToLower(m.Groups[1].Value[0]); // 'e' or 'a'
        var id      = m.Groups[2].Value;                  // "0164"
        var slotKey = m.Groups[3].Value.ToLower();        // "top" etc.

        if (!SlotInfo.ReverseMap.TryGetValue(slotKey, out var slot)) return;

        bool isAcc = prefix == 'a';
        found.TryAdd((slot, id), isAcc);
    }

    private List<GameItem> GetItemCache()
    {
        if (_itemCache != null) return _itemCache;

        var list = new List<GameItem>(8192);
        try
        {
            var sheet = _data.GetExcelSheet<Item>();
            if (sheet == null) goto done;

            foreach (var row in sheet)
            {
                // Skip items with no name
                var name = row.Name.ToString();
                if (string.IsNullOrWhiteSpace(name)) continue;

                // Skip items with no model
                var modelMain = row.ModelMain;
                if (modelMain == 0) continue;

                var primaryId = (ushort)(modelMain & 0xFFFF);
                if (primaryId == 0) continue;

                // Skip if we cannot map to an equipment/accessory slot we support
                if (!TryGetEquipSlot(row, out var slot, out var isAcc)) continue;

                var variant = (ushort)((modelMain >> 16) & 0xFFFF);

                list.Add(new GameItem
                {
                    RowId       = row.RowId,
                    Name        = name,
                    ModelId     = primaryId,
                    Variant     = variant,
                    Slot        = slot,
                    IsAccessory = isAcc,
                });
            }
        }
        catch (Exception ex)
        {
            _log.Warning(ex, "[APIC] Failed to build item name cache");
        }

        done:
        _itemCache = list;
        _log.Information("[APIC] Item cache built: {0} equipment/accessory items", list.Count);
        return _itemCache;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // EQP lookup
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Reads the vanilla Equipment Parameter (EQP) entry for the given equipment
    /// set ID and slot directly from the game binary.
    /// Returns null for accessory slots (they have no EQP) or on read failure.
    /// Body and Head slots produce a ushort value (16-bit flags);
    /// Hands, Legs and Feet produce a byte value.
    /// </summary>
    public ulong? GetDefaultEqpEntry(ushort setId, EquipSlot slot)
    {
        if (SlotInfo.IsAccessory(slot)) return null;

        try
        {
            var file = _data.GetFile("chara/xls/equipmentparameter/equipmentparameter.eqp");
            if (file == null) return null;

            var bytes = file.Data;
            if (bytes.Length < 8) return null;

            // 8-byte LE ulong header = number of set entries
            var count = BitConverter.ToUInt64(bytes, 0);
            if ((ulong)setId >= count) return null;

            // Each entry is 8 bytes, starting at offset 8
            int entryOffset = 8 + setId * 8;
            if (entryOffset + 8 > bytes.Length) return null;

            // Read the full 8-byte entry as a LE ulong and return the slot-specific
            // bits *in their correct bit positions*.  Penumbra's "Entry" JSON field is
            // the full 64-bit ulong; when it applies the manipulation it masks out only
            // the slot's region (e.g. bits 48-55 for Feet).  Returning just the raw byte
            // value (e.g. 15) without shifting would place those bits at positions 0-3
            // (Body slot), so Penumbra would see zeroes after masking for Feet.
            //
            // Byte layout (LE ulong):
            //   bits  0-15 (bytes 0-1) = Body  flags  → mask 0x000000000000FFFF
            //   bits 16-31 (bytes 2-3) = Head  flags  → mask 0x00000000FFFF0000
            //   bits 32-39 (byte 4)    = Hands flags  → mask 0x000000FF00000000
            //   bits 40-47 (byte 5)    = Legs  flags  → mask 0x0000FF0000000000
            //   bits 48-55 (byte 6)    = Feet  flags  → mask 0x00FF000000000000
            var fullEntry = BitConverter.ToUInt64(bytes, entryOffset);
            return slot switch
            {
                EquipSlot.Body  => fullEntry & 0x000000000000FFFFUL,
                EquipSlot.Head  => fullEntry & 0x00000000FFFF0000UL,
                EquipSlot.Hands => fullEntry & 0x000000FF00000000UL,
                EquipSlot.Legs  => fullEntry & 0x0000FF0000000000UL,
                EquipSlot.Feet  => fullEntry & 0x00FF000000000000UL,
                _               => null,
            };
        }
        catch (Exception ex)
        {
            _log.Warning(ex, "[APIC] GetDefaultEqpEntry failed for setId={0} slot={1}", setId, slot);
            return null;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // EQDP lookup
    // ─────────────────────────────────────────────────────────────────────────

    // All race/gender combinations that have their own EQDP file.
    // Race names match Penumbra's ModelRace enum spellings; Gender matches Gender enum.
    // The code is the 4-digit suffix used in the game file path.
    private static readonly (string Race, string Gender, string Code)[] EqdpRaces =
    {
        ("Midlander",  "Male",   "0101"),
        ("Midlander",  "Female", "0201"),
        ("Highlander", "Male",   "0301"),
        ("Highlander", "Female", "0401"),
        ("Elezen",     "Male",   "0501"),
        ("Elezen",     "Female", "0601"),
        ("Lalafell",   "Male",   "0701"),
        ("Lalafell",   "Female", "0801"),
        ("Miqote",     "Male",   "0901"),
        ("Miqote",     "Female", "1001"),
        ("Roegadyn",   "Male",   "1101"),
        ("Roegadyn",   "Female", "1201"),
        ("AuRa",       "Male",   "1301"),
        ("AuRa",       "Female", "1401"),
        ("Hrothgar",   "Male",   "1501"),
        ("Viera",      "Female", "1701"),
        ("Viera",      "Male",   "1801"),
    };

    // Bit-shift per equipment slot within the 2-byte EQDP entry ushort.
    // Each slot occupies 2 bits: bit0 = has model, bit1 = has material variant.
    private static int EqdpSlotShift(EquipSlot slot) => slot switch
    {
        EquipSlot.Head  => 0,
        EquipSlot.Body  => 2,
        EquipSlot.Hands => 4,
        EquipSlot.Legs  => 6,
        EquipSlot.Feet  => 8,
        // Accessories share the same per-slot bit layout in their own files
        EquipSlot.Earring   => 0,
        EquipSlot.Neck      => 2,
        EquipSlot.Wrists    => 4,
        EquipSlot.RingRight => 6,
        EquipSlot.RingLeft  => 8,
        _ => -1,
    };

    /// <summary>
    /// Reads the vanilla EQDP 2-bit entry for <paramref name="setId"/> + <paramref name="slot"/>
    /// for every available race/gender combination.
    /// Returns one tuple per race whose file exists and contains a non-zero entry.
    /// </summary>
    public List<(string Race, string Gender, byte Entry)> GetDefaultEqdpEntries(
        ushort setId, EquipSlot slot)
    {
        var result = new List<(string, string, byte)>();
        int shift  = EqdpSlotShift(slot);
        if (shift < 0) return result;

        foreach (var (race, gender, code) in EqdpRaces)
        {
            try
            {
                var path = $"chara/xls/equipmentdeformerparameter/c{code}.eqdp";
                var file = _data.GetFile(path);
                if (file == null) continue;

                var bytes = file.Data;
                if (bytes.Length < 8) continue;

                // Same header as EQP: 8-byte LE ulong = number of entries.
                // Each entry is a LE ushort (2 bytes), indexed by set ID.
                var count = BitConverter.ToUInt64(bytes, 0);
                if ((ulong)setId >= count) continue;

                int offset = 8 + setId * 2;
                if (offset + 2 > bytes.Length) continue;

                var fullEntry = BitConverter.ToUInt16(bytes, offset);
                var slotBits  = (byte)((fullEntry >> shift) & 0x3);

                if (slotBits != 0)
                    result.Add((race, gender, slotBits));
            }
            catch (Exception ex)
            {
                _log.Warning(ex, "[APIC] GetDefaultEqdpEntries failed for {0}/{1} setId={2}",
                             race, gender, setId);
            }
        }

        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // IMC lookup
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Reads the vanilla IMC file for <paramref name="setId"/> and returns all
    /// variant entries for the given equipment/accessory <paramref name="slot"/>.
    /// Each entry preserves all original fields; callers may override <c>MaterialId</c>
    /// before emitting a Penumbra manipulation.
    /// Returns an empty list when the file cannot be found or parsed.
    /// </summary>
    public List<ImcVariantEntry> GetImcVariantEntries(ushort setId, EquipSlot slot)
    {
        var result = new List<ImcVariantEntry>();
        try
        {
            bool isAcc    = SlotInfo.IsAccessory(slot);
            var  prefix   = isAcc ? "a" : "e";
            var  category = isAcc ? "accessory" : "equipment";
            var  path     = $"chara/{category}/{prefix}{setId:D4}/{prefix}{setId:D4}.imc";

            var file = _data.GetFile(path);
            if (file == null) return result;

            var data = file.Data;
            // IMC file layout:
            //   Offset 0 (uint16): packed field – bits 0-4 = numParts, bits 5-15 = subCount
            //     subCount = number of additional variant rows; total rows = subCount + 1
            //     numParts = parts per row (5 for equipment, 1 for accessories)
            //   Offset 2 (uint16): type/flags (unused here)
            //   Offset 4+ : entries, 6 bytes each, row-major: row0part0, row0part1, …
            //   Each 6-byte entry:
            //     byte  0 : MaterialId
            //     byte  1 : DecalId
            //     bytes 2-3: packed ushort – bits 0-9 = AttributeMask, bits 10-15 = SoundId
            //     byte  4 : VfxId
            //     byte  5 : MaterialAnimationId
            if (data.Length < 4) return result;

            ushort header    = BitConverter.ToUInt16(data, 0);
            int    numParts  = header & 0x1F;         // lower 5 bits
            int    subCount  = (header >> 5) & 0x7FF; // next 11 bits
            int    totalRows = subCount + 1;

            // Sanity-clamp: for equipment always ≥ 1 part (might be 0 for some exotic items)
            if (numParts == 0)
                numParts = isAcc ? 1 : 5;

            int partIdx = GetImcPartIndex(slot);
            if (partIdx < 0 || partIdx >= numParts) return result;

            for (int row = 0; row < totalRows; row++)
            {
                int offset = 4 + (row * numParts + partIdx) * 6;
                if (offset + 6 > data.Length) break;

                byte   materialId  = data[offset];
                byte   decalId     = data[offset + 1];
                ushort packed      = BitConverter.ToUInt16(data, offset + 2);
                ushort attrMask    = (ushort)(packed & 0x3FF);
                byte   soundId     = (byte)((packed >> 10) & 0x3F);
                byte   vfxId       = data[offset + 4];
                byte   animId      = data[offset + 5];

                result.Add(new ImcVariantEntry(
                    Variant             : row + 1,  // 1-based to match Penumbra
                    MaterialId          : materialId,
                    DecalId             : decalId,
                    AttributeMask       : attrMask,
                    SoundId             : soundId,
                    VfxId               : vfxId,
                    MaterialAnimationId : animId));
            }
        }
        catch (Exception ex)
        {
            _log.Warning(ex, "[APIC] GetImcVariantEntries failed for setId={0} slot={1}", setId, slot);
        }
        return result;
    }

    /// <summary>
    /// Returns the zero-based part index within an IMC row for the given equipment slot.
    /// Equipment rows have 5 parts (Head=0, Body=1, Hands=2, Legs=3, Feet=4).
    /// Accessory rows have 1 part (index 0).
    /// Returns -1 for unsupported slots.
    /// </summary>
    private static int GetImcPartIndex(EquipSlot slot) => slot switch
    {
        EquipSlot.Head      => 0,
        EquipSlot.Body      => 1,
        EquipSlot.Hands     => 2,
        EquipSlot.Legs      => 3,
        EquipSlot.Feet      => 4,
        EquipSlot.Earring   => 0,
        EquipSlot.Neck      => 0,
        EquipSlot.Wrists    => 0,
        EquipSlot.RingRight => 0,
        EquipSlot.RingLeft  => 0,
        _                   => -1,
    };

    /// <summary>
    /// Inspects the item's <c>EquipSlotCategory</c> sub-row to determine which
    /// equipment slot it occupies. Returns false for weapons, offhands and anything
    /// outside the ten slots we support.
    /// </summary>
    private static bool TryGetEquipSlot(Item row, out EquipSlot slot, out bool isAcc)
    {
        slot  = EquipSlot.Body;
        isAcc = false;

        try
        {
            var catRowId = row.EquipSlotCategory.RowId;
            if (catRowId == 0) return false;

            var cat = row.EquipSlotCategory.Value;

            // Equipment (prefix 'e')
            if (cat.Head   != 0) { slot = EquipSlot.Head;      return true; }
            if (cat.Body   != 0) { slot = EquipSlot.Body;      return true; }
            if (cat.Gloves != 0) { slot = EquipSlot.Hands;     return true; }
            if (cat.Legs   != 0) { slot = EquipSlot.Legs;      return true; }
            if (cat.Feet   != 0) { slot = EquipSlot.Feet;      return true; }

            // Accessories (prefix 'a')
            if (cat.Ears     != 0) { slot = EquipSlot.Earring;   isAcc = true; return true; }
            if (cat.Neck     != 0) { slot = EquipSlot.Neck;      isAcc = true; return true; }
            if (cat.Wrists   != 0) { slot = EquipSlot.Wrists;    isAcc = true; return true; }
            if (cat.FingerR  != 0) { slot = EquipSlot.RingRight; isAcc = true; return true; }
            if (cat.FingerL  != 0) { slot = EquipSlot.RingLeft;  isAcc = true; return true; }
        }
        catch
        {
            // Sub-row might not resolve for some items — skip silently
        }

        return false;
    }
}
