using System.Collections.Generic;

namespace AdvancedPenumbraItemConverter.Models;

/// <summary>Represents a single planned item conversion and all the changes it entails.</summary>
public class ConversionTask
{
    /// <summary>Absolute path to the root of the Penumbra mod folder.</summary>
    public string ModDirectory { get; set; } = string.Empty;

    /// <summary>The slot being converted.</summary>
    public EquipSlot Slot { get; set; } = EquipSlot.Body;

    /// <summary>Zero-padded source item ID string (e.g. "0164").</summary>
    public string OldIdPadded { get; set; } = string.Empty;

    /// <summary>Zero-padded target item ID string (e.g. "0200").</summary>
    public string NewIdPadded { get; set; } = string.Empty;

    /// <summary>When true, slot-specific filename filtering is skipped.</summary>
    public bool IgnoreSlot { get; set; } = false;

    /// <summary>
    /// For accessory cross-slot conversion: the output slot.
    /// When null (default), the output slot is identical to <see cref="Slot"/>.
    /// </summary>
    public EquipSlot? TargetSlot { get; set; } = null;

    /// <summary>
    /// Material variant index for the target item (1-based, matching the game's v000N
    /// material folder convention).  Used when creating a new mod to normalise all
    /// material-variant path components to this value and to inject IMC manipulations
    /// that redirect every game variant of the new item to this material variant.
    /// </summary>
    public int TargetVariant { get; set; } = 1;

    /// <summary>
    /// Material variant of the source item being converted (e.g. 4 for "9069-4").
    /// Used when multiple source material variants normalise to the same destination
    /// path so that the version matching the user-selected variant is preferred.
    /// 0 means unspecified (fall back to highest variant number).
    /// </summary>
    public int SourceVariant { get; set; } = 0;

    // ── Planned changes ───────────────────────────────────────────────────────

    public List<PlannedRename>      PlannedRenames      { get; } = new();
    public List<PlannedJsonChange>  PlannedJsonChanges  { get; } = new();
    public List<PlannedBinaryPatch> PlannedBinaryPatches{ get; } = new();

    /// <summary>
    /// All physical asset files discovered in the item's asset chain during planning
    /// (models, materials, textures – including files whose paths don't change).
    /// Populated by <see cref="ModConverterService.PlanConversion"/> and used when
    /// creating a new mod from just this item's assets.
    /// </summary>
    public List<string> AllAssetFiles { get; } = new();

    // ── State ────────────────────────────────────────────────────────────────

    public bool IsPlanned    { get; set; } = false;
    public bool IsApplied    { get; set; } = false;
    public string? ErrorMessage { get; set; }
}

/// <summary>A single file or directory rename.</summary>
public class PlannedRename
{
    public string OldPath   { get; set; } = string.Empty;
    public string NewPath   { get; set; } = string.Empty;
    public bool   IsDir     { get; set; } = false;
    public bool   Selected  { get; set; } = true;
}

/// <summary>A collection of field-level changes inside a single JSON file.</summary>
public class PlannedJsonChange
{
    public string FilePath   { get; set; } = string.Empty;
    public List<JsonFieldChange> Changes { get; } = new();
    public bool Selected { get; set; } = true;
}

/// <summary>One field replacement within a JSON file.</summary>
public class JsonFieldChange
{
    public string JsonPath  { get; set; } = string.Empty;
    public string OldValue  { get; set; } = string.Empty;
    public string NewValue  { get; set; } = string.Empty;
    public string ChangeType{ get; set; } = string.Empty; // "path_key", "path_value", "numeric_id"
    public bool   Selected  { get; set; } = true;
}

/// <summary>A binary file with in-place string patches (.mdl / .mtrl).</summary>
public class PlannedBinaryPatch
{
    public string FilePath   { get; set; } = string.Empty;
    public List<BinaryStringPatch> Patches { get; } = new();
    public bool Selected { get; set; } = true;
}

/// <summary>One ASCII string replacement within a binary file.</summary>
public class BinaryStringPatch
{
    public string OldString { get; set; } = string.Empty;
    public string NewString { get; set; } = string.Empty;
    public bool   Selected  { get; set; } = true;
}

/// <summary>A remaining reference to the old item found after conversion.</summary>
public class LeftoverHit
{
    /// <summary>File or directory path where the leftover was found.</summary>
    public string FilePath  { get; set; } = string.Empty;

    /// <summary>"filename", "json", or "binary".</summary>
    public string HitType   { get; set; } = string.Empty;

    /// <summary>Human-readable description of what was found.</summary>
    public string Detail    { get; set; } = string.Empty;
}
