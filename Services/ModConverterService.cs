using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using AdvancedPenumbraItemConverter.Models;
using Dalamud.Plugin.Services;

namespace AdvancedPenumbraItemConverter.Services;

/// <summary>
/// Core conversion engine – C# port of the Python logic.
/// Handles planning (preview) and applying file / JSON / binary changes
/// to a Penumbra mod directory when retargeting from one item ID to another.
/// </summary>
public sealed class ModConverterService
{
    private static readonly string[] AllSlotSuffixes =
        { "_met", "_top", "_glv", "_dwn", "_sho", "_ear", "_nek", "_wrs", "_rir", "_ril" };

    private readonly IPluginLog        _log;
    private readonly GameDataService?  _gameData;

    public ModConverterService(IPluginLog log, GameDataService? gameData = null)
    {
        _log      = log;
        _gameData = gameData;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Validate that <paramref name="modDir"/> looks like a Penumbra mod folder.
    /// </summary>
    public (bool ok, string error) ValidateModDirectory(string modDir)
    {
        if (!Directory.Exists(modDir))
            return (false, $"Directory not found: {modDir}");

        var required = new[] { "default_mod.json", "meta.json" };
        var missing  = required.Where(f => !File.Exists(Path.Combine(modDir, f))).ToList();
        if (missing.Count > 0)
            return (false, $"Missing required file(s): {string.Join(", ", missing)}");

        return (true, string.Empty);
    }

    /// <summary>
    /// Plan all changes without touching disk.
    /// Populates task.PlannedRenames / PlannedJsonChanges / PlannedBinaryPatches
    /// and sets task.IsPlanned = true on success.
    /// </summary>
    public void PlanConversion(ConversionTask task)
    {
        task.PlannedRenames.Clear();
        task.PlannedJsonChanges.Clear();
        task.PlannedBinaryPatches.Clear();
        task.IsPlanned = false;
        task.IsApplied = false;
        task.ErrorMessage = null;

        try
        {
            var padOld    = PadId(task.OldIdPadded);
            var padNew    = PadId(task.NewIdPadded);
            if (padOld == padNew) { task.ErrorMessage = "Source and target IDs are identical."; return; }

            var slotKey   = SlotInfo.KeyMap[task.Slot];
            var prefix    = SlotInfo.ItemPrefix(task.Slot);
            var altPrefix = prefix == "a" ? "e" : "a";
            var oldToken  = $"{prefix}{padOld}";
            var newToken  = $"{prefix}{padNew}";
            var oldAlt    = $"{altPrefix}{padOld}";
            var newAlt    = $"{altPrefix}{padNew}";
            var slotFlag  = $"_{slotKey}";

            // For accessory cross-slot conversion (e.g. necklace → earring)
            var targetSlot     = task.TargetSlot ?? task.Slot;
            var targetSlotKey  = SlotInfo.KeyMap[targetSlot];
            var targetSlotFlag = $"_{targetSlotKey}";
            var baseDir   = task.ModDirectory;

            // ── 1. Build case-insensitive file index ──────────────────────────────
            var fileIndex = BuildFileIndex(baseDir);

            // ── 2. Find candidate .mdl files (asset chain root) ──────────────────
            var mdlCandidates = FindCandidateMdls(baseDir, padOld, slotKey, slotFlag, task.IgnoreSlot);

            // ── 3. Trace asset chain: .mdl → .mtrl → .tex ───────────────────────
            // Mirrors Python's preview_changes: reads binary content of each model to
            // extract referenced .mtrl paths, then each material for .tex paths.
            var mtrlRelPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var binaryHits   = new List<string>(); // files whose bytes contain the old token

            foreach (var mdl in mdlCandidates)
            {
                try
                {
                    var b = File.ReadAllBytes(mdl);
                    if (ContainsOldToken(b, oldToken, oldAlt)) binaryHits.Add(mdl);
                    foreach (var rel in ExtractAsciiPathsFromBytes(b, ".mtrl"))
                        mtrlRelPaths.Add(rel);
                }
                catch (Exception ex) { _log.Warning(ex, "[APIC] Could not read mdl {0}", mdl); }
            }

            var mtrlFiles = ResolveWithTokenFallback(
                fileIndex, mtrlRelPaths, oldToken, newToken, oldAlt, newAlt, baseDir);

            var texRelPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var mtrl in mtrlFiles)
            {
                try
                {
                    var b = File.ReadAllBytes(mtrl);
                    if (ContainsOldToken(b, oldToken, oldAlt)) binaryHits.Add(mtrl);
                    foreach (var rel in ExtractAsciiPathsFromBytes(b, ".tex"))
                        texRelPaths.Add(rel);
                }
                catch (Exception ex) { _log.Warning(ex, "[APIC] Could not read mtrl {0}", mtrl); }
            }

            var texFiles = ResolveWithTokenFallback(
                fileIndex, texRelPaths, oldToken, newToken, oldAlt, newAlt, baseDir);

            // ── 4. Pre-scan all JSON files; discover extra files referenced there ─
            // Use the same lenient options as ApplyJsonChange so that Penumbra group
            // files with trailing commas are not silently dropped during planning.
            var jsonDocOpts = new JsonDocumentOptions
            {
                AllowTrailingCommas = true,
                CommentHandling     = JsonCommentHandling.Skip,
            };
            var jsonFileCache = new List<(string path, JsonNode node)>();
            foreach (var jf in Directory.EnumerateFiles(baseDir, "*.json", SearchOption.AllDirectories))
            {
                try
                {
                    var node = JsonNode.Parse(File.ReadAllText(jf),
                                             nodeOptions:     new JsonNodeOptions(),
                                             documentOptions: jsonDocOpts);
                    if (node != null) jsonFileCache.Add((jf, node));
                }
                catch { /* skip malformed */ }
            }

            // Files referenced in JSON Files/FileSwaps (local mod paths) matching the token
            var jsonResolvedFiles = new List<string>();
            foreach (var (_, jnode) in jsonFileCache)
                foreach (var localPath in ExtractJsonFilesValues(jnode))
                {
                    var lLow = localPath.Replace('\\', '/').ToLower();
                    if (!lLow.Contains(oldToken.ToLower()) && !lLow.Contains(oldAlt.ToLower())) continue;
                    // Reject only if path explicitly belongs to a different slot
                    var hasAnySuffix = AllSlotSuffixes.Any(s => lLow.Contains(s));
                    if (hasAnySuffix && !task.IgnoreSlot && !lLow.Contains(slotFlag.ToLower())) continue;
                    jsonResolvedFiles.AddRange(ResolveManyCI(fileIndex, new[] { localPath }));
                }

            // ── 5. Plan per-file full-path renames ───────────────────────────────
            // Python (non-race path): replace tokens anywhere in each file's full path,
            // producing a single move operation that spans directory boundaries.
            var allAssets = mdlCandidates
                .Concat(mtrlFiles)
                .Concat(texFiles)
                .Concat(jsonResolvedFiles)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            // Store the full asset list so CreateNewModFromAssetChain can include
            // textures/materials whose paths don't contain the item token.
            task.AllAssetFiles.Clear();
            task.AllAssetFiles.AddRange(allAssets);

            var seenOld = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var filePath in allAssets)
            {
                if (!seenOld.Add(filePath)) continue;
                var newFullPath = ReplaceTokensCi(filePath, oldToken, newToken, oldAlt, newAlt, slotFlag, targetSlotFlag);
                if (string.Equals(newFullPath, filePath, StringComparison.OrdinalIgnoreCase)) continue;
                task.PlannedRenames.Add(new PlannedRename
                {
                    OldPath  = filePath,
                    NewPath  = newFullPath,
                    IsDir    = false,
                    Selected = true,
                });
            }

            // ── 6. Build json_scope from planned renames ──────────────────────────
            // Python builds scope from rename_changes + dir_renames and uses it to
            // constrain which JSON entries are modified.
            var (scopeFiles, scopeDirs) = BuildJsonScope(baseDir, task.PlannedRenames);

            // ── 7. Plan JSON metadata changes (scope-constrained) ─────────────────
            PlanJsonChanges(task, padOld, padNew, slotKey,
                            oldToken, newToken, oldAlt, newAlt,
                            scopeFiles, scopeDirs, jsonFileCache);

            // ── 8. Plan binary patches (only files in the asset chain) ────────────
            var dedupedBin = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var bf in binaryHits)
            {
                if (!dedupedBin.Add(bf)) continue;
                try
                {
                    var bytes   = File.ReadAllBytes(bf);
                    var patches = ExtractBinaryStringPatches(bytes, oldToken, newToken, oldAlt, newAlt, slotFlag, targetSlotFlag);
                    if (patches.Count == 0) continue;
                    var bp = new PlannedBinaryPatch { FilePath = bf, Selected = true };
                    foreach (var p in patches) bp.Patches.Add(p);
                    task.PlannedBinaryPatches.Add(bp);
                }
                catch (Exception ex) { _log.Warning(ex, "[APIC] Could not scan binary {0}", bf); }
            }

            // ── 9. EQP injection for mods with no existing EQP manipulation ───────
            // Equipment slots only (accessories have no EQP table).
            // If the mod carries no Eqp manipulation for the source item, the game
            // was using the vanilla EQP. After conversion the new item may have
            // different vanilla EQP, so we inject the old item's values explicitly.
            if (_gameData != null && !SlotInfo.IsAccessory(task.Slot))
            {
                var slotLabel9 = SlotInfo.LabelMap[task.Slot];
                int oldInt9    = int.Parse(padOld.TrimStart('0').PadLeft(1, '0'));
                int newInt9    = int.Parse(padNew.TrimStart('0').PadLeft(1, '0'));

                if (!HasEqpManipulation(jsonFileCache, oldInt9, slotLabel9))
                {
                    var eqpVal = _gameData.GetDefaultEqpEntry((ushort)oldInt9, task.Slot);
                    if (eqpVal.HasValue)
                    {
                        // Sentinel stored in OldValue: "<slotLabel>|<newSetId>" (used to
                        // skip the insert if the entry already exists during apply).
                        var sentinel = $"{slotLabel9}|{newInt9}";
                        var injJson  = $"{{\"Type\":\"Eqp\",\"Manipulation\":{{\"Entry\":{eqpVal.Value},\"SetId\":\"{newInt9}\",\"Slot\":\"{slotLabel9}\"}}}}"; 

                        var oldTokLow = oldToken.ToLower();
                        var oldAltLow = oldAlt.ToLower();

                        foreach (var (jsonFile9, node9) in jsonFileCache)
                        {
                            var manipPaths = new List<string>();
                            FindManipulationsForItem(node9, "<root>", oldTokLow, oldAltLow, manipPaths);
                            if (manipPaths.Count == 0) continue;

                            // Reuse or create a PlannedJsonChange for this file
                            var planned9 = task.PlannedJsonChanges
                                .FirstOrDefault(j => string.Equals(
                                    j.FilePath, jsonFile9, StringComparison.OrdinalIgnoreCase));
                            if (planned9 == null)
                            {
                                planned9 = new PlannedJsonChange { FilePath = jsonFile9, Selected = true };
                                task.PlannedJsonChanges.Add(planned9);
                            }

                            foreach (var mp in manipPaths)
                                planned9.Changes.Add(new JsonFieldChange
                                {
                                    JsonPath   = mp,
                                    OldValue   = sentinel,
                                    NewValue   = injJson,
                                    ChangeType = "eqp_insert",
                                    Selected   = true,
                                });

                            _log.Debug("[APIC] EQP injection planned: Entry={0} SetId={1} Slot={2} in {3}",
                                       eqpVal.Value, newInt9, slotLabel9, Path.GetFileName(jsonFile9));
                        }
                    }
                }
            }

            // ── 10. EQDP injection for mods with no existing EQDP manipulation ──────
            // Both equipment and accessory slots have per-race EQDP entries that control
            // whether each race/gender has a model for the item.  If the mod carries no
            // explicit Eqdp manipulation for the source item, the game was using the
            // vanilla EQDP. After conversion the new item may have different vanilla EQDP,
            // so we inject the source item's per-race values explicitly for the new item.
            if (_gameData != null)
            {
                var slotLabel10   = SlotInfo.LabelMap[task.Slot];
                var targetSlot10  = task.TargetSlot ?? task.Slot;
                var targetLabel10 = SlotInfo.LabelMap[targetSlot10];
                int oldInt10      = int.Parse(padOld.TrimStart('0').PadLeft(1, '0'));
                int newInt10      = int.Parse(padNew.TrimStart('0').PadLeft(1, '0'));

                if (!HasEqdpManipulation(jsonFileCache, oldInt10, slotLabel10))
                {
                    var eqdpEntries = _gameData.GetDefaultEqdpEntries((ushort)oldInt10, task.Slot);
                    if (eqdpEntries.Count > 0)
                    {
                        var oldTokLow10 = oldToken.ToLower();
                        var oldAltLow10 = oldAlt.ToLower();

                        foreach (var (jsonFile10, node10) in jsonFileCache)
                        {
                            var manipPaths10 = new List<string>();
                            FindManipulationsForItem(node10, "<root>", oldTokLow10, oldAltLow10, manipPaths10);
                            if (manipPaths10.Count == 0) continue;

                            var planned10 = task.PlannedJsonChanges
                                .FirstOrDefault(j => string.Equals(
                                    j.FilePath, jsonFile10, StringComparison.OrdinalIgnoreCase));
                            if (planned10 == null)
                            {
                                planned10 = new PlannedJsonChange { FilePath = jsonFile10, Selected = true };
                                task.PlannedJsonChanges.Add(planned10);
                            }

                            foreach (var (race10, gender10, entry10) in eqdpEntries)
                            {
                                var sentinel10 = $"{targetLabel10}|{newInt10}|{race10}|{gender10}";
                                var injJson10  = $"{{\"Type\":\"Eqdp\",\"Manipulation\":{{\"Entry\":{entry10},\"SetId\":{newInt10},\"Slot\":\"{targetLabel10}\",\"Race\":\"{race10}\",\"Gender\":\"{gender10}\"}}}}";

                                foreach (var mp in manipPaths10)
                                    planned10.Changes.Add(new JsonFieldChange
                                    {
                                        JsonPath   = mp,
                                        OldValue   = sentinel10,
                                        NewValue   = injJson10,
                                        ChangeType = "eqdp_insert",
                                        Selected   = true,
                                    });
                            }

                            _log.Debug("[APIC] EQDP injection planned: {0} entries for SetId={1} Slot={2} in {3}",
                                       eqdpEntries.Count, newInt10, targetLabel10, Path.GetFileName(jsonFile10));
                        }
                    }
                }
            }

            task.IsPlanned = true;
        }
        catch (Exception ex)
        {
            task.ErrorMessage = ex.Message;
            _log.Error(ex, "[APIC] PlanConversion failed");
        }
    }

    /// <summary>
    /// Apply all user-selected changes to disk.
    /// Calls Penumbra's ReloadMod via the supplied callback after write.
    /// </summary>
    public void ApplyConversion(ConversionTask task, Action<string>? onLog = null)
    {
        task.IsApplied = false;
        task.ErrorMessage = null;

        void Log(string msg) { onLog?.Invoke(msg); _log.Information("[APIC] {0}", msg); }

        try
        {
            // --- Binaries first (before renames move the files) ---
            foreach (var bp in task.PlannedBinaryPatches.Where(p => p.Selected))
                ApplyBinaryPatch(bp, Log);

            // --- JSON ---
            foreach (var jc in task.PlannedJsonChanges.Where(j => j.Selected))
                ApplyJsonChange(jc, Log);

            // --- Renames (dirs depth-first, files before dirs at each level) ---
            var renames = task.PlannedRenames
                .Where(r => r.Selected)
                .OrderByDescending(r => r.OldPath.Length); // deepest first

            foreach (var rename in renames)
                ApplyRename(rename, Log);

            // --- Remove empty directories left behind by file moves ---
            var padOldAc   = PadId(task.OldIdPadded);
            var prefixAc   = SlotInfo.ItemPrefix(task.Slot);
            var altPrefixAc = prefixAc == "a" ? "e" : "a";
            PruneEmptyOldTokenDirs(
                task.ModDirectory,
                $"{prefixAc}{padOldAc}",
                $"{altPrefixAc}{padOldAc}",
                Log);

            task.IsApplied = true;
            Log("Conversion applied successfully.");
        }
        catch (Exception ex)
        {
            task.ErrorMessage = ex.Message;
            _log.Error(ex, "[APIC] ApplyConversion failed");
            onLog?.Invoke($"Error: {ex.Message}");
        }
    }

    /// <summary>
    /// Creates a new mod by copying the original, applying the planned conversion to
    /// the copy, and updating <c>meta.json</c> with <paramref name="modDisplayName"/>.
    /// The original mod is left completely untouched.
    /// Returns the new mod directory path on success, or <c>null</c> on failure.
    /// </summary>
    public string? ApplyConversionAsNewMod(
        ConversionTask task,
        string         newModDir,
        string         modDisplayName,
        Action<string>? onLog = null)
    {
        task.IsApplied    = false;
        task.ErrorMessage = null;
        void Log(string msg) { onLog?.Invoke(msg); _log.Information("[APIC] {0}", msg); }

        try
        {
            // 1. Copy the entire source mod directory to the new location.
            Log($"Copying mod to: {newModDir}");
            CopyDirectory(task.ModDirectory, newModDir);
            Log($"Copied mod ({Directory.EnumerateFiles(newModDir, "*", SearchOption.AllDirectories).Count()} file(s)).");

            // 2. Remap the planned changes to point at the new directory.
            var remapped = RemapTask(task, newModDir);

            // 3. Apply the conversion in-place on the copy.
            ApplyConversion(remapped, Log);
            if (!remapped.IsApplied)
            {
                task.ErrorMessage = remapped.ErrorMessage;
                try { Directory.Delete(newModDir, recursive: true); } catch { }
                return null;
            }

            // 4. Stamp the new mod name into meta.json.
            UpdateMetaJsonName(newModDir, modDisplayName, Log);

            task.IsApplied = true;
            Log($"New mod created: {newModDir}");
            return newModDir;
        }
        catch (Exception ex)
        {
            task.ErrorMessage = ex.Message;
            _log.Error(ex, "[APIC] ApplyConversionAsNewMod failed");
            onLog?.Invoke($"Error: {ex.Message}");
            try { if (Directory.Exists(newModDir)) Directory.Delete(newModDir, recursive: true); } catch { }
            return null;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Post-conversion verification
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Scans the mod directory for any remaining mentions of the old item
    /// (by token) in file names, JSON content and binary files.
    /// Returns one <see cref="LeftoverHit"/> per distinct find.
    /// </summary>
    public List<LeftoverHit> VerifyConversion(ConversionTask task, Action<string>? onLog = null)
    {
        var hits = new List<LeftoverHit>();
        void Log(string msg) { onLog?.Invoke(msg); _log.Information("[APIC] {0}", msg); }

        try
        {
            var padOld   = PadId(task.OldIdPadded);
            var prefix   = SlotInfo.ItemPrefix(task.Slot);
            var altPrefix = prefix == "a" ? "e" : "a";
            var oldToken = $"{prefix}{padOld}";
            var oldAlt   = $"{altPrefix}{padOld}";
            var slotKey  = SlotInfo.KeyMap[task.Slot];
            var slotFlag = $"_{slotKey}".ToLower();

            var oldTokenLow = oldToken.ToLower();
            var oldAltLow   = oldAlt.ToLower();

            // ── 1. File / directory names ────────────────────────────────────
            foreach (var entry in EnumerateAll(task.ModDirectory))
            {
                var nameLow = Path.GetFileName(entry.FullName).ToLower();
                if (!nameLow.Contains(oldTokenLow) && !nameLow.Contains(oldAltLow)) continue;
                if (!task.IgnoreSlot && !entry.IsDirectory && !nameLow.Contains(slotFlag)) continue;

                var rel = RelativePath(task.ModDirectory, entry.FullName);
                hits.Add(new LeftoverHit
                {
                    FilePath = entry.FullName,
                    HitType  = "filename",
                    Detail   = $"Name still contains old token: {rel}",
                });
            }

            // ── 2. JSON files ────────────────────────────────────────────────
            foreach (var jsonFile in Directory.EnumerateFiles(
                task.ModDirectory, "*.json", SearchOption.AllDirectories))
            {
                try
                {
                    var raw     = File.ReadAllText(jsonFile);
                    var rawLow  = raw.ToLower();
                    var matches = new List<string>();

                    if (rawLow.Contains(oldTokenLow)) matches.Add(oldToken);
                    if (rawLow.Contains(oldAltLow))   matches.Add(oldAlt);
                    if (matches.Count == 0) continue;

                    var rel = RelativePath(task.ModDirectory, jsonFile);
                    hits.Add(new LeftoverHit
                    {
                        FilePath = jsonFile,
                        HitType  = "json",
                        Detail   = $"Still contains: {string.Join(", ", matches)}  in {rel}",
                    });
                }
                catch (Exception ex)
                {
                    _log.Warning(ex, "[APIC] Verify: could not read {0}", jsonFile);
                }
            }

            // ── 2b. JSON metadata — leftover numeric PrimaryId / SetId ──────────
            // The token-string check above cannot see integer values like
            // "PrimaryId": 687 because they don't contain the "e0687" token.
            // Walk the JSON tree explicitly to catch those.
            int oldInt = int.Parse(PadId(task.OldIdPadded).TrimStart('0').PadLeft(1, '0'));
            var slotLabelForVerify = SlotInfo.LabelMap[task.Slot];
            foreach (var jsonFile2 in Directory.EnumerateFiles(
                task.ModDirectory, "*.json", SearchOption.AllDirectories))
            {
                try
                {
                    var rawJ = File.ReadAllText(jsonFile2);
                    if (!HasLeftoverNumericMetaId(rawJ, oldInt, slotLabelForVerify, task.IgnoreSlot))
                        continue;

                    var relJ = RelativePath(task.ModDirectory, jsonFile2);
                    // Only add a new hit if this file was not already flagged by the token check
                    if (hits.Any(h => string.Equals(h.FilePath, jsonFile2,
                                                    StringComparison.OrdinalIgnoreCase))) continue;
                    hits.Add(new LeftoverHit
                    {
                        FilePath = jsonFile2,
                        HitType  = "json_metadata",
                        Detail   = $"Metadata still contains old numeric ID {oldInt} ({slotLabelForVerify}) in {relJ}",
                    });
                }
                catch (Exception ex)
                {
                    _log.Warning(ex, "[APIC] Verify: could not scan metadata in {0}", jsonFile2);
                }
            }

            // ── 3. Binary files (.mdl / .mtrl) ───────────────────────────────
            var binExts = new[] { ".mdl", ".mtrl" };
            foreach (var binFile in Directory
                .EnumerateFiles(task.ModDirectory, "*.*", SearchOption.AllDirectories)
                .Where(f => binExts.Contains(Path.GetExtension(f).ToLower())))
            {
                try
                {
                    var bytes       = File.ReadAllBytes(binFile);
                    var oldBytes    = Encoding.Latin1.GetBytes(oldTokenLow);
                    var oldAltBytes = Encoding.Latin1.GetBytes(oldAltLow);

                    bool hasOld    = IndexOf(bytes, oldBytes)    >= 0;
                    bool hasOldAlt = IndexOf(bytes, oldAltBytes) >= 0;
                    if (!hasOld && !hasOldAlt) continue;

                    var found = new List<string>();
                    if (hasOld)    found.Add(oldToken);
                    if (hasOldAlt) found.Add(oldAlt);

                    var rel = RelativePath(task.ModDirectory, binFile);
                    hits.Add(new LeftoverHit
                    {
                        FilePath = binFile,
                        HitType  = "binary",
                        Detail   = $"Still contains bytes for: {string.Join(", ", found)}  in {rel}",
                    });
                }
                catch (Exception ex)
                {
                    _log.Warning(ex, "[APIC] Verify: could not read binary {0}", binFile);
                }
            }
        }
        catch (Exception ex)
        {
            _log.Error(ex, "[APIC] VerifyConversion failed");
            Log($"Verification error: {ex.Message}");
        }

        return hits;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // New-mod helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Creates a new, minimal Penumbra mod that contains ONLY the files and metadata
    /// belonging to the converted item's asset chain (models, materials, textures needed
    /// by those materials, plus the matching EQP/EQDP manipulations).
    /// File mappings and manipulations from every JSON file in the source mod (including
    /// option-group files) are filtered and merged into a single <c>default_mod.json</c>.
    /// The source mod is left completely untouched.
    /// Returns the new mod directory path on success, or <c>null</c> on failure.
    /// </summary>
    public string? CreateNewModFromAssetChain(
        ConversionTask  task,
        string          newModDir,
        string          modDisplayName,
        Action<string>? onLog = null)
    {
        task.IsApplied    = false;
        task.ErrorMessage = null;
        void Log(string msg) { onLog?.Invoke(msg); _log.Information("[APIC] {0}", msg); }

        try
        {
            var sourceBase    = task.ModDirectory.TrimEnd('\\', '/');
            var jsonExts      = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".json" };
            var oldPathSet    = new HashSet<string>(
                task.PlannedRenames.Where(r => r.Selected).Select(r => r.OldPath),
                StringComparer.OrdinalIgnoreCase);
            var bpByPath      = task.PlannedBinaryPatches
                .Where(p => p.Selected)
                .ToDictionary(p => p.FilePath, StringComparer.OrdinalIgnoreCase);
            int targetVariant = task.TargetVariant > 0 ? task.TargetVariant : 1;

            Directory.CreateDirectory(newModDir);
            int fileCount = 0;

            // ── 1. Copy renamed asset files (applying binary patches + variant normalisation) ─
            // Sort so the source variant chosen by the user is processed first.
            // When multiple material variants normalise to the same destination
            // (e.g. v0001 and v0004 both → v0001 after NormalizeMaterialVariant),
            // the copy with the correct source variant wins the dedup check.
            foreach (var rename in task.PlannedRenames
                .Where(r => r.Selected)
                .OrderBy(r => MtrlVariantSortKey(r.OldPath, task.SourceVariant)))
            {
                if (jsonExts.Contains(Path.GetExtension(rename.OldPath))) continue;
                if (!File.Exists(rename.OldPath)) continue;

                // Normalise the material-variant folder in the destination path
                // (e.g. material/v0003/ → material/v0001/ when targetVariant == 1).
                var relNew   = NormalizeMaterialVariant(
                                   Path.GetRelativePath(sourceBase, rename.NewPath),
                                   targetVariant);
                var destPath = Path.Combine(newModDir, relNew);
                if (File.Exists(destPath)) continue;   // skip if already written (dedup)
                Directory.CreateDirectory(Path.GetDirectoryName(destPath)!);

                var bytes = File.ReadAllBytes(rename.OldPath);
                if (bpByPath.TryGetValue(rename.OldPath, out var bp))
                    bytes = ApplyPatchesToBytes(bytes, bp);
                // Patch embedded material-path strings (e.g. inside .mdl files).
                bytes = NormalizeBinaryMaterialVariant(bytes, targetVariant);
                File.WriteAllBytes(destPath, bytes);
                fileCount++;
            }

            // ── 2. Copy unchanged asset-chain files at the variant-normalised location ───
            foreach (var assetPath in task.AllAssetFiles)
            {
                if (oldPathSet.Contains(assetPath)) continue;   // already written above
                if (!File.Exists(assetPath)) continue;

                var relSrc   = NormalizeMaterialVariant(
                                   Path.GetRelativePath(sourceBase, assetPath),
                                   targetVariant);
                var destPath = Path.Combine(newModDir, relSrc);
                if (File.Exists(destPath)) continue;
                Directory.CreateDirectory(Path.GetDirectoryName(destPath)!);

                var bytes = File.ReadAllBytes(assetPath);
                if (bpByPath.TryGetValue(assetPath, out var bp))
                    bytes = ApplyPatchesToBytes(bytes, bp);
                bytes = NormalizeBinaryMaterialVariant(bytes, targetVariant);
                File.WriteAllBytes(destPath, bytes);
                fileCount++;
            }
            Log($"Copied {fileCount} asset file(s).");

            // ── 3. Build the set of local paths present in the new mod (NORMALISED) ───
            // Used to filter which Files-dict entries belong in the new mod.
            var includedLocalPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var r in task.PlannedRenames.Where(r => r.Selected))
            {
                if (jsonExts.Contains(Path.GetExtension(r.OldPath))) continue;
                includedLocalPaths.Add(
                    NormalizeMaterialVariant(
                        Path.GetRelativePath(sourceBase, r.NewPath).Replace('\\', '/'),
                        targetVariant));
            }
            foreach (var assetPath in task.AllAssetFiles)
            {
                if (oldPathSet.Contains(assetPath)) continue;
                includedLocalPaths.Add(
                    NormalizeMaterialVariant(
                        Path.GetRelativePath(sourceBase, assetPath).Replace('\\', '/'),
                        targetVariant));
            }

            // Shared variables needed by both step 3b and step 4.
            var padOld          = PadId(task.OldIdPadded);
            int oldInt          = int.Parse(padOld.TrimStart('0').PadLeft(1, '0'));
            var padNew          = PadId(task.NewIdPadded);
            int newInt          = int.Parse(padNew.TrimStart('0').PadLeft(1, '0'));
            var targetSlot      = task.TargetSlot ?? task.Slot;
            var sourceSlotLabel = SlotInfo.LabelMap[task.Slot];
            var slotLabel       = SlotInfo.LabelMap[targetSlot];   // target slot
            var newTokenLow     = $"{SlotInfo.ItemPrefix(targetSlot)}{padNew}".ToLower();

            var jsonDocOpts = new JsonDocumentOptions
            {
                AllowTrailingCommas = true,
                CommentHandling     = JsonCommentHandling.Skip,
            };
            var writeOpts = new JsonSerializerOptions { WriteIndented = true };

            // Build a lookup of planned changes keyed by source file path so that
            // files without planned changes (e.g. groups whose paths share no old token)
            // are still scanned — and files that DO have changes get them applied first.
            var jcByPath = task.PlannedJsonChanges
                .Where(j => j.Selected)
                .ToDictionary(j => j.FilePath, StringComparer.OrdinalIgnoreCase);

            // ── 3b. Expand includedLocalPaths with files referenced via game-path keys
            //        that belong to the new item (key contains newTokenLow after changes).
            //        The common case: textures/masks stored under a sibling item ID
            //        (e.g. e9068) in the source mod, referenced via old-item game-path
            //        redirects in the Files dict. Binary asset-chain tracing misses
            //        these because the game path in the binary does not match a physical
            //        file path in the mod directory.
            int extraFileCount = 0;
            foreach (var srcJsonEx in Directory.EnumerateFiles(
                sourceBase, "*.json", SearchOption.TopDirectoryOnly))
            {
                var fnameEx = Path.GetFileName(srcJsonEx);
                if (string.Equals(fnameEx, "meta.json", StringComparison.OrdinalIgnoreCase)) continue;

                try
                {
                    var nodeEx = JsonNode.Parse(File.ReadAllText(srcJsonEx),
                                     new JsonNodeOptions(), jsonDocOpts);
                    if (nodeEx == null) continue;

                    // Apply planned changes so renamed keys (e9069→e0387) are visible.
                    if (jcByPath.TryGetValue(srcJsonEx, out var jcEx))
                    {
                        foreach (var ch in jcEx.Changes.Where(c => c.Selected)
                                     .OrderBy(c => c.ChangeType == "path_key" ? 1 : 0))
                            ApplyJsonChangeAtPath(nodeEx, ch);
                    }

                    CollectLocalFilesForNewToken(nodeEx, newTokenLow, sourceBase, newModDir,
                                                 includedLocalPaths, bpByPath, ref extraFileCount,
                                                 targetVariant);
                }
                catch (Exception ex)
                {
                    Log($"[WARNING] Step 3b ({fnameEx}): {ex.Message}");
                }
            }
            if (extraFileCount > 0)
                Log($"Copied {extraFileCount} additional shared file(s) via game-path redirects.");

            // Build the set of item IDs that are actually referenced in this mod's
            // asset chain (e.g. e9068 sibling textures). Used to include IMC groups
            // that target those IDs so they get remapped to the new item.
            var referencedItemIds = new HashSet<int> { oldInt, newInt };
            var itemIdRx = new System.Text.RegularExpressions.Regex(
                @"[/\\][ea](\d{4})[/\\]",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase);
            foreach (var lp in includedLocalPaths)
            {
                var m = itemIdRx.Match(lp);
                if (m.Success && int.TryParse(m.Groups[1].Value, out var rid))
                    referencedItemIds.Add(rid);
            }

            // Build lowercase item tokens for sibling IDs (all IDs in the asset chain
            // except the new item itself). These are used to allow file entries for
            // e.g. e9068 to pass through FilterGroupJson / ExtractModEntriesRecursive
            // and to be picked up in the expanded step-3b scan.
            var siblingTokensLow = referencedItemIds
                .Where(id => id != newInt)
                .Select(id => $"e{id:D4}")
                .ToHashSet(StringComparer.OrdinalIgnoreCase);

            // ── 3c. Copy sibling-item local files referenced in group JSON ─────────
            // Extends step 3b to handle keys whose item segment is a sibling ID
            // (e.g. e9068) rather than the new item's token.  This copies files like
            // the Diffuse Option textures that are keyed under e9068 game paths.
            int siblingFileCount = 0;
            foreach (var srcJsonSib3 in Directory.EnumerateFiles(
                sourceBase, "*.json", SearchOption.TopDirectoryOnly))
            {
                var fnameSib = Path.GetFileName(srcJsonSib3);
                if (string.Equals(fnameSib, "meta.json", StringComparison.OrdinalIgnoreCase)) continue;

                try
                {
                    var nodeSib = JsonNode.Parse(File.ReadAllText(srcJsonSib3),
                                      new JsonNodeOptions(), jsonDocOpts);
                    if (nodeSib == null) continue;

                    if (jcByPath.TryGetValue(srcJsonSib3, out var jcSib))
                    {
                        foreach (var ch in jcSib.Changes.Where(c => c.Selected)
                                     .OrderBy(c => c.ChangeType == "path_key" ? 1 : 0))
                            ApplyJsonChangeAtPath(nodeSib, ch);
                    }

                    CollectLocalFilesForSiblingTokens(nodeSib, siblingTokensLow,
                                                      sourceBase, newModDir,
                                                      includedLocalPaths, bpByPath,
                                                      ref siblingFileCount, targetVariant);
                }
                catch (Exception ex)
                {
                    Log($"[WARNING] Step 3c ({fnameSib}): {ex.Message}");
                }
            }
            if (siblingFileCount > 0)
                Log($"Copied {siblingFileCount} sibling-item file(s) (e.g. diffuse options).");

            // ── 4. Build filtered JSON files (default_mod.json + option groups) ────
            var mergedFiles         = new JsonObject();
            var mergedFileSwaps     = new JsonObject();
            var mergedManipulations = new JsonArray();
            var seenManipulations   = new HashSet<string>();

            // Collect group files to write; written after the loop with renumbered
            // filenames (group_001_…, group_002_…, …) so sequential numbering is
            // maintained even when some source groups are filtered out.
            var pendingGroups = new List<(int origNum, string suffix, JsonObject node)>();

            // Enumerate ALL root-level JSON files so that option-group files are never
            // silently skipped when they happen to carry no old-token references of
            // their own (e.g. groups referencing only shared/unchanged texture paths).
            foreach (var srcJson in Directory.EnumerateFiles(
                sourceBase, "*.json", SearchOption.TopDirectoryOnly))
            {
                var fname = Path.GetFileName(srcJson);
                if (string.Equals(fname, "meta.json", StringComparison.OrdinalIgnoreCase)) continue;
                if (!File.Exists(srcJson)) continue;

                try
                {
                    var node = JsonNode.Parse(File.ReadAllText(srcJson),
                                   new JsonNodeOptions(), jsonDocOpts);
                    if (node == null) continue;

                    // Apply any planned changes into the in-memory tree before extracting.
                    if (jcByPath.TryGetValue(srcJson, out var jc))
                    {
                        foreach (var change in jc.Changes
                            .Where(c => c.Selected)
                            .OrderBy(c => c.ChangeType == "path_key" ? 1 : 0))
                            ApplyJsonChangeAtPath(node, change);
                    }

                    if (string.Equals(fname, "default_mod.json", StringComparison.OrdinalIgnoreCase))
                    {
                        // Extract flat always-on Files/FileSwaps/Manipulations only
                        // from the source default_mod.json — not from group files.
                        ExtractModEntriesRecursive(
                            node, includedLocalPaths, newTokenLow, newInt, slotLabel,
                            mergedFiles, mergedFileSwaps, mergedManipulations, seenManipulations,
                            targetVariant, siblingTokensLow);
                    }
                    else
                    {
                        // Option-group file: handle both regular (Files/FileSwaps) and
                        // IMC (Identifier/PrimaryId) group types.
                        var typ = (node as JsonObject)?["Type"]?.GetValue<string>() ?? string.Empty;
                        JsonObject? filtered;
                        if (string.Equals(typ, "Imc", StringComparison.OrdinalIgnoreCase))
                            // Only include IMC groups for the exact source item (oldInt).
                            // Sibling-item IMC groups (e.g. e9068 attribute toggles) are
                            // irrelevant to the converted item and must NOT be carried over.
                            filtered = TransformImcGroupJson(
                                node as JsonObject, oldInt, newInt,
                                sourceSlotLabel, slotLabel);
                        else
                            filtered = FilterGroupJson(
                                node, includedLocalPaths, newTokenLow, newInt, slotLabel,
                                targetVariant, siblingTokensLow);

                        if (filtered != null)
                        {
                            var (origNum, suffix) = ParseGroupFilename(fname);
                            pendingGroups.Add((origNum, suffix, filtered));
                        }
                    }
                }
                catch (Exception ex)
                {
                    Log($"[WARNING] Could not process {fname}: {ex.Message}");
                }
            }

            // Write group files with sequential 1-based numbering.
            pendingGroups.Sort((a, b) => a.origNum.CompareTo(b.origNum));
            for (int gi = 0; gi < pendingGroups.Count; gi++)
            {
                var (_, suffix, groupNode) = pendingGroups[gi];
                var destFname = $"group_{gi + 1:D3}_{suffix}.json";
                File.WriteAllText(
                    Path.Combine(newModDir, destFname),
                    groupNode.ToJsonString(writeOpts),
                    Encoding.UTF8);
                Log($"Preserved group: {destFname}");
            }
            Log($"Merged {mergedFiles.Count} default file mapping(s) and " +
                $"{mergedManipulations.Count} manipulation(s); " +
                $"preserved {pendingGroups.Count} option group file(s).");

            // ── 4b. Inject IMC variant-redirect manipulations ─────────────────────
            // For every variant of the new item, set MaterialId = targetVariant so the
            // game always loads material/v{targetVariant:D4}/ regardless of which
            // in-game variant the player has equipped.
            if (targetVariant > 0)
                AddImcVariantRedirects(newInt, targetSlot, targetVariant,
                                       mergedManipulations, seenManipulations, Log);

            // ── 5. Write default_mod.json ─────────────────────────────────────────
            var defaultMod = new JsonObject
            {
                ["Name"]          = "",
                ["Description"]   = "",
                ["Files"]         = mergedFiles,
                ["FileSwaps"]     = mergedFileSwaps,
                ["Manipulations"] = mergedManipulations,
            };
            File.WriteAllText(
                Path.Combine(newModDir, "default_mod.json"),
                defaultMod.ToJsonString(writeOpts),
                Encoding.UTF8);

            // ── 6. Write meta.json (copy source fields, override Name) ────────────
            var metaSrc  = Path.Combine(task.ModDirectory, "meta.json");
            var metaDest = Path.Combine(newModDir, "meta.json");
            if (File.Exists(metaSrc))
            {
                try
                {
                    var metaNode = JsonNode.Parse(File.ReadAllText(metaSrc),
                                       new JsonNodeOptions(), jsonDocOpts);
                    if (metaNode is JsonObject metaObj)
                    {
                        metaObj["Name"] = modDisplayName;
                        File.WriteAllText(metaDest, metaNode.ToJsonString(writeOpts), Encoding.UTF8);
                    }
                    else
                        WriteDefaultMetaJson(metaDest, modDisplayName, writeOpts);
                }
                catch { WriteDefaultMetaJson(metaDest, modDisplayName, writeOpts); }
            }
            else
            {
                WriteDefaultMetaJson(metaDest, modDisplayName, writeOpts);
            }

            task.IsApplied = true;
            Log($"New mod created (asset-chain only): {newModDir}");
            return newModDir;
        }
        catch (Exception ex)
        {
            task.ErrorMessage = ex.Message;
            _log.Error(ex, "[APIC] CreateNewModFromAssetChain failed");
            onLog?.Invoke($"Error: {ex.Message}");
            try { if (Directory.Exists(newModDir)) Directory.Delete(newModDir, recursive: true); } catch { }
            return null;
        }
    }

    /// <summary>
    /// Returns a shallow clone of <paramref name="original"/> with every absolute
    /// path remapped from the original mod directory to <paramref name="newBaseDir"/>.
    /// The clone is pre-marked as planned so <see cref="ApplyConversion"/> can run
    /// immediately without re-scanning the disk.
    /// </summary>
    private static ConversionTask RemapTask(ConversionTask original, string newBaseDir)
    {
        var oldBase = original.ModDirectory.TrimEnd('\\', '/');
        var newBase = newBaseDir.TrimEnd('\\', '/');

        string Remap(string p) =>
            p.StartsWith(oldBase, StringComparison.OrdinalIgnoreCase)
                ? newBase + p[oldBase.Length..]
                : p;

        var remapped = new ConversionTask
        {
            ModDirectory  = newBaseDir,
            Slot          = original.Slot,
            OldIdPadded   = original.OldIdPadded,
            NewIdPadded   = original.NewIdPadded,
            TargetSlot    = original.TargetSlot,
            IgnoreSlot    = original.IgnoreSlot,
            SourceVariant = original.SourceVariant,
            IsPlanned     = true,
        };

        foreach (var r in original.PlannedRenames)
            remapped.PlannedRenames.Add(new PlannedRename
            {
                OldPath  = Remap(r.OldPath),
                NewPath  = Remap(r.NewPath),
                IsDir    = r.IsDir,
                Selected = r.Selected,
            });

        foreach (var jc in original.PlannedJsonChanges)
        {
            var rc = new PlannedJsonChange { FilePath = Remap(jc.FilePath), Selected = jc.Selected };
            foreach (var c in jc.Changes)
                rc.Changes.Add(new JsonFieldChange
                {
                    JsonPath   = c.JsonPath,
                    OldValue   = c.OldValue,
                    NewValue   = c.NewValue,
                    ChangeType = c.ChangeType,
                    Selected   = c.Selected,
                });
            remapped.PlannedJsonChanges.Add(rc);
        }

        foreach (var bp in original.PlannedBinaryPatches)
        {
            var rb = new PlannedBinaryPatch { FilePath = Remap(bp.FilePath), Selected = bp.Selected };
            foreach (var p in bp.Patches)
                rb.Patches.Add(new BinaryStringPatch
                {
                    OldString = p.OldString,
                    NewString = p.NewString,
                    Selected  = p.Selected,
                });
            remapped.PlannedBinaryPatches.Add(rb);
        }

        return remapped;
    }

    /// <summary>Recursively copies <paramref name="sourceDir"/> to <paramref name="destDir"/>.</summary>
    private static void CopyDirectory(string sourceDir, string destDir)
    {
        Directory.CreateDirectory(destDir);
        foreach (var file in Directory.EnumerateFiles(sourceDir, "*", SearchOption.AllDirectories))
        {
            var rel      = Path.GetRelativePath(sourceDir, file);
            var destFile = Path.Combine(destDir, rel);
            Directory.CreateDirectory(Path.GetDirectoryName(destFile)!);
            File.Copy(file, destFile, overwrite: false);
        }
    }

    /// <summary>
    /// Reads the existing <c>meta.json</c> in <paramref name="modDir"/> and sets its
    /// <c>Name</c> field to <paramref name="displayName"/>, then writes it back.
    /// </summary>
    private void UpdateMetaJsonName(string modDir, string displayName, Action<string> log)
    {
        var metaPath = Path.Combine(modDir, "meta.json");
        try
        {
            if (!File.Exists(metaPath))
            {
                log("[WARNING] meta.json not found in new mod directory.");
                return;
            }

            var raw  = File.ReadAllText(metaPath);
            var node = JsonNode.Parse(raw,
                           new JsonNodeOptions(),
                           new JsonDocumentOptions { AllowTrailingCommas = true, CommentHandling = JsonCommentHandling.Skip });

            if (node is not JsonObject obj)
            {
                log("[WARNING] meta.json has unexpected format — Name not updated.");
                return;
            }

            obj["Name"] = displayName;
            var opts = new JsonSerializerOptions { WriteIndented = true };
            File.WriteAllText(metaPath, node.ToJsonString(opts), Encoding.UTF8);
            log($"Set mod display name to: {displayName}");
        }
        catch (Exception ex)
        {
            log($"[WARNING] Could not update meta.json: {ex.Message}");
            _log.Warning(ex, "[APIC] UpdateMetaJsonName failed for {0}", metaPath);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Planning helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Plan JSON changes using the pre-scanned cache and a scope derived from planned renames.
    /// Only edits JSON entries whose path values are within the scope (i.e. files being renamed).
    /// Files-dict KEYS (game paths) always bypass scope — they match Python's ignore_scope=True.
    /// </summary>
    private void PlanJsonChanges(
        ConversionTask task,
        string padOld, string padNew,
        string slotKey,
        string oldToken, string newToken,
        string oldAlt,   string newAlt,
        HashSet<string> scopeFiles, HashSet<string> scopeDirs,
        IReadOnlyList<(string path, JsonNode node)> jsonFileCache)
    {
        int oldInt    = int.Parse(padOld.TrimStart('0').PadLeft(1, '0'));
        int newInt    = int.Parse(padNew.TrimStart('0').PadLeft(1, '0'));
        var slotLabel = SlotInfo.LabelMap[task.Slot];
        var slotFlag  = $"_{slotKey}".ToLower();

        // Cross-slot support: determine target slot label and flag for slot-metadata replacement
        var targetSlot2      = task.TargetSlot ?? task.Slot;
        var targetSlotKey2   = SlotInfo.KeyMap[targetSlot2];
        var targetSlotLabel  = SlotInfo.LabelMap[targetSlot2];
        var targetSlotFlag2  = $"_{targetSlotKey2}";

        foreach (var (jsonFile, node) in jsonFileCache)
        {
            try
            {
                var changes = new List<JsonFieldChange>();
                CollectJsonChanges(node, "<root>", changes,
                                   oldToken, newToken, oldAlt, newAlt,
                                   oldInt, newInt, slotLabel, slotFlag, task.IgnoreSlot,
                                   scopeFiles, scopeDirs,
                                   targetSlotLabel, targetSlotFlag2);
                if (changes.Count == 0) continue;

                var planned = new PlannedJsonChange { FilePath = jsonFile, Selected = true };
                foreach (var c in changes) planned.Changes.Add(c);
                task.PlannedJsonChanges.Add(planned);
            }
            catch (Exception ex)
            {
                _log.Warning(ex, "[APIC] Could not process {0}", jsonFile);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Apply helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// After file moves, walk the mod directory bottom-up and delete any directory
    /// that is now empty and whose name contains the old item token (e.g. "e0687").
    /// </summary>
    private void PruneEmptyOldTokenDirs(
        string baseDir, string oldToken, string oldAlt, Action<string> log)
    {
        var oldTokenLow = oldToken.ToLower();
        var oldAltLow   = oldAlt.ToLower();

        // EnumerateDirectories with AllDirectories gives us all subdirs; reversing by
        // descending path length processes deepest children before their parents.
        var dirs = Directory
            .EnumerateDirectories(baseDir, "*", SearchOption.AllDirectories)
            .OrderByDescending(d => d.Length);

        foreach (var dir in dirs)
        {
            var nameLow = Path.GetFileName(dir).ToLower();
            if (!nameLow.Contains(oldTokenLow) && !nameLow.Contains(oldAltLow))
                continue;

            // Only delete if truly empty (no files, no subdirs remain).
            if (Directory.EnumerateFileSystemEntries(dir).Any())
                continue;

            try
            {
                Directory.Delete(dir);
                log($"Removed empty dir: {RelativePath(baseDir, dir)}");
            }
            catch (Exception ex)
            {
                _log.Warning(ex, "[APIC] Could not remove empty dir {0}", dir);
            }
        }
    }

    private void ApplyRename(PlannedRename rename, Action<string> log)
    {
        if (!rename.IsDir)
        {
            if (!File.Exists(rename.OldPath))
            {
                log($"[WARNING] Skipping (not found): {rename.OldPath}");
                return;
            }
            if (File.Exists(rename.NewPath))
            {
                log($"[WARNING] Skipping (destination exists): {rename.NewPath}");
                return;
            }
            // Ensure the destination directory exists — mirrors Python's
            // os.makedirs(os.path.dirname(new_path), exist_ok=True) before shutil.move.
            var destDir = Path.GetDirectoryName(rename.NewPath);
            if (!string.IsNullOrEmpty(destDir))
                Directory.CreateDirectory(destDir);
            var sameDir = string.Equals(
                Path.GetDirectoryName(rename.OldPath),
                Path.GetDirectoryName(rename.NewPath),
                StringComparison.OrdinalIgnoreCase);
            File.Move(rename.OldPath, rename.NewPath);
            log(sameDir
                ? $"Renamed: {Path.GetFileName(rename.OldPath)} → {Path.GetFileName(rename.NewPath)}"
                : $"Moved:   {rename.OldPath} → {rename.NewPath}");
        }
        else
        {
            if (!Directory.Exists(rename.OldPath))
            {
                log($"[WARNING] Skipping dir (not found): {rename.OldPath}");
                return;
            }
            if (Directory.Exists(rename.NewPath))
            {
                log($"[WARNING] Skipping dir (destination exists): {rename.NewPath}");
                return;
            }
            Directory.Move(rename.OldPath, rename.NewPath);
            log($"Renamed dir: {Path.GetFileName(rename.OldPath)} → {Path.GetFileName(rename.NewPath)}");
        }
    }

    private void ApplyJsonChange(PlannedJsonChange jc, Action<string> log)
    {
        var selectedChanges = jc.Changes.Where(c => c.Selected).ToList();
        if (selectedChanges.Count == 0) return;

        try
        {
            var raw  = File.ReadAllText(jc.FilePath);
            var node = JsonNode.Parse(raw, new JsonNodeOptions { PropertyNameCaseInsensitive = false },
                           new JsonDocumentOptions { AllowTrailingCommas = true, CommentHandling = JsonCommentHandling.Skip });
            if (node == null) return;

            int applied = 0;
            foreach (var change in selectedChanges.OrderBy(c => c.ChangeType == "path_key" ? 1 : 0))
            {
                if (ApplyJsonChangeAtPath(node, change))
                    applied++;
            }

            if (applied > 0)
            {
                var opts = new JsonSerializerOptions { WriteIndented = true };
                File.WriteAllText(jc.FilePath, node.ToJsonString(opts), Encoding.UTF8);
                log($"Updated JSON: {Path.GetFileName(jc.FilePath)} ({applied} changes)");
            }
        }
        catch (Exception ex)
        {
            log($"JSON update failed for {jc.FilePath}: {ex.Message}");
            _log.Error(ex, "[APIC] ApplyJsonChange failed for {0}", jc.FilePath);
        }
    }

    private static bool ApplyJsonChangeAtPath(JsonNode root, JsonFieldChange change)
    {
        try
        {
            // ── Special case: insert a new element into a Manipulations array ──────
            if (change.ChangeType == "eqp_insert")
                return ApplyEqpInsert(root, change);

            if (change.ChangeType == "eqdp_insert")
                return ApplyEqdpInsert(root, change);

            var path = change.JsonPath;
            if (!path.StartsWith("<root>")) return false;
            path = path[6..]; // strip "<root>"

            bool isKeyChange = path.EndsWith("[key]");
            if (isKeyChange) path = path[..^5];

            var segments = ParseJsonPath(path);
            if (segments.Count == 0) return false;

            // Navigate to the parent
            JsonNode? current = root;
            for (int i = 0; i < segments.Count - 1; i++)
            {
                var seg = segments[i];
                current = seg.Kind switch
                {
                    PathSegKind.Property => (current as JsonObject)?[seg.Name],
                    PathSegKind.Index    => (current as JsonArray)?[seg.Index],
                    PathSegKind.Key      => (current as JsonObject)?[seg.Name],
                    _ => null,
                };
                if (current == null) return false;
            }

            var last = segments[^1];

            if (isKeyChange)
            {
                // Navigate into the last property to find the dict
                var dictNode = (current as JsonObject)?[last.Name] as JsonObject;
                if (dictNode == null) return false;
                // Find the key that equals old value
                foreach (var kv in dictNode.ToList())
                {
                    if (string.Equals(kv.Key, change.OldValue, StringComparison.OrdinalIgnoreCase))
                    {
                        var val = dictNode[kv.Key];
                        dictNode.Remove(kv.Key);
                        dictNode[change.NewValue] = val;
                        return true;
                    }
                }
                return false;
            }

            switch (last.Kind)
            {
                case PathSegKind.Property:
                {
                    var obj = current as JsonObject;
                    if (obj == null) return false;
                    var val = obj[last.Name];
                    if (val == null) return false;
                    if (change.ChangeType == "numeric_id")
                    {
                        var intVal = val.GetValue<int>();
                        if (intVal == int.Parse(change.OldValue))
                        {
                            obj[last.Name] = int.Parse(change.NewValue);
                            return true;
                        }
                    }
                    else if (change.ChangeType == "numeric_id_string")
                    {
                        // SetId/PrimaryId serialised as a quoted string — preserve that format.
                        var strVal = val.GetValue<string>();
                        if (int.TryParse(strVal, out var intVal) && intVal == int.Parse(change.OldValue))
                        {
                            obj[last.Name] = change.NewValue;
                            return true;
                        }
                    }
                    else
                    {
                        var strVal = val.GetValue<string>();
                        var replaced = ReplaceInString(strVal, change.OldValue, change.NewValue);
                        if (replaced != strVal) { obj[last.Name] = replaced; return true; }
                    }
                    return false;
                }
                case PathSegKind.Key:
                {
                    var obj = current as JsonObject;
                    if (obj == null) return false;
                    var val = obj[last.Name];
                    if (val == null) return false;
                    var strVal = val.GetValue<string>();
                    var replaced = ReplaceInString(strVal, change.OldValue, change.NewValue);
                    if (replaced != strVal) { obj[last.Name] = replaced; return true; }
                    return false;
                }
                default:
                    return false;
            }
        }
        catch { return false; }
    }

    private void ApplyBinaryPatch(PlannedBinaryPatch bp, Action<string> log)
    {
        var selectedPatches = bp.Patches.Where(p => p.Selected).ToList();
        if (selectedPatches.Count == 0) return;

        try
        {
            var bytes = File.ReadAllBytes(bp.FilePath);
            int count = 0;
            foreach (var patch in selectedPatches)
            {
                var oldBytes = Encoding.Latin1.GetBytes(patch.OldString);
                var newBytes = Encoding.Latin1.GetBytes(patch.NewString);

                if (oldBytes.Length != newBytes.Length)
                {
                    // Pad shorter with nulls (same length required for binary patching)
                    int maxLen = Math.Max(oldBytes.Length, newBytes.Length);
                    Array.Resize(ref oldBytes, maxLen);
                    Array.Resize(ref newBytes, maxLen);
                }

                int replaced = ReplaceBytesInArray(ref bytes, oldBytes, newBytes);
                count += replaced;
            }
            if (count > 0)
            {
                File.WriteAllBytes(bp.FilePath, bytes);
                log($"Patched binary: {Path.GetFileName(bp.FilePath)} ({count} replacement(s))");
            }
        }
        catch (Exception ex)
        {
            log($"Binary patch failed: {bp.FilePath}: {ex.Message}");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // JSON walking
    // ─────────────────────────────────────────────────────────────────────────

    private void CollectJsonChanges(
        JsonNode? node, string path,
        List<JsonFieldChange> changes,
        string oldToken, string newToken,
        string oldAlt,   string newAlt,
        int oldInt, int newInt,
        string slotLabel, string slotFlag,
        bool ignoreSlot,
        HashSet<string> scopeFiles, HashSet<string> scopeDirs,
        string targetSlotLabel = "", string targetSlotFlag = "")
    {
        switch (node)
        {
            case JsonObject obj:
            {
                // Detect slot match for numeric ID fields
                bool slotMatch = false;
                foreach (var sf in new[] { "EquipSlot", "Slot" })
                {
                    if (obj[sf]?.GetValue<string>() is { } sv &&
                        string.Equals(sv, slotLabel, StringComparison.OrdinalIgnoreCase))
                    {
                        slotMatch = true;
                        break;
                    }
                }

                // Cross-slot: when this object belongs to the source slot and the output
                // slot differs, also emit a change for the EquipSlot/Slot field itself.
                if (slotMatch && !string.IsNullOrEmpty(targetSlotLabel) &&
                    !string.Equals(slotLabel, targetSlotLabel, StringComparison.OrdinalIgnoreCase))
                {
                    foreach (var sf in new[] { "EquipSlot", "Slot" })
                    {
                        if (obj[sf]?.GetValue<string>() is { } sv &&
                            string.Equals(sv, slotLabel, StringComparison.OrdinalIgnoreCase))
                        {
                            changes.Add(new JsonFieldChange
                            {
                                JsonPath   = $"{path}.{sf}",
                                OldValue   = sv,
                                NewValue   = targetSlotLabel,
                                ChangeType = "path_string",
                                Selected   = true,
                            });
                            break;
                        }
                    }
                }

                foreach (var kv in obj)
                {
                    var subPath = $"{path}.{kv.Key}";

                    if ((kv.Key == "Files" || kv.Key == "FileSwaps") && kv.Value is JsonObject filesObj)
                    {
                        CollectFilesDictChanges(filesObj, subPath, changes,
                            oldToken, newToken, oldAlt, newAlt, slotFlag, ignoreSlot,
                            scopeFiles, scopeDirs, targetSlotFlag);
                        continue;
                    }

                    if ((kv.Key == "PrimaryId" || kv.Key == "SetId") && slotMatch)
                    {
                        if (kv.Value?.GetValueKind() == JsonValueKind.Number)
                        {
                            var iv = kv.Value.GetValue<int>();
                            if (iv == oldInt)
                                changes.Add(new JsonFieldChange
                                {
                                    JsonPath   = subPath,
                                    OldValue   = oldInt.ToString(),
                                    NewValue   = newInt.ToString(),
                                    ChangeType = "numeric_id",
                                    Selected   = true,
                                });
                        }
                        else if (kv.Value?.GetValueKind() == JsonValueKind.String)
                        {
                            // Penumbra serialises SetId/PrimaryId as a quoted string (e.g. "687").
                            var sv = kv.Value.GetValue<string>();
                            if (int.TryParse(sv, out var iv) && iv == oldInt)
                                changes.Add(new JsonFieldChange
                                {
                                    JsonPath   = subPath,
                                    OldValue   = oldInt.ToString(),
                                    NewValue   = newInt.ToString(),
                                    ChangeType = "numeric_id_string",
                                    Selected   = true,
                                });
                        }
                        continue;
                    }

                    CollectJsonChanges(kv.Value, subPath, changes,
                        oldToken, newToken, oldAlt, newAlt,
                        oldInt, newInt, slotLabel, slotFlag, ignoreSlot,
                        scopeFiles, scopeDirs, targetSlotLabel, targetSlotFlag);
                }
                break;
            }
            case JsonArray arr:
            {
                for (int i = 0; i < arr.Count; i++)
                    CollectJsonChanges(arr[i], $"{path}[{i}]", changes,
                        oldToken, newToken, oldAlt, newAlt,
                        oldInt, newInt, slotLabel, slotFlag, ignoreSlot,
                        scopeFiles, scopeDirs, targetSlotLabel, targetSlotFlag);
                break;
            }
            case JsonValue val when val.GetValueKind() == JsonValueKind.String:
            {
                var str = val.GetValue<string>();
                CheckStringChange(str, path, changes, oldToken, newToken, oldAlt, newAlt,
                                  slotFlag, ignoreSlot, "path_string", scopeFiles, scopeDirs,
                                  targetSlotFlag);
                break;
            }
        }
    }

    private void CollectFilesDictChanges(
        JsonObject filesObj, string basePath,
        List<JsonFieldChange> changes,
        string oldToken, string newToken,
        string oldAlt,   string newAlt,
        string slotFlag, bool ignoreSlot,
        HashSet<string> scopeFiles, HashSet<string> scopeDirs,
        string targetSlotFlag = "")
    {
        foreach (var kv in filesObj)
        {
            var fPath = $"{basePath}[{kv.Key}]";

            // Check key (game path → could change).
            // Keys bypass scope — Python uses ignore_scope=True for game paths.
            var keyLow = kv.Key.ToLower();
            bool keyHas = keyLow.Contains(oldToken.ToLower()) || keyLow.Contains(oldAlt.ToLower());
            if (keyHas && PathSlotOk(kv.Key, slotFlag, ignoreSlot))
            {
                var newKey = ReplaceTokensCi(kv.Key, oldToken, newToken, oldAlt, newAlt, slotFlag, targetSlotFlag);
                if (newKey != kv.Key)
                    changes.Add(new JsonFieldChange
                    {
                        JsonPath   = $"{basePath}[key]",
                        OldValue   = kv.Key,
                        NewValue   = newKey,
                        ChangeType = "path_key",
                        Selected   = true,
                    });
            }

            // Check value — subject to scope filtering (mirrors Python's scope check on Files values).
            if (kv.Value?.GetValueKind() == JsonValueKind.String)
            {
                var val = kv.Value.GetValue<string>();
                CheckStringChange(val, fPath, changes, oldToken, newToken, oldAlt, newAlt,
                                  slotFlag, ignoreSlot, "path_value", scopeFiles, scopeDirs,
                                  targetSlotFlag);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // EQDP insertion helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Navigates to the Manipulations array identified by <paramref name="change"/>.JsonPath,
    /// checks for a duplicate Eqdp entry (Slot + SetId + Race + Gender), and appends
    /// the new element if not already present.
    /// </summary>
    private static bool ApplyEqdpInsert(JsonNode root, JsonFieldChange change)
    {
        var arrPath = change.JsonPath;
        if (!arrPath.StartsWith("<root>")) return false;
        arrPath = arrPath[6..];

        var segs = ParseJsonPath(arrPath);
        if (segs.Count == 0) return false;

        JsonNode? curr = root;
        for (int i = 0; i < segs.Count - 1; i++)
        {
            curr = segs[i].Kind switch
            {
                PathSegKind.Property => (curr as JsonObject)?[segs[i].Name],
                PathSegKind.Index    => (curr as JsonArray)?[segs[i].Index],
                _ => null,
            };
            if (curr == null) return false;
        }

        // Navigate to the terminal array, creating it if it is absent on a JsonObject parent.
        var termSeg = segs[^1];
        if (termSeg.Kind == PathSegKind.Property && curr is JsonObject parentObj)
        {
            if (parentObj[termSeg.Name] is not JsonArray)
            {
                if (parentObj[termSeg.Name] != null) return false; // exists but wrong type
                parentObj[termSeg.Name] = new JsonArray();
            }
            curr = parentObj[termSeg.Name];
        }
        else
        {
            curr = termSeg.Kind switch
            {
                PathSegKind.Property => (curr as JsonObject)?[termSeg.Name],
                PathSegKind.Index    => (curr as JsonArray)?[termSeg.Index],
                _ => null,
            };
        }

        if (curr is not JsonArray manipArr) return false;

        // Duplicate guard: sentinel is "Slot|newSetId|Race|Gender"
        var sp = change.OldValue.Split('|');
        if (sp.Length == 4 && int.TryParse(sp[1], out var chkSetId))
        {
            var chkSlot = sp[0]; var chkRace = sp[2]; var chkGender = sp[3];
            foreach (var item in manipArr)
            {
                if (item is not JsonObject mobj) continue;
                if (mobj["Type"]?.GetValue<string>() != "Eqdp") continue;
                var mm = mobj["Manipulation"] as JsonObject;
                if (mm == null) continue;

                bool slotOk   = string.Equals(mm["Slot"]?.GetValue<string>(),   chkSlot,   StringComparison.OrdinalIgnoreCase);
                bool raceOk   = string.Equals(mm["Race"]?.GetValue<string>(),   chkRace,   StringComparison.OrdinalIgnoreCase);
                bool genderOk = string.Equals(mm["Gender"]?.GetValue<string>(), chkGender, StringComparison.OrdinalIgnoreCase);
                var  mSet     = mm["SetId"];
                bool setOk    = (mSet?.GetValueKind() == JsonValueKind.Number   && mSet.GetValue<int>() == chkSetId) ||
                                (mSet?.GetValueKind() == JsonValueKind.String   &&
                                 int.TryParse(mSet.GetValue<string>(), out var si) && si == chkSetId);

                if (slotOk && raceOk && genderOk && setOk) return false; // already present
            }
        }

        var newNode = JsonNode.Parse(change.NewValue);
        if (newNode == null) return false;
        manipArr.Add(newNode);
        return true;
    }

    /// <summary>Returns true if any JSON file already contains an Eqdp manipulation
    /// for <paramref name="oldSetId"/> + <paramref name="slotLabel"/> (any race/gender).</summary>
    private static bool HasEqdpManipulation(
        IReadOnlyList<(string path, JsonNode node)> jsonCache,
        int oldSetId, string slotLabel)
    {
        foreach (var (_, node) in jsonCache)
            if (WalkForEqdpManipulation(node, oldSetId, slotLabel))
                return true;
        return false;
    }

    private static bool WalkForEqdpManipulation(JsonNode? node, int oldSetId, string slotLabel)
    {
        if (node is JsonObject obj)
        {
            if (obj["Type"]?.GetValue<string>() == "Eqdp")
            {
                if (obj["Manipulation"] is JsonObject manip)
                {
                    var mSlot = manip["Slot"]?.GetValue<string>();
                    if (string.Equals(mSlot, slotLabel, StringComparison.OrdinalIgnoreCase))
                    {
                        var mSet = manip["SetId"];
                        if (mSet?.GetValueKind() == JsonValueKind.Number &&
                            mSet.GetValue<int>() == oldSetId)
                            return true;
                        if (mSet?.GetValueKind() == JsonValueKind.String &&
                            int.TryParse(mSet.GetValue<string>(), out var si) && si == oldSetId)
                            return true;
                    }
                }
            }
            foreach (var kv in obj)
                if (WalkForEqdpManipulation(kv.Value, oldSetId, slotLabel)) return true;
        }
        else if (node is JsonArray arr)
        {
            foreach (var item in arr)
                if (WalkForEqdpManipulation(item, oldSetId, slotLabel)) return true;
        }
        return false;
    }

    // EQP insertion helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Navigates to the Manipulations array identified by <paramref name="change"/>.JsonPath,
    /// checks for an existing Eqp entry with the same SetId+Slot (using OldValue as
    /// "slotLabel|newSetId" sentinel), and appends the new element if not already present.
    /// </summary>
    private static bool ApplyEqpInsert(JsonNode root, JsonFieldChange change)
    {
        var arrPath = change.JsonPath;
        if (!arrPath.StartsWith("<root>")) return false;
        arrPath = arrPath[6..];

        var segs = ParseJsonPath(arrPath);
        if (segs.Count == 0) return false;

        JsonNode? curr = root;
        for (int i = 0; i < segs.Count - 1; i++)
        {
            curr = segs[i].Kind switch
            {
                PathSegKind.Property => (curr as JsonObject)?[segs[i].Name],
                PathSegKind.Index    => (curr as JsonArray)?[segs[i].Index],
                _ => null,
            };
            if (curr == null) return false;
        }

        // Navigate to the terminal array, creating it if it is absent on a JsonObject parent.
        var termSeg = segs[^1];
        if (termSeg.Kind == PathSegKind.Property && curr is JsonObject parentObj)
        {
            if (parentObj[termSeg.Name] is not JsonArray)
            {
                if (parentObj[termSeg.Name] != null) return false; // exists but wrong type
                parentObj[termSeg.Name] = new JsonArray();
            }
            curr = parentObj[termSeg.Name];
        }
        else
        {
            curr = termSeg.Kind switch
            {
                PathSegKind.Property => (curr as JsonObject)?[termSeg.Name],
                PathSegKind.Index    => (curr as JsonArray)?[termSeg.Index],
                _ => null,
            };
        }

        if (curr is not JsonArray manipArr) return false;

        // Duplicate guard: parse "slotLabel|newSetId" from OldValue
        var sentinelParts = change.OldValue.Split('|');
        if (sentinelParts.Length == 2 &&
            int.TryParse(sentinelParts[1], out var chkSetId))
        {
            var chkSlot = sentinelParts[0];
            foreach (var item in manipArr)
            {
                if (item is not JsonObject mobj) continue;
                if (mobj["Type"]?.GetValue<string>() != "Eqp") continue;
                var mm = mobj["Manipulation"] as JsonObject;
                if (mm == null) continue;
                var mSlot = mm["Slot"]?.GetValue<string>();
                var mSet  = mm["SetId"];
                bool slotMatch = string.Equals(mSlot, chkSlot, StringComparison.OrdinalIgnoreCase);
                bool setMatch  = (mSet?.GetValueKind() == JsonValueKind.String &&
                                  int.TryParse(mSet.GetValue<string>(), out var si) && si == chkSetId) ||
                                 (mSet?.GetValueKind() == JsonValueKind.Number &&
                                  mSet.GetValue<int>() == chkSetId);
                if (slotMatch && setMatch) return false; // already present
            }
        }

        var newNode = JsonNode.Parse(change.NewValue);
        if (newNode == null) return false;
        manipArr.Add(newNode);
        return true;
    }

    /// <summary>
    /// Returns true if any JSON file in <paramref name="jsonCache"/> already contains an
    /// Eqp manipulation for <paramref name="oldSetId"/> + <paramref name="slotLabel"/>.
    /// </summary>
    private static bool HasEqpManipulation(
        IReadOnlyList<(string path, JsonNode node)> jsonCache,
        int oldSetId, string slotLabel)
    {
        foreach (var (_, node) in jsonCache)
            if (WalkForEqpManipulation(node, oldSetId, slotLabel))
                return true;
        return false;
    }

    private static bool WalkForEqpManipulation(JsonNode? node, int oldSetId, string slotLabel)
    {
        if (node is JsonObject obj)
        {
            if (obj["Type"]?.GetValue<string>() == "Eqp")
            {
                if (obj["Manipulation"] is JsonObject manip)
                {
                    var mSlot = manip["Slot"]?.GetValue<string>();
                    if (string.Equals(mSlot, slotLabel, StringComparison.OrdinalIgnoreCase))
                    {
                        var mSet = manip["SetId"];
                        if (mSet?.GetValueKind() == JsonValueKind.Number &&
                            mSet.GetValue<int>() == oldSetId)
                            return true;
                        if (mSet?.GetValueKind() == JsonValueKind.String &&
                            int.TryParse(mSet.GetValue<string>(), out var si) && si == oldSetId)
                            return true;
                    }
                }
            }
            foreach (var kv in obj)
                if (WalkForEqpManipulation(kv.Value, oldSetId, slotLabel)) return true;
        }
        else if (node is JsonArray arr)
        {
            foreach (var item in arr)
                if (WalkForEqpManipulation(item, oldSetId, slotLabel)) return true;
        }
        return false;
    }

    /// <summary>
    /// Recursively finds the JSON paths of every <c>Manipulations</c> array that lives
    /// inside an option object whose <c>Files</c> or <c>FileSwaps</c> dict references
    /// either <paramref name="oldTokenLow"/> or <paramref name="oldAltLow"/>.
    /// </summary>
    private static void FindManipulationsForItem(
        JsonNode? node, string path,
        string oldTokenLow, string oldAltLow,
        List<string> manipPaths)
    {
        if (node is JsonObject obj)
        {
            bool hasOldFiles = false;
            foreach (var filesKey in new[] { "Files", "FileSwaps" })
            {
                if (obj[filesKey] is JsonObject filesObj)
                {
                    foreach (var kv in filesObj)
                    {
                        var kl = kv.Key.ToLower();
                        var vl = kv.Value?.GetValueKind() == JsonValueKind.String
                            ? kv.Value.GetValue<string>().ToLower() : string.Empty;
                        if (kl.Contains(oldTokenLow) || kl.Contains(oldAltLow) ||
                            vl.Contains(oldTokenLow) || vl.Contains(oldAltLow))
                        { hasOldFiles = true; break; }
                    }
                }
                if (hasOldFiles) break;
            }

            if (hasOldFiles)
                manipPaths.Add($"{path}.Manipulations");

            foreach (var kv in obj)
                FindManipulationsForItem(kv.Value, $"{path}.{kv.Key}",
                                         oldTokenLow, oldAltLow, manipPaths);
        }
        else if (node is JsonArray arr)
        {
            for (int i = 0; i < arr.Count; i++)
                FindManipulationsForItem(arr[i], $"{path}[{i}]",
                                         oldTokenLow, oldAltLow, manipPaths);
        }
    }

    // Numeric-metadata leftover helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Returns true if <paramref name="jsonText"/> contains a Penumbra manipulation
    /// object whose PrimaryId or SetId still equals <paramref name="oldInt"/> for the
    /// matching slot.  Uses lenient parse options so trailing-comma files work.
    /// </summary>
    private static bool HasLeftoverNumericMetaId(
        string jsonText, int oldInt, string slotLabel, bool ignoreSlot)
    {
        try
        {
            var node = JsonNode.Parse(jsonText,
                nodeOptions:     new JsonNodeOptions(),
                documentOptions: new JsonDocumentOptions
                {
                    AllowTrailingCommas = true,
                    CommentHandling     = JsonCommentHandling.Skip,
                });
            return node != null && WalkForNumericMetaId(node, oldInt, slotLabel, ignoreSlot);
        }
        catch { return false; }
    }

    /// <summary>
    /// Recursive tree walk: returns true as soon as a PrimaryId/SetId == <paramref name="oldInt"/>
    /// is found inside an object whose EquipSlot/Slot matches (or slot filtering is off).
    /// </summary>
    private static bool WalkForNumericMetaId(
        JsonNode? node, int oldInt, string slotLabel, bool ignoreSlot)
    {
        if (node is JsonObject obj)
        {
            bool slotMatch = ignoreSlot;
            if (!slotMatch)
            {
                foreach (var sf in new[] { "EquipSlot", "Slot" })
                {
                    if (obj[sf]?.GetValue<string>() is { } sv &&
                        string.Equals(sv, slotLabel, StringComparison.OrdinalIgnoreCase))
                    { slotMatch = true; break; }
                }
            }

            if (slotMatch)
            {
                foreach (var idField in new[] { "PrimaryId", "SetId" })
                {
                    var idNode = obj[idField];
                    if (idNode == null) continue;
                    if (idNode.GetValueKind() == JsonValueKind.Number &&
                        idNode.GetValue<int>() == oldInt)
                        return true;
                    // Also handle quoted-string form: "SetId": "687"
                    if (idNode.GetValueKind() == JsonValueKind.String &&
                        int.TryParse(idNode.GetValue<string>(), out var sv) &&
                        sv == oldInt)
                        return true;
                }
            }

            foreach (var kv in obj)
                if (WalkForNumericMetaId(kv.Value, oldInt, slotLabel, ignoreSlot))
                    return true;
        }
        else if (node is JsonArray arr)
        {
            foreach (var item in arr)
                if (WalkForNumericMetaId(item, oldInt, slotLabel, ignoreSlot))
                    return true;
        }
        return false;
    }

    private static void CheckStringChange(
        string value, string path,
        List<JsonFieldChange> changes,
        string oldToken, string newToken,
        string oldAlt,   string newAlt,
        string slotFlag, bool ignoreSlot,
        string changeType,
        HashSet<string> scopeFiles, HashSet<string> scopeDirs,
        string targetSlotFlag = "")
    {
        var vl = value.ToLower();
        bool has = vl.Contains(oldToken.ToLower()) || vl.Contains(oldAlt.ToLower());
        if (!has) return;
        if (!PathSlotOk(value, slotFlag, ignoreSlot)) return;

        // Scope filter: only replace paths that correspond to files being renamed.
        // Non-path plain strings (no slash) always pass — only path-like strings are scoped.
        // Mirrors Python's _replace_path_string scope check (has_effective_scope logic).
        bool looksLikePath = value.Contains('/') || value.Contains('\\');
        if (looksLikePath && !PathInJsonScope(value, scopeFiles, scopeDirs)) return;

        var replaced = ReplaceTokensCi(value, oldToken, newToken, oldAlt, newAlt, slotFlag, targetSlotFlag);
        if (replaced == value) return;

        changes.Add(new JsonFieldChange
        {
            JsonPath   = path,
            OldValue   = value,
            NewValue   = replaced,
            ChangeType = changeType,
            Selected   = true,
        });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Binary helpers
    // ─────────────────────────────────────────────────────────────────────────

    private static List<BinaryStringPatch> ExtractBinaryStringPatches(
        byte[] data,
        string oldToken, string newToken,
        string oldAlt,   string newAlt,
        string slotFlag, string targetSlotFlag = "")
    {
        var result   = new List<BinaryStringPatch>();
        var seen     = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var allowed  = BuildAllowedSet();

        var tokens = new[] { (oldToken.ToLower(), newToken), (oldAlt.ToLower(), newAlt) };

        foreach (var (searchLow, replacement) in tokens)
        {
            var searchBytes = Encoding.Latin1.GetBytes(searchLow);
            int idx = 0;
            while (true)
            {
                int pos = IndexOf(data, searchBytes, idx);
                if (pos < 0) break;
                idx = pos + 1;

                // Extract surrounding ASCII word
                int start = pos - 1;
                while (start >= 0 && allowed.Contains(data[start])) start--;
                start++;

                int end = pos + searchBytes.Length;
                while (end < data.Length && allowed.Contains(data[end])) end++;

                var seg      = data[start..end];
                var segStr   = Encoding.Latin1.GetString(seg);
                var segStrLow = segStr.ToLower().Replace('\\', '/');

                if (seen.Contains(segStr)) continue;
                seen.Add(segStr);

                if (!PathSlotOk(segStr, slotFlag, false)) continue;

                var replaced = ReplaceTokensCi(segStr, oldToken, replacement, oldAlt, newAlt, slotFlag, targetSlotFlag);
                if (replaced == segStr) continue;

                result.Add(new BinaryStringPatch
                {
                    OldString = segStr,
                    NewString = replaced,
                    Selected  = true,
                });
            }
        }

        return result;
    }

    private static int ReplaceBytesInArray(ref byte[] data, byte[] oldBytes, byte[] newBytes)
    {
        int count = 0;
        int idx   = 0;
        while (true)
        {
            int pos = IndexOf(data, oldBytes, idx);
            if (pos < 0) break;
            Buffer.BlockCopy(newBytes, 0, data, pos, Math.Min(newBytes.Length, data.Length - pos));
            idx = pos + newBytes.Length;
            count++;
        }
        return count;
    }

    private static int IndexOf(byte[] data, byte[] pattern, int start = 0)
    {
        for (int i = start; i <= data.Length - pattern.Length; i++)
        {
            bool match = true;
            for (int j = 0; j < pattern.Length; j++)
            {
                if (data[i + j] != pattern[j]) { match = false; break; }
            }
            if (match) return i;
        }
        return -1;
    }

    private static HashSet<byte> BuildAllowedSet()
    {
        var set = new HashSet<byte>();
        const string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-/\\";
        foreach (var c in chars) set.Add((byte)c);
        return set;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Utility
    // ─────────────────────────────────────────────────────────────────────────

    private static string PadId(string id) => id.TrimStart('0').PadLeft(4, '0');

    private static string ReplaceTokensCi(string input, string oldPrimary, string newPrimary,
                                          string oldAlt, string newAlt,
                                          string oldSlotFlag = "", string newSlotFlag = "")
    {
        var result = Regex.Replace(input, Regex.Escape(oldPrimary), newPrimary, RegexOptions.IgnoreCase);
        result     = Regex.Replace(result, Regex.Escape(oldAlt),     newAlt,     RegexOptions.IgnoreCase);
        if (!string.IsNullOrEmpty(oldSlotFlag) && !string.IsNullOrEmpty(newSlotFlag) &&
            !string.Equals(oldSlotFlag, newSlotFlag, StringComparison.OrdinalIgnoreCase))
            result = Regex.Replace(result, Regex.Escape(oldSlotFlag), newSlotFlag, RegexOptions.IgnoreCase);
        return result;
    }

    private static string ReplaceInString(string input, string oldVal, string newVal)
        => Regex.Replace(input, Regex.Escape(oldVal), newVal, RegexOptions.IgnoreCase);

    /// <summary>
    /// Return true if the path is compatible with the target slot.
    /// Paths without ANY slot suffix always pass (e.g. shared textures).
    /// </summary>
    private static bool PathSlotOk(string path, string slotFlag, bool ignoreSlot)
    {
        if (ignoreSlot) return true;
        var pl = path.ToLower().Replace('\\', '/');
        bool hasSuffix = AllSlotSuffixes.Any(s => pl.Contains(s));
        if (!hasSuffix) return true;          // no slot suffix → always OK
        return pl.Contains(slotFlag);          // must match target slot
    }

    // ─────────────────────────────────────────────────────────────────────────
    // JSON path parser
    // ─────────────────────────────────────────────────────────────────────────

    private enum PathSegKind { Property, Index, Key }

    private record PathSeg(PathSegKind Kind, string Name, int Index);

    private static List<PathSeg> ParseJsonPath(string path)
    {
        var segs = new List<PathSeg>();
        int i    = 0;
        while (i < path.Length)
        {
            if (path[i] == '.')
            {
                i++;
                int start = i;
                while (i < path.Length && path[i] != '.' && path[i] != '[') i++;
                if (i > start)
                    segs.Add(new PathSeg(PathSegKind.Property, path[start..i], 0));
            }
            else if (path[i] == '[')
            {
                i++;
                int start = i;
                while (i < path.Length && path[i] != ']') i++;
                var inner = path[start..i];
                i++; // skip ']'
                if (int.TryParse(inner, out int idx))
                    segs.Add(new PathSeg(PathSegKind.Index, string.Empty, idx));
                else
                    segs.Add(new PathSeg(PathSegKind.Key, inner, 0));
            }
            else i++;
        }
        return segs;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Asset-chain & scope helpers  (mirrors Python preview_changes logic)
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Build a case-insensitive map of relative-path → absolute-path for every file
    /// under <paramref name="baseDir"/> — equivalent to Python's _index_files_ci.
    /// </summary>
    private static Dictionary<string, string> BuildFileIndex(string baseDir)
    {
        var idx = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var f in Directory.EnumerateFiles(baseDir, "*", SearchOption.AllDirectories))
        {
            var rel = Path.GetRelativePath(baseDir, f).Replace('\\', '/').ToLower();
            idx[rel] = f;
        }
        return idx;
    }

    /// <summary>
    /// Resolve a collection of relative paths to absolute paths using the CI file index.
    /// Falls back to an endswith match when an exact key is not found.
    /// Equivalent to Python's _resolve_many_ci.
    /// </summary>
    private static List<string> ResolveManyCI(
        Dictionary<string, string> index, IEnumerable<string> relPaths)
    {
        var resolved = new List<string>();
        var seen     = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var rp in relPaths)
        {
            var rpNorm = rp.Replace('\\', '/').TrimStart('.', '/', ' ').ToLower();

            if (index.TryGetValue(rpNorm, out var exact))
            {
                if (seen.Add(exact)) resolved.Add(exact);
                continue;
            }
            // Endswith fallback (mirrors Python's `key.endswith(rp)`)
            foreach (var (key, ap) in index)
            {
                if (key == rpNorm ||
                    key.EndsWith("/" + rpNorm, StringComparison.OrdinalIgnoreCase))
                {
                    if (seen.Add(ap)) resolved.Add(ap);
                }
            }
        }
        return resolved;
    }

    /// <summary>
    /// Like <see cref="ResolveManyCI"/> but also tries swapping new-token back to old-token
    /// so that paths already containing the new IDs (e.g. extracted from binary) still resolve.
    /// Equivalent to Python's _resolve_with_token_fallback.
    /// </summary>
    private static List<string> ResolveWithTokenFallback(
        Dictionary<string, string> index, IEnumerable<string> relPaths,
        string oldToken, string newToken, string oldAlt, string newAlt,
        string baseDir)
    {
        var resolved = new List<string>();
        var seen     = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var rp in relPaths)
        {
            var rpNorm = rp.Replace('\\', '/').TrimStart('.', '/', ' ');
            var candidates = new List<string> { rpNorm };
            var rpLow = rpNorm.ToLower();

            // If the path contains the new token, also try the old token (file not yet renamed)
            if (rpLow.Contains(newToken.ToLower()))
                candidates.Add(Regex.Replace(rpNorm, Regex.Escape(newToken), oldToken,
                                             RegexOptions.IgnoreCase));
            if (rpLow.Contains(newAlt.ToLower()))
                candidates.Add(Regex.Replace(rpNorm, Regex.Escape(newAlt), oldAlt,
                                             RegexOptions.IgnoreCase));

            foreach (var cand in candidates)
                foreach (var ap in ResolveManyCI(index, new[] { cand }))
                    if (seen.Add(ap)) resolved.Add(ap);
        }
        return resolved;
    }

    /// <summary>
    /// Find all .mdl files under <paramref name="baseDir"/> that match the item ID and slot.
    /// Equivalent to Python's _find_candidate_models / _find_models_by_id.
    /// </summary>
    private static List<string> FindCandidateMdls(
        string baseDir, string padOld, string slotKey, string slotFlag, bool ignoreSlot)
    {
        var result      = new List<string>();
        var ePat        = $"e{padOld}".ToLower();
        var aPat        = $"a{padOld}".ToLower();
        var slotFlagLow = slotFlag.ToLower();   // e.g. "_top"
        var slotKeyLow  = slotKey.ToLower();    // e.g. "top"

        foreach (var file in Directory.EnumerateFiles(baseDir, "*", SearchOption.AllDirectories))
        {
            var fnLow = Path.GetFileName(file).ToLower();
            var ext   = Path.GetExtension(fnLow);

            // Accept .mdl files and extensionless files inside a 'model' directory
            bool isMdl = ext == ".mdl" ||
                         (string.IsNullOrEmpty(ext) &&
                          file.Replace('\\', '/').ToLower().Contains("/model/"));
            if (!isMdl) continue;

            // Must contain the item ID (either prefix)
            if (!fnLow.Contains(ePat) && !fnLow.Contains(aPat)) continue;

            if (!ignoreSlot)
            {
                // Mirrors Python's _path_matches_slot:
                //   1. slot flag in filename  (e.g. "_top" in "c0201e0164_top.mdl")
                //   2. /slot_key/ as path segment (e.g. "/top/")
                //   3. basename-without-ext ends with slot key
                var pathLow    = file.Replace('\\', '/').ToLower();
                var baseNoExt  = ext.Length > 0 ? fnLow[..^ext.Length] : fnLow;
                bool slotMatch = fnLow.Contains(slotFlagLow)
                              || pathLow.Contains("/" + slotKeyLow + "/")
                              || baseNoExt.EndsWith(slotKeyLow);
                if (!slotMatch) continue;
            }

            result.Add(file);
        }
        return result;
    }

    /// <summary>
    /// Return true if <paramref name="data"/> contains either old-token byte sequence.
    /// </summary>
    private static bool ContainsOldToken(byte[] data, string oldToken, string oldAlt)
    {
        var p1 = Encoding.Latin1.GetBytes(oldToken.ToLower());
        var p2 = Encoding.Latin1.GetBytes(oldAlt.ToLower());
        return IndexOf(data, p1) >= 0 || IndexOf(data, p2) >= 0;
    }

    /// <summary>
    /// Extract ASCII path substrings from <paramref name="data"/> that contain the given
    /// file extension. Equivalent to Python's _extract_paths_from_bytes_generic.
    /// </summary>
    private static IEnumerable<string> ExtractAsciiPathsFromBytes(byte[] data, string suffix)
    {
        var suf     = Encoding.Latin1.GetBytes(suffix);
        var allowed = BuildAllowedSet();
        var results = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        int idx     = 0;

        while (true)
        {
            int pos = IndexOf(data, suf, idx);
            if (pos < 0) break;
            idx = pos + 1;

            int start = pos - 1;
            while (start >= 0 && allowed.Contains(data[start])) start--;
            start++;

            int end = pos + suf.Length;
            while (end < data.Length && allowed.Contains(data[end])) end++;

            var seg = Encoding.Latin1.GetString(data[start..end]).Replace('\\', '/');
            if (!string.IsNullOrEmpty(seg))
                results.Add(seg);
        }
        return results;
    }

    /// <summary>
    /// Recursively yield all VALUES from Files / FileSwaps dictionaries in a JSON tree.
    /// Used to discover locally referenced mod files.
    /// </summary>
    private static IEnumerable<string> ExtractJsonFilesValues(JsonNode? node)
    {
        if (node is JsonObject obj)
        {
            foreach (var kv in obj)
            {
                if ((kv.Key == "Files" || kv.Key == "FileSwaps") && kv.Value is JsonObject filesObj)
                {
                    foreach (var fkv in filesObj)
                        if (fkv.Value?.GetValue<string>() is { } v)
                            yield return v;
                }
                else
                {
                    foreach (var v in ExtractJsonFilesValues(kv.Value))
                        yield return v;
                }
            }
        }
        else if (node is JsonArray arr)
        {
            foreach (var item in arr)
                foreach (var v in ExtractJsonFilesValues(item))
                    yield return v;
        }
    }

    /// <summary>
    /// Build norm-relative scope sets from the planned rename list.
    /// Equivalent to Python's _build_json_scope(base_dir, rename_pairs=…).
    /// </summary>
    private static (HashSet<string> files, HashSet<string> dirs) BuildJsonScope(
        string baseDir, IEnumerable<PlannedRename> renames)
    {
        var files = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var dirs  = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var r in renames)
        {
            try
            {
                var rel = Path.GetRelativePath(baseDir, r.OldPath)
                              .Replace('\\', '/').TrimStart('.', '/').ToLower();
                if (r.IsDir) dirs.Add(rel);
                else         files.Add(rel);
            }
            catch { /* ignore paths outside baseDir */ }
        }
        return (files, dirs);
    }

    /// <summary>
    /// Return true if <paramref name="pathValue"/> is within the rename scope.
    /// Equivalent to Python's _path_in_scope.
    /// An empty scope (no files, no dirs) always returns true (no constraint).
    /// </summary>
    private static bool PathInJsonScope(
        string pathValue, HashSet<string> scopeFiles, HashSet<string> scopeDirs)
    {
        // Empty scope = no effective constraint
        if (scopeFiles.Count == 0 && scopeDirs.Count == 0) return true;

        var s = pathValue.Replace('\\', '/').TrimStart('.', '/').ToLower();

        if (scopeFiles.Contains(s)) return true;

        foreach (var d in scopeDirs)
            if (s == d || s.StartsWith(d + "/", StringComparison.OrdinalIgnoreCase))
                return true;

        return false;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // FileSystem helpers
    // ─────────────────────────────────────────────────────────────────────────

    private record FsEntry(string FullName, bool IsDirectory);

    private static IEnumerable<FsEntry> EnumerateAll(string root)
    {
        // Yield files first, then dirs (deepest first handled by caller ordering)
        var files = Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories);
        var dirs  = Directory.EnumerateDirectories(root, "*", SearchOption.AllDirectories);

        foreach (var f in files) yield return new FsEntry(f, false);
        foreach (var d in dirs)  yield return new FsEntry(d, true);
    }

    public static string RelativePath(string basePath, string fullPath)
    {
        try { return Path.GetRelativePath(basePath, fullPath).Replace('\\', '/'); }
        catch { return fullPath; }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Asset-chain new-mod helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Injects Penumbra IMC manipulations into <paramref name="manipulations"/> that
    /// redirect every game-variant of item <paramref name="newItemId"/> to use material
    /// folder <c>v{targetVariant:D4}</c>.  Vanilla entry data is read from the game's
    /// IMC file when available (to preserve attributes, sounds, VFX etc.); a minimal
    /// fallback entry with just MaterialId set is used when the file cannot be read.
    /// Duplicate entries (already present in <paramref name="seen"/>) are skipped.
    /// </summary>
    private void AddImcVariantRedirects(
        int             newItemId,
        EquipSlot       slot,
        int             targetVariant,
        JsonArray       manipulations,
        HashSet<string> seen,
        Action<string>  log)
    {
        if (_gameData == null) return;
        var slotLabel  = SlotInfo.LabelMap[slot];
        bool isAcc     = SlotInfo.IsAccessory(slot);
        string objType = isAcc ? "Accessory" : "Equipment";

        var entries = _gameData.GetImcVariantEntries((ushort)newItemId, slot);

        int count = 0;
        foreach (var e in entries)
        {
            // Build Penumbra Imc manipulation JSON with MaterialId overridden.
            var entryObj = new JsonObject
            {
                ["MaterialId"]          = targetVariant,
                ["DecalId"]             = (int)e.DecalId,
                ["AttributeMask"]       = (int)e.AttributeMask,
                ["SoundId"]             = (int)e.SoundId,
                ["VfxId"]               = (int)e.VfxId,
                ["MaterialAnimationId"] = (int)e.MaterialAnimationId,
            };
            var identObj = new JsonObject
            {
                ["Entry"]       = entryObj,
                ["PrimaryId"]   = newItemId,
                ["SecondaryId"] = 0,
                ["Variant"]     = e.Variant,
                ["ObjectType"]  = objType,
                ["EquipSlot"]   = slotLabel,
            };
            var manip = new JsonObject
            {
                ["Type"]         = "Imc",
                ["Manipulation"] = identObj,
            };
            var key = manip.ToJsonString();
            if (seen.Add(key))
            {
                manipulations.Add(manip);
                count++;
            }
        }

        if (count > 0)
            log($"Injected {count} IMC variant-redirect manipulation(s) for item {newItemId} " +
                $"→ material v{targetVariant:D4}.");
        else if (entries.Count == 0)
            log($"[INFO] No IMC data found for item {newItemId} slot {slotLabel}; skipping IMC injections.");
    }

    /// <summary>
    /// Walks all "Files" dicts in <paramref name="node"/> and, for every entry whose
    /// game-path key contains <paramref name="newTokenLow"/>, adds the local-path value
    /// to <paramref name="included"/> and copies the corresponding physical source file
    /// to <paramref name="destBase"/> (preserving relative path) if not already present.
    /// This handles textures/masks stored under a sibling item ID in the source mod
    /// that are only reachable through game-path redirects, not via binary tracing.
    /// </summary>
    /// <summary>
    /// Copies local files referenced by game-path keys that contain any of the sibling
    /// item tokens (e.g. "e9068") into the new mod directory, adding them to
    /// <paramref name="included"/> so that <see cref="FilterGroupJson"/> and
    /// <see cref="ExtractModEntriesRecursive"/> can later find them.
    /// </summary>
    private static void CollectLocalFilesForSiblingTokens(
        JsonNode?       node,
        IReadOnlySet<string> siblingTokensLow,
        string          sourceBase,
        string          destBase,
        HashSet<string> included,
        Dictionary<string, PlannedBinaryPatch> bpByPath,
        ref int         fileCount,
        int             targetVariant = 1)
    {
        switch (node)
        {
            case JsonObject obj:
            {
                if (obj["Files"] is JsonObject files)
                {
                    foreach (var kv in files)
                    {
                        var kLow = kv.Key.ToLower();
                        // Only process keys that are item-specific AND belong to a sibling ID.
                        if (!ItemSegmentRx.IsMatch(kLow)) continue;
                        if (!siblingTokensLow.Any(t => kLow.Contains(t))) continue;
                        if (kv.Value is null) continue;
                        string localVal;
                        try   { localVal = kv.Value.GetValue<string>(); }
                        catch { continue; }
                        var rawNorm = localVal.Replace('\\', '/').TrimStart('/');
                        var norm    = NormalizeMaterialVariant(rawNorm, targetVariant);
                        if (!included.Add(norm)) continue; // already present

                        var srcFile  = Path.Combine(sourceBase, rawNorm.Replace('/', Path.DirectorySeparatorChar));
                        if (!File.Exists(srcFile)) continue;
                        var destFile = Path.Combine(destBase, norm.Replace('/', Path.DirectorySeparatorChar));
                        if (File.Exists(destFile)) continue;
                        Directory.CreateDirectory(Path.GetDirectoryName(destFile)!);
                        var bytes = File.ReadAllBytes(srcFile);
                        if (bpByPath.TryGetValue(srcFile, out var bp))
                            bytes = ApplyPatchesToBytes(bytes, bp);
                        bytes = NormalizeBinaryMaterialVariant(bytes, targetVariant);
                        File.WriteAllBytes(destFile, bytes);
                        fileCount++;
                    }
                }
                foreach (var kv in obj)
                {
                    if (kv.Key == "Files") continue;
                    CollectLocalFilesForSiblingTokens(kv.Value, siblingTokensLow, sourceBase, destBase,
                                                     included, bpByPath, ref fileCount, targetVariant);
                }
                break;
            }
            case JsonArray arr:
            {
                foreach (var item in arr)
                    CollectLocalFilesForSiblingTokens(item, siblingTokensLow, sourceBase, destBase,
                                                     included, bpByPath, ref fileCount, targetVariant);
                break;
            }
        }
    }

    private static void CollectLocalFilesForNewToken(
        JsonNode?       node,
        string          newTokenLow,
        string          sourceBase,
        string          destBase,
        HashSet<string> included,
        Dictionary<string, PlannedBinaryPatch> bpByPath,
        ref int         fileCount,
        int             targetVariant = 1)
    {
        switch (node)
        {
            case JsonObject obj:
            {
                if (obj["Files"] is JsonObject files)
                {
                    foreach (var kv in files)
                    {
                        if (!kv.Key.ToLower().Contains(newTokenLow)) continue;
                        if (kv.Value is null) continue;
                        string localVal;
                        try   { localVal = kv.Value.GetValue<string>(); }
                        catch { continue; }
                        var rawNorm = localVal.Replace('\\', '/').TrimStart('/');
                        var norm    = NormalizeMaterialVariant(rawNorm, targetVariant);
                        if (!included.Add(norm)) continue; // already present (normalised)

                        // Source file still lives at the un-normalised path.
                        var srcFile  = Path.Combine(sourceBase, rawNorm.Replace('/', Path.DirectorySeparatorChar));
                        if (!File.Exists(srcFile)) continue;
                        // Destination uses the normalised variant folder.
                        var destFile = Path.Combine(destBase, norm.Replace('/', Path.DirectorySeparatorChar));
                        if (File.Exists(destFile)) continue;
                        Directory.CreateDirectory(Path.GetDirectoryName(destFile)!);
                        var bytes = File.ReadAllBytes(srcFile);
                        if (bpByPath.TryGetValue(srcFile, out var bp))
                            bytes = ApplyPatchesToBytes(bytes, bp);
                        bytes = NormalizeBinaryMaterialVariant(bytes, targetVariant);
                        File.WriteAllBytes(destFile, bytes);
                        fileCount++;
                    }
                }
                foreach (var kv in obj)
                {
                    if (kv.Key == "Files") continue;
                    CollectLocalFilesForNewToken(kv.Value, newTokenLow, sourceBase, destBase,
                                                 included, bpByPath, ref fileCount, targetVariant);
                }
                break;
            }
            case JsonArray arr:
            {
                foreach (var item in arr)
                    CollectLocalFilesForNewToken(item, newTokenLow, sourceBase, destBase,
                                                 included, bpByPath, ref fileCount, targetVariant);
                break;
            }
        }
    }

    /// <summary>
    /// Parses a Penumbra group filename (e.g. <c>group_003_bodysuit toggles.json</c>)
    /// into its numeric prefix and name suffix.
    /// Returns (999, filename-without-extension) for non-matching names.
    /// </summary>
    private static (int num, string suffix) ParseGroupFilename(string fname)
    {
        var m = Regex.Match(
            fname, @"^group_(\d+)_(.+)\.json$",
            RegexOptions.IgnoreCase);
        if (m.Success && int.TryParse(m.Groups[1].Value, out var n))
            return (n, m.Groups[2].Value);
        return (999, Path.GetFileNameWithoutExtension(fname));
    }

    /// <summary>
    /// Handles an IMC-type Penumbra group file during new-mod creation.
    /// If <c>Identifier.PrimaryId</c> is in <paramref name="referencedItemIds"/> and
    /// <c>Identifier.EquipSlot</c> matches <paramref name="sourceSlotLabel"/>, returns
    /// a deep-cloned copy with the PrimaryId updated to <paramref name="newInt"/> and
    /// (for cross-slot conversions) the EquipSlot updated to <paramref name="targetSlotLabel"/>.
    /// Returns <c>null</c> if the group does not belong to this conversion.
    /// </summary>
    private static JsonObject? TransformImcGroupJson(
        JsonObject?     groupObj,
        int             oldInt,
        int             newInt,
        string          sourceSlotLabel,
        string          targetSlotLabel)
    {
        if (groupObj == null) return null;
        if (groupObj["Identifier"] is not JsonObject ident) return null;

        // Read PrimaryId (Penumbra stores it as int or quoted string).
        var pidNode = ident["PrimaryId"];
        int primaryId;
        if (pidNode?.GetValueKind() == JsonValueKind.Number)
            primaryId = pidNode.GetValue<int>();
        else if (pidNode?.GetValueKind() == JsonValueKind.String &&
                 int.TryParse(pidNode.GetValue<string>(), out var si))
            primaryId = si;
        else return null;

        // Only convert IMC groups that belong to the actual source item.
        // Sibling-item IMC groups (e.g. e9068 attribute toggles while converting
        // e9069→e0387) must not be carried over — they control a different item's
        // mesh-visibility bits and are meaningless for the target item.
        if (primaryId != oldInt) return null;

        // Match on EquipSlot when a source slot label is known.
        if (!string.IsNullOrEmpty(sourceSlotLabel))
        {
            var equipSlot = ident["EquipSlot"]?.GetValue<string>() ?? string.Empty;
            if (!string.IsNullOrEmpty(equipSlot) &&
                !string.Equals(equipSlot, sourceSlotLabel, StringComparison.OrdinalIgnoreCase))
                return null;
        }

        // Deep-clone so we don't mutate the parsed source node.
        var result = (JsonObject)JsonNode.Parse(groupObj.ToJsonString())!;
        var resultIdent = (JsonObject)result["Identifier"]!;

        // Update PrimaryId to the new item.
        if (pidNode?.GetValueKind() == JsonValueKind.String)
            resultIdent["PrimaryId"] = newInt.ToString();
        else
            resultIdent["PrimaryId"] = newInt;

        // Update EquipSlot for cross-slot conversions.
        if (!string.IsNullOrEmpty(targetSlotLabel) &&
            !string.Equals(sourceSlotLabel, targetSlotLabel, StringComparison.OrdinalIgnoreCase))
            resultIdent["EquipSlot"] = targetSlotLabel;

        return result;
    }

    /// <summary>
    /// Rebuilds a Penumbra option-group JSON node, filtering each option down to only
    /// the Files/FileSwaps/Manipulations entries that belong to the converted item.
    /// All other option metadata (Name, Description, Priority, Type, …) is preserved.
    /// Returns <c>null</c> if no option in the group has any relevant content after
    /// filtering (so the caller can skip writing the file to the new mod).
    /// </summary>
    private static JsonObject? FilterGroupJson(
        JsonNode        node,
        HashSet<string> includedLocalPaths,
        string          newTokenLow,
        int             newInt,
        string          slotLabel,
        int             targetVariant = 1,
        IReadOnlySet<string>? siblingTokensLow = null)
    {
        if (node is not JsonObject groupObj) return null;
        if (groupObj["Options"] is not JsonArray srcOptions) return null;

        bool anyContent      = false;
        var  filteredOptions = new JsonArray();

        foreach (var opt in srcOptions)
        {
            if (opt is not JsonObject optObj)
            {
                // Preserve unexpected non-object entries as-is.
                filteredOptions.Add(opt?.DeepClone());
                continue;
            }

            // ── Filter Files ──────────────────────────────────────────────────────
            var filtFiles = new JsonObject();
            if (optObj["Files"] is JsonObject files)
            {
                foreach (var kv in files)
                {
                    if (kv.Value is null) continue;
                    // Allow keys for the new item, for sibling items referenced in the
                    // asset chain, or for non-item-specific paths.  Purely foreign-item
                    // keys (not in the asset chain) are excluded.
                    if (!IsGamePathKeyForNewItem(kv.Key, newTokenLow, siblingTokensLow)) continue;
                    string localVal;
                    try   { localVal = kv.Value.GetValue<string>(); }
                    catch { continue; }
                    var norm    = NormalizeMaterialVariant(
                                      localVal.Replace('\\', '/').TrimStart('/'), targetVariant);
                    var normKey = NormalizeMaterialVariant(kv.Key, targetVariant);
                    if (includedLocalPaths.Contains(norm) &&
                        !filtFiles.TryGetPropertyValue(normKey, out _))
                        filtFiles[normKey] = JsonValue.Create(norm);
                }
            }

            // ── Filter FileSwaps ──────────────────────────────────────────────────
            var filtSwaps = new JsonObject();
            if (optObj["FileSwaps"] is JsonObject swaps)
            {
                foreach (var kv in swaps)
                {
                    if (!kv.Key.ToLower().Contains(newTokenLow)) continue;
                    filtSwaps[kv.Key] = kv.Value is null
                        ? null
                        : JsonNode.Parse(kv.Value.ToJsonString());
                }
            }

            // ── Filter Manipulations ──────────────────────────────────────────────
            var filtManips = new JsonArray();
            var seenManips = new HashSet<string>();
            if (optObj["Manipulations"] is JsonArray manips)
            {
                foreach (var m in manips)
                {
                    if (!IsManipulationForItem(m, newInt, slotLabel)) continue;
                    var key = m!.ToJsonString();
                    if (seenManips.Add(key))
                        filtManips.Add(JsonNode.Parse(key));
                }
            }

            if (filtFiles.Count > 0 || filtSwaps.Count > 0 || filtManips.Count > 0)
                anyContent = true;

            // Rebuild the option preserving its metadata fields.
            var newOpt = new JsonObject();
            foreach (var kv in optObj)
            {
                if (kv.Key is "Files" or "FileSwaps" or "Manipulations") continue;
                newOpt[kv.Key] = kv.Value?.DeepClone();
            }
            newOpt["Files"]         = filtFiles;
            newOpt["FileSwaps"]     = filtSwaps;
            newOpt["Manipulations"] = filtManips;
            filteredOptions.Add(newOpt);
        }

        if (!anyContent) return null;

        // Rebuild the group node preserving all metadata fields (Name, Description, …).
        var result = new JsonObject();
        foreach (var kv in groupObj)
        {
            if (kv.Key == "Options") continue;
            result[kv.Key] = kv.Value?.DeepClone();
        }
        result["Options"] = filteredOptions;
        return result;
    }

    /// <summary>
    /// Recursively walks <paramref name="node"/> (which may be a flat
    /// <c>default_mod.json</c> object or a nested option-group file) and collects:
    /// <list type="bullet">
    ///   <item><c>Files</c> entries whose local-path value is in <paramref name="includedLocalPaths"/>.</item>
    ///   <item><c>FileSwaps</c> entries whose game-path key contains the new item token.</item>
    ///   <item><c>Manipulations</c> entries whose SetId/PrimaryId matches <paramref name="newInt"/>
    ///     and whose slot (if present) matches <paramref name="slotLabel"/>.</item>
    /// </list>
    /// All matches are merged into the three output collections (duplicates skipped).
    /// </summary>
    private static void ExtractModEntriesRecursive(
        JsonNode?       node,
        HashSet<string> includedLocalPaths,
        string          newTokenLow,
        int             newInt,
        string          slotLabel,
        JsonObject      mergedFiles,
        JsonObject      mergedFileSwaps,
        JsonArray       mergedManipulations,
        HashSet<string> seenManipulations,
        int             targetVariant = 1,
        IReadOnlySet<string>? siblingTokensLow = null)
    {
        switch (node)
        {
            case JsonObject obj:
            {
                if (obj["Files"] is JsonObject files)
                {
                    foreach (var kv in files)
                    {
                        if (kv.Value is null) continue;
                        // Keep entries for the new item, for sibling items in the asset
                        // chain, or for non-item-specific paths.  Entries for foreign
                        // items not part of the conversion are excluded.
                        if (!IsGamePathKeyForNewItem(kv.Key, newTokenLow, siblingTokensLow)) continue;
                        string localVal;
                        try { localVal = kv.Value.GetValue<string>(); }
                        catch { continue; }
                        var norm    = NormalizeMaterialVariant(
                                          localVal.Replace('\\', '/').TrimStart('/'), targetVariant);
                        var normKey = NormalizeMaterialVariant(kv.Key, targetVariant);
                        if (includedLocalPaths.Contains(norm) &&
                            !mergedFiles.TryGetPropertyValue(normKey, out _))
                            mergedFiles[normKey] = JsonValue.Create(norm);
                    }
                }

                if (obj["FileSwaps"] is JsonObject swaps)
                {
                    foreach (var kv in swaps)
                    {
                        if (!kv.Key.ToLower().Contains(newTokenLow)) continue;
                        if (!mergedFileSwaps.TryGetPropertyValue(kv.Key, out _))
                            mergedFileSwaps[kv.Key] = kv.Value is null
                                ? null
                                : JsonNode.Parse(kv.Value.ToJsonString());
                    }
                }

                if (obj["Manipulations"] is JsonArray manips)
                {
                    foreach (var m in manips)
                    {
                        if (!IsManipulationForItem(m, newInt, slotLabel)) continue;
                        var key = m!.ToJsonString();
                        if (seenManipulations.Add(key))
                            mergedManipulations.Add(JsonNode.Parse(key));
                    }
                }

                // Recurse into all other children (handles nested Options arrays, groups, etc.)
                foreach (var kv in obj)
                {
                    if (kv.Key is "Files" or "FileSwaps" or "Manipulations") continue;
                    ExtractModEntriesRecursive(kv.Value, includedLocalPaths, newTokenLow,
                        newInt, slotLabel, mergedFiles, mergedFileSwaps,
                        mergedManipulations, seenManipulations, targetVariant, siblingTokensLow);
                }
                break;
            }
            case JsonArray arr:
            {
                foreach (var item in arr)
                    ExtractModEntriesRecursive(item, includedLocalPaths, newTokenLow,
                        newInt, slotLabel, mergedFiles, mergedFileSwaps,
                        mergedManipulations, seenManipulations, targetVariant, siblingTokensLow);
                break;
            }
        }
    }

    /// <summary>
    /// Returns <c>true</c> when the Files-dict game-path <paramref name="key"/> is either:
    /// (a) for the new item (contains <paramref name="newTokenLow"/>),
    /// (b) for a sibling item whose ID is in <paramref name="siblingTokensLow"/>, or
    /// (c) not item-specific (no <c>[/\][ea]\d{4}[/\]</c> segment).
    /// </summary>
    private static readonly Regex ItemSegmentRx = new(
        @"[/\\][ea]\d{4}[/\\]",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    private static bool IsGamePathKeyForNewItem(
        string key, string newTokenLow,
        IReadOnlySet<string>? siblingTokensLow = null)
    {
        var kLow = key.ToLower();
        if (kLow.Contains(newTokenLow)) return true;          // ✔ it is the new item
        if (siblingTokensLow != null && ItemSegmentRx.IsMatch(kLow))
        {
            // Allow if the item segment belongs to a known sibling in the asset chain.
            foreach (var sib in siblingTokensLow)
                if (kLow.Contains(sib)) return true;
        }
        return !ItemSegmentRx.IsMatch(kLow);                  // ✔ not item-specific
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Material-variant path normalisation helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Rewrites every <c>material/v{NNNN}/</c> folder segment in <paramref name="path"/>
    /// to <c>material/v{targetVariant:D4}/</c>.
    /// Only the segment immediately after a <c>material</c> folder is matched, so
    /// unrelated numeric sub-strings are not touched.
    /// </summary>
    private static readonly Regex MaterialVariantSegmentRx = new(
        @"(?<=[/\\]material[/\\])v\d{4}(?=[/\\])",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    private static string NormalizeMaterialVariant(string path, int targetVariant)
    {
        if (targetVariant <= 0) return path;
        return MaterialVariantSegmentRx.Replace(path, $"v{targetVariant:D4}");
    }

    /// <summary>
    /// Patches <c>/material/v{N:D4}/</c> byte sequences inside binary file content
    /// (primarily <c>.mdl</c> files, which embed ASCII material-path strings) to use
    /// <paramref name="targetVariant"/>.  Variants 1–9 are checked; same-variant
    /// occurrences are skipped.  The replacement is in-place safe because all
    /// variant folder names have the same length (<c>v0001</c> … <c>v0009</c>).
    /// </summary>
    private static byte[] NormalizeBinaryMaterialVariant(byte[] bytes, int targetVariant)
    {
        if (targetVariant <= 0) return bytes;
        var target = Encoding.Latin1.GetBytes($"/material/v{targetVariant:D4}/");
        for (int v = 1; v <= 9; v++)
        {
            if (v == targetVariant) continue;
            var search = Encoding.Latin1.GetBytes($"/material/v{v:D4}/");
            if (search.Length == target.Length)
                ReplaceBytesInArray(ref bytes, search, target);
        }
        return bytes;
    }

    /// <summary>
    /// Returns true when <paramref name="m"/> is a Manipulations array element
    /// whose SetId or PrimaryId equals <paramref name="targetId"/> and whose
    /// Slot/EquipSlot field (if present) matches <paramref name="slotLabel"/>.
    /// </summary>
    private static bool IsManipulationForItem(JsonNode? m, int targetId, string slotLabel)
    {
        if (m is not JsonObject mo) return false;
        if (mo["Manipulation"] is not JsonObject manipObj) return false;

        int? foundId = null;
        foreach (var idKey in new[] { "SetId", "PrimaryId" })
        {
            var idNode = manipObj[idKey];
            if (idNode is null) continue;
            int v;
            bool ok;
            try { v = idNode.GetValue<int>(); ok = true; }
            catch { v = 0; ok = false; }
            if (!ok)
                try { ok = int.TryParse(idNode.GetValue<string>(), out v); }
                catch { ok = false; }
            if (ok) { foundId = v; break; }
        }
        if (foundId == null || foundId != targetId) return false;

        // Check slot if present
        string slot = string.Empty;
        try { slot = manipObj["Slot"]?.GetValue<string>() ?? string.Empty; } catch { }
        if (string.IsNullOrEmpty(slot))
            try { slot = manipObj["EquipSlot"]?.GetValue<string>() ?? string.Empty; } catch { }

        return string.IsNullOrEmpty(slot) ||
               string.Equals(slot, slotLabel, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Applies selected binary string patches to a byte array and returns the result.</summary>
    private static byte[] ApplyPatchesToBytes(byte[] bytes, PlannedBinaryPatch bp)
    {
        foreach (var patch in bp.Patches.Where(p => p.Selected))
        {
            var oldBytes = Encoding.Latin1.GetBytes(patch.OldString);
            var newBytes = Encoding.Latin1.GetBytes(patch.NewString);
            if (oldBytes.Length != newBytes.Length)
            {
                var maxLen = Math.Max(oldBytes.Length, newBytes.Length);
                Array.Resize(ref oldBytes, maxLen);
                Array.Resize(ref newBytes, maxLen);
            }
            ReplaceBytesInArray(ref bytes, oldBytes, newBytes);
        }
        return bytes;
    }

    /// <summary>
    /// Returns a sort key for a PlannedRename's old path when ordering .mtrl copies.
    /// The rename whose material-variant folder matches <paramref name="sourceVariant"/>
    /// gets key 0 (highest priority); other variants are ordered by descending variant
    /// number so that later variations are preferred over earlier ones.
    /// Non-material paths always return <see cref="int.MaxValue"/> (lowest priority,
    /// interleaved arbitrarily — ordering only matters for material dedup).
    /// </summary>
    private static int MtrlVariantSortKey(string oldPath, int sourceVariant)
    {
        var norm = oldPath.Replace('\\', '/').ToLower();
        var m    = Regex.Match(norm, @"/material/v(\d{4})/");
        if (!m.Success) return int.MaxValue;
        if (!int.TryParse(m.Groups[1].Value, out var v)) return int.MaxValue;
        if (sourceVariant > 0 && v == sourceVariant) return 0; // exact match → first
        return 10_000 - v;                                      // higher variant → earlier
    }

    /// <summary>Writes a minimal Penumbra <c>meta.json</c> with the given display name.</summary>
    private static void WriteDefaultMetaJson(string path, string name, JsonSerializerOptions opts)
    {
        var meta = new JsonObject
        {
            ["FileVersion"] = 3,
            ["Name"]        = name,
            ["Author"]      = "",
            ["Description"] = "",
            ["Version"]     = "",
            ["Website"]     = "",
            ["ModTags"]     = new JsonArray(),
        };
        File.WriteAllText(path, meta.ToJsonString(opts), Encoding.UTF8);
    }

    /// <summary>
    /// Returns a version of <paramref name="name"/> safe for use as a directory name
    /// by replacing any characters forbidden in Windows file names with underscores.
    /// </summary>
    public static string SanitizeFolderName(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var sb = new StringBuilder(name.Length);
        foreach (var c in name)
            sb.Append(Array.IndexOf(invalid, c) >= 0 ? '_' : c);
        return sb.ToString().Trim(' ', '.').TrimEnd();
    }
}
