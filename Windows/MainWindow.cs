using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using AdvancedPenumbraItemConverter.Models;
using AdvancedPenumbraItemConverter.Services;
using Dalamud.Bindings.ImGui;
using Dalamud.Interface.Windowing;

namespace AdvancedPenumbraItemConverter.Windows;

/// <summary>
/// Primary plugin window.  Three-tab layout:
///   Setup   – mod/slot/ID inputs + Penumbra mod picker
///   Preview – expandable lists for renames, JSON edits, binary patches
///   Log     – scrolling operation log
/// </summary>
public sealed class MainWindow : Window, IDisposable
{
    // ─────────────────────────────────────────────────────────────────────────
    // State
    // ─────────────────────────────────────────────────────────────────────────

    private readonly Plugin              _plugin;
    private          ConversionTask      _task = new();

    // Setup-tab inputs
    private string   _modDirInput  = string.Empty;

    // Source item detection
    private List<DetectedItem>  _detectedItems  = new();
    private int                 _sourceItemIdx  = -1;

    // Target item search
    private string           _targetSearch    = string.Empty;
    private List<GameItem>   _targetResults   = new();   // filtered view
    private List<GameItem>   _allSlotItems    = new();   // full sorted list for current slot
    private int              _targetItemIdx   = -1;

    // Accessory cross-slot output selection
    private EquipSlot _accessoryOutputSlot = EquipSlot.Earring;

    private static readonly EquipSlot[] AccessorySlots =
    {
        EquipSlot.Earring,
        EquipSlot.Neck,
        EquipSlot.Wrists,
        EquipSlot.RingRight,
        EquipSlot.RingLeft,
    };

    private static readonly string[] AccessorySlotLabels =
        AccessorySlots.Select(s => SlotInfo.LabelMap[s]).ToArray();

    // Penumbra mod browser
    private Dictionary<string, string>? _penumbraMods;   // dir → name
    private string[]?  _modNames;
    private string[]?  _modDirs;
    private int        _modPickerIndex = -1;
    private string     _modSearch      = string.Empty;
    private bool       _penumbraAvailable = false;

    // Preview state
    private bool  _previewDirty = true;

    // JSON tab expand state
    private readonly HashSet<int> _jsonExpanded = new();

    // Log
    private readonly StringBuilder _logBuffer  = new();
    private          bool          _logScrollToBottom = false;
    private          int           _lastApplyLogOffset = -1;

    // Post-conversion notification shown in the Setup tab
    private string? _postConversionNotice = null;

    // Output mode
    private bool   _createNewMod           = false;
    private string _newModName             = string.Empty;
    private bool   _newModNameIsDefault    = true;  // false once the user manually edits the field

    // Colours
    private static readonly Vector4 ColAdd   = new(0.41f, 0.86f, 0.48f, 1f);
    private static readonly Vector4 ColDel   = new(0.94f, 0.36f, 0.42f, 1f);
    private static readonly Vector4 ColWarn  = new(1.00f, 0.82f, 0.42f, 1f);
    private static readonly Vector4 ColMuted = new(0.50f, 0.50f, 0.63f, 1f);
    private static readonly Vector4 ColAccent= new(0.49f, 0.30f, 1.00f, 1f);

    // ─────────────────────────────────────────────────────────────────────────
    // Ctor / Dispose
    // ─────────────────────────────────────────────────────────────────────────

    public MainWindow(Plugin plugin) : base(
        "Advanced Penumbra Item Converter###APICMain",
        ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoScrollWithMouse)
    {
        _plugin = plugin;
        Size          = new Vector2(780, 600);
        SizeCondition = ImGuiCond.FirstUseEver;

        // Restore last-used values from config
        var cfg        = plugin.Configuration;
        _modDirInput   = cfg.LastModDirectory;
        _createNewMod  = cfg.CreateNewMod;
        _newModName    = cfg.LastNewModName;
        _newModNameIsDefault = string.IsNullOrEmpty(_newModName);
    }

    public void Dispose() { }

    // Called by Plugin when Penumbra availability changes
    public void OnPenumbraStateChanged()
    {
        _penumbraAvailable = _plugin.PenumbraIpc.IsAvailable;
        if (_penumbraAvailable)
            RefreshModList();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Draw
    // ─────────────────────────────────────────────────────────────────────────

    public override void Draw()
    {
        _penumbraAvailable = _plugin.PenumbraIpc.IsAvailable;

        // Status bar
        DrawStatusBar();
        ImGui.Spacing();

        if (ImGui.BeginTabBar("##APICTabs"))
        {
            if (ImGui.BeginTabItem("Setup"))
            {
                DrawSetupTab();
                ImGui.EndTabItem();
            }

            bool previewDisabled = !_task.IsPlanned;
            if (previewDisabled) ImGui.BeginDisabled();
            if (ImGui.BeginTabItem("Preview"))
            {
                DrawPreviewTab();
                ImGui.EndTabItem();
            }
            if (previewDisabled) ImGui.EndDisabled();

            if (ImGui.BeginTabItem("Log"))
            {
                DrawLogTab();
                ImGui.EndTabItem();
            }

            ImGui.EndTabBar();
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Status bar
    // ─────────────────────────────────────────────────────────────────────────

    private void DrawStatusBar()
    {
        if (_penumbraAvailable)
        {
            ImGui.TextColored(ColAdd, "● Penumbra: connected");
        }
        else
        {
            ImGui.TextColored(ColDel, "● Penumbra: not available");
            ImGui.SameLine();
            ImGui.TextColored(ColMuted, "(manual path mode)");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Setup tab
    // ─────────────────────────────────────────────────────────────────────────

    private void DrawSetupTab()
    {
        ImGui.PushItemWidth(-1);

        // ── Mod selection ─────────────────────────────────────────────────
        ImGui.Separator(); ImGui.Text("Mod Folder:");

        if (_penumbraAvailable && _modNames != null && _modNames.Length > 0)
        {
            ImGui.SetNextItemWidth(ImGui.GetContentRegionAvail().X - 100);
            ImGui.InputTextWithHint("##ModSearch", "Filter mods...", ref _modSearch, 128);
            ImGui.SameLine();
            if (ImGui.Button("Refresh##mod")) RefreshModList();

            // Filtered list
            var filtered = FilteredMods();
            if (filtered.Count > 0)
            {
                var displayNames = filtered.Select(i => _modNames![i]).ToArray();
                int localIdx = _modPickerIndex >= 0 && _modPickerIndex < _modDirs!.Length
                    ? filtered.IndexOf(_modPickerIndex) : 0;
                if (localIdx < 0) localIdx = 0;

                ImGui.SetNextItemWidth(-1);
                float listHeight = 5 * ImGui.GetTextLineHeightWithSpacing();
                if (ImGui.BeginListBox("##ModList", new Vector2(-1, listHeight)))
                {
                    for (int li = 0; li < displayNames.Length; li++)
                    {
                        bool sel = li == localIdx;
                        if (ImGui.Selectable(displayNames[li], sel))
                        {
                            localIdx        = li;
                            _modPickerIndex = filtered[localIdx];
                            _modDirInput    = _modDirs![_modPickerIndex];
                            MarkDirty();
                            ScanAndDetectItems();
                        }
                        if (sel) ImGui.SetItemDefaultFocus();
                    }
                    ImGui.EndListBox();
                }
            }
            else
            {
                ImGui.TextColored(ColMuted, "(no mods match filter)");
            }

            ImGui.Spacing();
            ImGui.TextColored(ColMuted, "Selected folder:");
        }
        else if (_penumbraAvailable)
        {
            if (ImGui.Button("Refresh Mod List")) RefreshModList();
            ImGui.SameLine();
        }

        // Manual path input (read-only when a mod list is available; editable for manual mode)
        ImGui.SetNextItemWidth(-1);
        var modDirFlags = (_penumbraAvailable && _modNames != null && _modNames.Length > 0)
            ? ImGuiInputTextFlags.ReadOnly
            : ImGuiInputTextFlags.None;
        if (ImGui.InputTextWithHint("##ModDir", "Path to mod folder (contains default_mod.json)...",
                ref _modDirInput, 512, modDirFlags))
            MarkDirty();

        if (_penumbraAvailable)
        {
            ImGui.SameLine(0, 4);
            if (ImGui.SmallButton("Dir##penmod"))
            {
                // Try to resolve via Penumbra if we have a selection
                if (_modPickerIndex >= 0 && _modDirs != null)
                {
                    var (ok, fullPath) = _plugin.PenumbraIpc.GetModPath(
                        Path.GetFileName(_modDirs[_modPickerIndex]));
                    if (ok && !string.IsNullOrEmpty(fullPath))
                        _modDirInput = fullPath;
                }
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Resolve full path via Penumbra");
        }

        ImGui.Spacing();

        // ── Source Item ───────────────────────────────────────────────────
        ImGui.Separator(); ImGui.Text("Source Item:");

        if (_detectedItems.Count == 0)
        {
            if (string.IsNullOrEmpty(_modDirInput))
            {
                ImGui.TextColored(ColMuted, "Select a mod folder above.");
            }
            else
            {
                ImGui.TextColored(ColMuted, "No items detected. ");
                ImGui.SameLine();
                if (ImGui.SmallButton("Scan Mod"))
                    ScanAndDetectItems();
            }
        }
        else
        {
            ImGui.TextColored(ColMuted, $"{_detectedItems.Count} item(s) found:");
            ImGui.SameLine();
            if (ImGui.SmallButton("Refresh##src"))
                ScanAndDetectItems();

            float srcH = Math.Min(_detectedItems.Count, 6) * ImGui.GetTextLineHeightWithSpacing() + 4;
            if (ImGui.BeginListBox("##SourceItems", new Vector2(-1, srcH)))
            {
                for (int i = 0; i < _detectedItems.Count; i++)
                {
                    var di  = _detectedItems[i];
                    var lbl = $"[{SlotInfo.LabelMap[di.Slot]}]  {di.ItemName}  (ID: {di.ModelIdDisplay})";
                    bool sel = (i == _sourceItemIdx);
                    if (ImGui.Selectable(lbl, sel))
                    {
                        if (_sourceItemIdx != i)
                        {
                            _sourceItemIdx = i;
                            _targetItemIdx = -1;
                            _targetResults.Clear();
                            _targetSearch  = string.Empty;
                            _accessoryOutputSlot = _detectedItems[i].Slot;
                            MarkDirty();
                            RefreshTargetSlotItems();
                        }
                    }
                    if (sel) ImGui.SetItemDefaultFocus();
                }
                ImGui.EndListBox();
            }
        }

        ImGui.Spacing();

        // ── Target Item ───────────────────────────────────────────────────
        ImGui.Separator(); ImGui.Text("Target Item:");

        if (_sourceItemIdx < 0 || _sourceItemIdx >= _detectedItems.Count)
        {
            ImGui.TextColored(ColMuted, "Select a source item first.");
        }
        else
        {
            var srcSlot = _detectedItems[_sourceItemIdx].Slot;
            var srcItem = _detectedItems[_sourceItemIdx];

            // When the source is an accessory, let the user pick which accessory slot to target.
            if (srcItem.IsAccessory)
            {
                ImGui.Text("Convert to slot:");
                ImGui.SameLine();
                ImGui.SetNextItemWidth(160);
                int outIdx = Array.IndexOf(AccessorySlots, _accessoryOutputSlot);
                if (outIdx < 0) outIdx = 0;
                if (ImGui.Combo("##AccessoryOutputSlot", ref outIdx, AccessorySlotLabels, AccessorySlotLabels.Length))
                {
                    var newOutSlot = AccessorySlots[outIdx];
                    if (newOutSlot != _accessoryOutputSlot)
                    {
                        _accessoryOutputSlot = newOutSlot;
                        _targetItemIdx = -1;
                        _targetResults.Clear();
                        _targetSearch  = string.Empty;
                        MarkDirty();
                        RefreshTargetSlotItems();
                    }
                }
                ImGui.Spacing();
            }

            // The effective slot for the target item list (may differ from source for accessories).
            var effectiveSlot = srcItem.IsAccessory ? _accessoryOutputSlot : srcSlot;

            // Filter bar
            ImGui.SetNextItemWidth(-1);
            if (ImGui.InputTextWithHint("##TargetSearch",
                    $"Filter {SlotInfo.LabelMap[effectiveSlot]} items\u2026",
                    ref _targetSearch, 256))
            {
                ApplyTargetFilter();
            }

            var showList = _targetResults.Count > 0 ? _targetResults : _allSlotItems;

            if (showList.Count > 0)
            {
                float lineH = ImGui.GetTextLineHeightWithSpacing();
                float listH = Math.Clamp(
                    ImGui.GetContentRegionAvail().Y - lineH * 4f,
                    lineH * 3f,
                    lineH * 8f);

                if (ImGui.BeginChild("##TargetItemsOuter", new Vector2(-1, listH), true))
                {
                    for (int i = 0; i < showList.Count; i++)
                    {
                        var gi     = showList[i];
                        bool sel   = (i == _targetItemIdx);
                        var  lbl   = $"{gi.Name}  (ID: {gi.ModelIdDisplay})##ti{i}";

                        if (ImGui.Selectable(lbl, sel, ImGuiSelectableFlags.None, new Vector2(0, 0)))
                        {
                            _targetItemIdx = i;
                            MarkDirty();
                        }
                        if (sel) ImGui.SetItemDefaultFocus();
                    }
                }
                ImGui.EndChild();
            }
            else
            {
                ImGui.TextColored(ColMuted, "(loading item list…)");
            }

        }

        ImGui.Spacing();

        // ── Output mode ───────────────────────────────────────────────────
        ImGui.Separator(); ImGui.Text("Output:");

        bool modeInPlace = !_createNewMod;
        if (ImGui.RadioButton("Modify in place##outmode", modeInPlace))
            _createNewMod = false;
        ImGui.SameLine();
        if (ImGui.RadioButton("Create new mod##outmode", !modeInPlace))
        {
            _createNewMod = true;
            if (_newModNameIsDefault || string.IsNullOrEmpty(_newModName))
                RefreshDefaultNewModName();
        }

        if (_createNewMod)
        {
            ImGui.Spacing();
            ImGui.Text("New mod name:");
            ImGui.SetNextItemWidth(-1);
            if (ImGui.InputText("##NewModName", ref _newModName, 256))
                _newModNameIsDefault = false;

            // Show the target path
            if (!string.IsNullOrEmpty(_modDirInput))
            {
                var nmParent = Path.GetDirectoryName(
                    _modDirInput.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
                if (!string.IsNullOrEmpty(nmParent) && !string.IsNullOrEmpty(_newModName))
                {
                    var safe = ModConverterService.SanitizeFolderName(_newModName);
                    ImGui.TextColored(ColMuted, "→ " + Path.Combine(nmParent, safe));
                }
            }
        }

        ImGui.Spacing();

        // ── Action buttons ────────────────────────────────────────────────
        float btnW = 140;

        if (_previewDirty) ImGui.PushStyleColor(ImGuiCol.Button, ColAccent);
        if (ImGui.Button(_previewDirty ? "Preview Changes *" : "Preview Changes", new Vector2(btnW, 0)))
            RunPreview();
        if (_previewDirty) ImGui.PopStyleColor();

        ImGui.SameLine();

        bool canApply = _task.IsPlanned && !_task.IsApplied;
        if (!canApply) ImGui.BeginDisabled();
        string applyLabel = _createNewMod ? "Create New Mod" : "Apply Selected";
        if (ImGui.Button(applyLabel, new Vector2(btnW, 0)))
            RunApply();
        if (!canApply) ImGui.EndDisabled();

        ImGui.PopItemWidth();

        if (!string.IsNullOrEmpty(_task.ErrorMessage))
        {
            ImGui.Spacing();
            ImGui.TextColored(ColDel, $"⚠ {_task.ErrorMessage}");
        }

        if (!string.IsNullOrEmpty(_postConversionNotice))
        {
            ImGui.Spacing();
            ImGui.TextColored(ColAdd, _postConversionNotice);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Preview tab
    // ─────────────────────────────────────────────────────────────────────────

    private void DrawPreviewTab()
    {
        if (!_task.IsPlanned)
        {
            ImGui.TextColored(ColMuted, "Run 'Preview Changes' on the Setup tab first.");
            return;
        }

        float totalH    = ImGui.GetContentRegionAvail().Y;
        float childH    = totalH - 4;

        ImGui.BeginChild("##PreviewScroll", new Vector2(0, childH), false, ImGuiWindowFlags.HorizontalScrollbar);

        // ── File Renames ──────────────────────────────────────────────────
        DrawCollapsibleSection($"File Renames  ({_task.PlannedRenames.Count})", () =>
        {
            if (_task.PlannedRenames.Count == 0)
            {
                ImGui.TextColored(ColMuted, "No files or directories to rename.");
                return;
            }

            DrawSelectAllButtons("renames");

            foreach (var rename in _task.PlannedRenames)
            {
                bool sel = rename.Selected;
                ImGui.Checkbox($"##{rename.OldPath}", ref sel);
                rename.Selected = sel;
                ImGui.SameLine();
                var label = rename.IsDir ? "[DIR] " : "[FILE]";
                ImGui.TextColored(ColMuted, label);
                ImGui.SameLine();
                var baseDir = _modDirInput;
                ImGui.TextColored(ColDel,   ModConverterService.RelativePath(baseDir, rename.OldPath));
                ImGui.SameLine();
                ImGui.TextColored(ColMuted, " → ");
                ImGui.SameLine();
                ImGui.TextColored(ColAdd,   ModConverterService.RelativePath(baseDir, rename.NewPath));
            }
        });

        // ── JSON / Metadata ───────────────────────────────────────────────
        int  totalJsonChanges = _task.PlannedJsonChanges.Sum(j => j.Changes.Count);
        DrawCollapsibleSection(
            $"Metadata  ({_task.PlannedJsonChanges.Count} files, {totalJsonChanges} changes)", () =>
        {
            if (_task.PlannedJsonChanges.Count == 0)
            {
                ImGui.TextColored(ColMuted, "No JSON metadata changes.");
                return;
            }

            DrawSelectAllButtons("jsons");

            for (int fi = 0; fi < _task.PlannedJsonChanges.Count; fi++)
            {
                var jc = _task.PlannedJsonChanges[fi];
                bool expanded = _jsonExpanded.Contains(fi);

                // File header row
                ImGui.BeginGroup();
                {
                    bool fileSel = jc.Selected;
                    ImGui.Checkbox($"##jf{fi}", ref fileSel);
                    jc.Selected = fileSel;
                    ImGui.SameLine();

                    string expBtn = expanded ? "▼ " : "▶ ";
                    ImGui.TextColored(ColAccent, expBtn);
                    ImGui.SameLine();

                    // Make the name clickable to toggle expand
                    var relName = ModConverterService.RelativePath(_modDirInput, jc.FilePath);
                    if (ImGui.Selectable($"{relName}  ({jc.Changes.Count} changes)##jfsel{fi}",
                            false, ImGuiSelectableFlags.None, new Vector2(0, 0)))
                    {
                        if (expanded) _jsonExpanded.Remove(fi);
                        else          _jsonExpanded.Add(fi);
                    }
                }
                ImGui.EndGroup();

                if (expanded)
                {
                    ImGui.Indent(24);
                    foreach (var change in jc.Changes)
                    {
                        bool chSel = change.Selected;
                        ImGui.Checkbox($"##jc{fi}{change.JsonPath}", ref chSel);
                        change.Selected = chSel;
                        ImGui.SameLine();

                        ImGui.TextColored(ColMuted, TypeLabel(change.ChangeType));
                        ImGui.SameLine();
                        ImGui.TextColored(ColDel,   Truncate(change.OldValue, 60));
                        ImGui.SameLine();
                        ImGui.TextColored(ColMuted, " → ");
                        ImGui.SameLine();
                        ImGui.TextColored(ColAdd,   Truncate(change.NewValue, 60));

                        if (ImGui.IsItemHovered())
                        {
                            ImGui.BeginTooltip();
                            ImGui.TextUnformatted($"Path:  {change.JsonPath}");
                            ImGui.TextUnformatted($"Old:   {change.OldValue}");
                            ImGui.TextUnformatted($"New:   {change.NewValue}");
                            ImGui.EndTooltip();
                        }
                    }
                    ImGui.Unindent(24);
                }
            }
        });

        // ── Binary Patches (.mdl / .mtrl) ─────────────────────────────────
        int totalPatches = _task.PlannedBinaryPatches.Sum(b => b.Patches.Count);
        DrawCollapsibleSection(
            $"Binary Patches  ({_task.PlannedBinaryPatches.Count} files, {totalPatches} patches)", () =>
        {
            if (_task.PlannedBinaryPatches.Count == 0)
            {
                ImGui.TextColored(ColMuted, "No binary string patches.");
                return;
            }

            DrawSelectAllButtons("binaries");

            foreach (var bp in _task.PlannedBinaryPatches)
            {
                bool fileSel = bp.Selected;
                ImGui.Checkbox($"##{bp.FilePath}bin", ref fileSel);
                bp.Selected = fileSel;
                ImGui.SameLine();
                ImGui.TextColored(ColAccent,
                    ModConverterService.RelativePath(_modDirInput, bp.FilePath));

                ImGui.Indent(24);
                foreach (var patch in bp.Patches)
                {
                    bool pSel = patch.Selected;
                    ImGui.Checkbox($"##{patch.OldString}", ref pSel);
                    patch.Selected = pSel;
                    ImGui.SameLine();
                    ImGui.TextColored(ColDel,   Truncate(patch.OldString, 60));
                    ImGui.SameLine();
                    ImGui.TextColored(ColMuted, " → ");
                    ImGui.SameLine();
                    ImGui.TextColored(ColAdd,   Truncate(patch.NewString, 60));
                }
                ImGui.Unindent(24);
            }
        });

        ImGui.EndChild();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Log tab
    // ─────────────────────────────────────────────────────────────────────────

    private void DrawLogTab()
    {
        if (ImGui.Button("Clear"))
        {
            _logBuffer.Clear();
            _lastApplyLogOffset = -1;
        }

        ImGui.SameLine();
        bool hasApplyLog = _lastApplyLogOffset >= 0 && _lastApplyLogOffset <= _logBuffer.Length;
        if (!hasApplyLog) ImGui.BeginDisabled();
        if (ImGui.Button("Copy Last Conversion"))
        {
            var text = _logBuffer.ToString(_lastApplyLogOffset, _logBuffer.Length - _lastApplyLogOffset);
            ImGui.SetClipboardText(text);
        }
        if (ImGui.IsItemHovered(hasApplyLog ? ImGuiHoveredFlags.None : ImGuiHoveredFlags.AllowWhenDisabled))
            ImGui.SetTooltip("Copy all log entries from the last conversion to the clipboard");
        if (!hasApplyLog) ImGui.EndDisabled();

        ImGui.BeginChild("##LogScroll", new Vector2(0, -1), false,
            ImGuiWindowFlags.HorizontalScrollbar);
        ImGui.TextUnformatted(_logBuffer.ToString());
        if (_logScrollToBottom) { ImGui.SetScrollHereY(1f); _logScrollToBottom = false; }
        ImGui.EndChild();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Actions
    // ─────────────────────────────────────────────────────────────────────────

    private void RunPreview()
    {
        _postConversionNotice = null;
        SaveInputsToConfig();

        var (ok, err) = _plugin.Converter.ValidateModDirectory(_modDirInput);
        if (!ok) { _task.ErrorMessage = err; return; }

        if (_sourceItemIdx < 0 || _sourceItemIdx >= _detectedItems.Count)
        {
            _task.ErrorMessage = "Select a source item from the detected list.";
            return;
        }

        var src = _detectedItems[_sourceItemIdx];

        // target index resolves against the same list shown in the UI
        var showList = _targetResults.Count > 0 ? _targetResults : _allSlotItems;

        if (_targetItemIdx < 0 || _targetItemIdx >= showList.Count)
        {
            _task.ErrorMessage = "Select a target item.";
            return;
        }

        var tgt = showList[_targetItemIdx];

        _task = new ConversionTask
        {
            ModDirectory  = _modDirInput,
            Slot          = src.Slot,
            OldIdPadded   = src.ModelIdPadded,
            NewIdPadded   = tgt.ModelIdPadded,
            TargetVariant = (int)tgt.Variant,
            SourceVariant = (int)src.Variant,
            TargetSlot    = (src.IsAccessory && _accessoryOutputSlot != src.Slot)
                                ? _accessoryOutputSlot
                                : (EquipSlot?)null,
        };

        _jsonExpanded.Clear();
        _plugin.Converter.PlanConversion(_task);
        _previewDirty = false;

        // Auto-update the new mod name if it hasn't been manually customised.
        if (_newModNameIsDefault || string.IsNullOrEmpty(_newModName))
            RefreshDefaultNewModName();

        if (_task.IsPlanned)
        {
            AppendLog($"Preview complete:  {_task.PlannedRenames.Count} renames, " +
                      $"{_task.PlannedJsonChanges.Sum(j => j.Changes.Count)} JSON changes, " +
                      $"{_task.PlannedBinaryPatches.Sum(b => b.Patches.Count)} binary patches.");
            var slotNote = _task.TargetSlot.HasValue
                ? $" [{SlotInfo.LabelMap[src.Slot]} \u2192 {SlotInfo.LabelMap[_task.TargetSlot.Value]}]"
                : string.Empty;
            AppendLog($"   {src.ItemName} (ID {src.ModelIdDisplay})  \u2192  {tgt.Name} (ID {tgt.ModelIdDisplay}){slotNote}");
        }
        else
        {
            AppendLog($"Preview failed: {_task.ErrorMessage}");
        }
    }

    private void RunApply()
    {
        if (_createNewMod)
        {
            RunApplyNewMod();
            return;
        }

        _lastApplyLogOffset = _logBuffer.Length;
        _postConversionNotice = null;
        _plugin.Converter.ApplyConversion(_task, AppendLog);
        if (!_task.IsApplied) return;

        // ── Post-conversion verification ──────────────────────────────────
        AppendLog("Scanning for leftover old-item references...");
        var leftovers = _plugin.Converter.VerifyConversion(_task);
        if (leftovers.Count == 0)
        {
            AppendLog("Verification passed — no leftover references found.");
        }
        else
        {
            AppendLog($"[WARNING]  {leftovers.Count} leftover reference(s) found after conversion:");
            foreach (var hit in leftovers)
            {
                AppendLog($"  [{hit.HitType.ToUpper()}] {hit.Detail}");
            }
            AppendLog("These may be intentional (shared textures, unrelated groups) or missed changes.");
        }

        if (_penumbraAvailable)
        {
            ReloadInPenumbra();
            _postConversionNotice = "Conversion applied and mod reloaded in Penumbra.";
        }
        else
        {
            _postConversionNotice = "Conversion applied successfully.";
        }
    }

    private void RunApplyNewMod()
    {
        _lastApplyLogOffset   = _logBuffer.Length;
        _postConversionNotice = null;

        if (string.IsNullOrWhiteSpace(_newModName))
        {
            _task.ErrorMessage = "Enter a name for the new mod before creating it.";
            return;
        }

        var parent = Path.GetDirectoryName(
            _modDirInput.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        if (string.IsNullOrEmpty(parent))
        {
            _task.ErrorMessage = "Cannot determine the parent directory of the source mod.";
            return;
        }

        var safeName  = ModConverterService.SanitizeFolderName(_newModName);
        var newModDir = Path.Combine(parent, safeName);

        if (Directory.Exists(newModDir))
        {
            _task.ErrorMessage = $"A folder named '{safeName}' already exists in {parent}.";
            return;
        }

        var resultDir = _plugin.Converter.CreateNewModFromAssetChain(_task, newModDir, _newModName, AppendLog);
        if (resultDir == null) return;

        // ── Post-creation verification ────────────────────────────────────
        AppendLog("Scanning new mod for leftover old-item references...");
        var verifyTask = new ConversionTask
        {
            ModDirectory = resultDir,
            Slot         = _task.Slot,
            OldIdPadded  = _task.OldIdPadded,
            NewIdPadded  = _task.NewIdPadded,
            TargetSlot   = _task.TargetSlot,
        };
        var leftovers = _plugin.Converter.VerifyConversion(verifyTask);
        if (leftovers.Count == 0)
        {
            AppendLog("Verification passed — no leftover references found.");
        }
        else
        {
            AppendLog($"[WARNING]  {leftovers.Count} leftover reference(s) found:");
            foreach (var hit in leftovers)
                AppendLog($"  [{hit.HitType.ToUpper()}] {hit.Detail}");
            AppendLog("These may be intentional (shared textures, unrelated groups) or missed changes.");
        }

        // ── Register and reload in Penumbra ──────────────────────────────
        if (_penumbraAvailable)
        {
            var dirName = Path.GetFileName(resultDir);

            // AddMod registers the folder in Penumbra's mod list
            // ReloadMod then forces a full re-read of files from disk so metadata/options are populated immediately.
            bool added   = _plugin.PenumbraIpc.AddMod(dirName);
            bool reloaded = added && _plugin.PenumbraIpc.ReloadMod(dirName);

            if (reloaded)
            {
                AppendLog($"New mod registered and reloaded in Penumbra: {dirName}");
                _postConversionNotice = $"New mod '{_newModName}' is now visible in Penumbra.";
            }
            else if (added)
            {
                AppendLog($"New mod registered in Penumbra: {dirName} (reload may be needed).");
                _postConversionNotice = $"New mod '{_newModName}' added to Penumbra. Reload it if it appears empty.";
            }
            else
            {
                AppendLog($"[WARNING] Could not register '{dirName}' with Penumbra automatically. Add it manually via Rediscover Mods.");
                _postConversionNotice = $"New mod '{_newModName}' created. Trigger 'Rediscover Mods' in Penumbra to see it.";
            }

            // Refresh the in-plugin mod list so the new entry shows up in the picker.
            RefreshModList();
        }
        else
        {
            _postConversionNotice = $"New mod '{_newModName}' created at: {resultDir}";
        }

        // Persist the chosen name.
        _plugin.Configuration.LastNewModName = _newModName;
        _plugin.Configuration.Save();
    }

    private void ReloadInPenumbra()
    {
        if (!_penumbraAvailable) { AppendLog("Penumbra is not available."); return; }

        // Derive the mod folder name (last path component) for the IPC call
        var dirName = Path.GetFileName(_modDirInput.TrimEnd(Path.DirectorySeparatorChar,
                                                            Path.AltDirectorySeparatorChar));
        bool success = _plugin.PenumbraIpc.ReloadMod(dirName);
        AppendLog(success ? $"Mod reloaded in Penumbra: {dirName}"
                          : $"[WARNING] Penumbra ReloadMod returned an error for '{dirName}'.");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    private void RefreshModList()
    {
        var mods = _plugin.PenumbraIpc.GetModList();
        if (mods == null || mods.Count == 0)
        {
            _penumbraMods = null;
            _modNames     = null;
            _modDirs      = null;
            return;
        }

        // Penumbra returns relative folder names; prepend the mod root to get full paths.
        var root   = _plugin.PenumbraIpc.GetModDirectory()?.TrimEnd('/', '\\') ?? string.Empty;
        var sorted = mods.OrderBy(kv => kv.Value, StringComparer.OrdinalIgnoreCase).ToList();
        _penumbraMods = mods;
        _modDirs  = sorted.Select(kv =>
            string.IsNullOrEmpty(root) ? kv.Key : Path.Combine(root, kv.Key)).ToArray();
        _modNames = sorted.Select(kv => kv.Value).ToArray();
        _modPickerIndex = -1;
    }

    /// <summary>
    /// Scans the current mod directory for item tokens and populates <see cref="_detectedItems"/>.
    /// Resets source/target selections.
    /// </summary>
    private void ScanAndDetectItems()
    {
        _detectedItems  = _plugin.GameData.ScanModForItems(_modDirInput);
        _sourceItemIdx  = -1;
        _targetItemIdx  = -1;
        _allSlotItems.Clear();
        _targetResults.Clear();
        _targetSearch   = string.Empty;
        MarkDirty();

        AppendLog(_detectedItems.Count > 0
            ? $"Scan found {_detectedItems.Count} item(s) in mod."
            : "Scan found no equipment/accessory items in mod.");

        // Auto-select the only item and load target list immediately
        if (_detectedItems.Count == 1)
        {
            _sourceItemIdx = 0;
            _accessoryOutputSlot = _detectedItems[0].Slot;
            RefreshTargetSlotItems();
        }
    }

    /// <summary>
    /// Loads (or reloads) the full item list for the current source slot, then applies any active filter.
    /// </summary>
    private void RefreshTargetSlotItems()
    {
        if (_sourceItemIdx < 0 || _sourceItemIdx >= _detectedItems.Count) return;

        var src  = _detectedItems[_sourceItemIdx];
        var slot = src.IsAccessory ? _accessoryOutputSlot : src.Slot;
        _allSlotItems = _plugin.GameData.GetAllItemsForSlot(slot);

        _targetItemIdx = -1;
        _targetResults.Clear();
        ApplyTargetFilter();
    }

    /// <summary>
    /// Filters <see cref="_allSlotItems"/> by the current <see cref="_targetSearch"/> text
    /// into <see cref="_targetResults"/>. An empty filter leaves results empty so the full
    /// list is shown directly.
    /// </summary>
    private void ApplyTargetFilter()
    {
        if (string.IsNullOrWhiteSpace(_targetSearch))
        {
            _targetResults.Clear();
            _targetItemIdx = -1;
            return;
        }

        var q = _targetSearch.Trim();
        bool StartsWithQ(GameItem i) => i.Name.StartsWith(q, StringComparison.OrdinalIgnoreCase)
                                     || i.ModelIdPadded.StartsWith(q, StringComparison.OrdinalIgnoreCase)
                                     || i.ModelIdDisplay.StartsWith(q, StringComparison.OrdinalIgnoreCase);
        bool ContainsQ(GameItem i)   => i.Name.Contains(q, StringComparison.OrdinalIgnoreCase)
                                     || i.ModelIdPadded.Contains(q, StringComparison.OrdinalIgnoreCase)
                                     || i.ModelIdDisplay.Contains(q, StringComparison.OrdinalIgnoreCase);

        var starts   = _allSlotItems
            .Where(StartsWithQ)
            .OrderBy(i => i.Name, StringComparer.OrdinalIgnoreCase);
        var contains = _allSlotItems
            .Where(i => !StartsWithQ(i) && ContainsQ(i))
            .OrderBy(i => i.Name, StringComparer.OrdinalIgnoreCase);

        _targetResults = starts.Concat(contains).ToList();
        _targetItemIdx = -1;
    }

    private List<int> FilteredMods()
    {
        if (_modNames == null) return new();
        var result = new List<int>();
        for (int i = 0; i < _modNames.Length; i++)
        {
            if (string.IsNullOrEmpty(_modSearch) ||
                _modNames[i].Contains(_modSearch, StringComparison.OrdinalIgnoreCase) ||
                (_modDirs![i].Contains(_modSearch, StringComparison.OrdinalIgnoreCase)))
                result.Add(i);
        }
        return result;
    }

    private void DrawCollapsibleSection(string header, Action content)
    {
        ImGui.SetNextItemOpen(true, ImGuiCond.FirstUseEver);
        if (ImGui.CollapsingHeader(header))
        {
            ImGui.Indent(8);
            content();
            ImGui.Unindent(8);
            ImGui.Spacing();
        }
    }

    private void DrawSelectAllButtons(string tag)
    {
        if (ImGui.SmallButton($"Select All##{tag}sa"))
            SetAllSelected(tag, true);
        ImGui.SameLine();
        if (ImGui.SmallButton($"Clear All##{tag}ca"))
            SetAllSelected(tag, false);
        ImGui.Spacing();
    }

    private void SetAllSelected(string tag, bool state)
    {
        switch (tag)
        {
            case "renames":
                foreach (var r in _task.PlannedRenames) r.Selected = state;
                break;
            case "jsons":
                foreach (var j in _task.PlannedJsonChanges) { j.Selected = state; foreach (var c in j.Changes) c.Selected = state; }
                break;
            case "binaries":
                foreach (var b in _task.PlannedBinaryPatches) { b.Selected = state; foreach (var p in b.Patches) p.Selected = state; }
                break;
        }
    }

    private void MarkDirty()
    {
        _previewDirty = true;
        if (_newModNameIsDefault)
            RefreshDefaultNewModName();
    }

    private void SaveInputsToConfig()
    {
        var cfg = _plugin.Configuration;
        cfg.LastModDirectory = _modDirInput;
        cfg.CreateNewMod     = _createNewMod;
        cfg.Save();
    }

    /// <summary>
    /// Returns the display name of the selected mod: prefers the Penumbra mod-list name,
    /// falls back to the folder name.
    /// </summary>
    private string GetModDisplayName()
    {
        if (_modPickerIndex >= 0 && _modNames != null && _modPickerIndex < _modNames.Length)
            return _modNames[_modPickerIndex];
        return Path.GetFileName(_modDirInput.TrimEnd(
            Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
    }

    /// <summary>
    /// Regenerates <see cref="_newModName"/> from the current source mod name and effective
    /// target slot, unless the user has manually edited it.
    /// </summary>
    private void RefreshDefaultNewModName()
    {
        var effectiveSlot = (_sourceItemIdx >= 0 && _sourceItemIdx < _detectedItems.Count)
            ? (_detectedItems[_sourceItemIdx].IsAccessory ? _accessoryOutputSlot
                                                          : _detectedItems[_sourceItemIdx].Slot)
            : (EquipSlot?)null;

        var slotLabel = effectiveSlot.HasValue ? SlotInfo.LabelMap[effectiveSlot.Value] : string.Empty;
        var baseName  = GetModDisplayName();

        _newModName          = string.IsNullOrEmpty(slotLabel) ? baseName : $"{baseName} ({slotLabel})";
        _newModNameIsDefault = true;
    }

    private void AppendLog(string msg)
    {
        _logBuffer.AppendLine($"[{DateTime.Now:HH:mm:ss}] {msg}");
        _logScrollToBottom = true;
        Plugin.Log.Information("[APIC] {0}", msg);
    }

    private static string TypeLabel(string changeType) => changeType switch
    {
        "path_key"    => "[Key]   ",
        "path_value"  => "[Path]  ",
        "path_string" => "[String]",
        "numeric_id"  => "[ID]    ",
        _             => "[?]     ",
    };

    private static string Truncate(string s, int max) =>
        s.Length <= max ? s : s[..max] + "…";
}
