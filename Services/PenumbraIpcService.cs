using System;
using System.Collections.Generic;
using Dalamud.Plugin;
using Dalamud.Plugin.Ipc;
using Dalamud.Plugin.Services;

namespace AdvancedPenumbraItemConverter.Services;

/// <summary>
/// Thin wrapper around Penumbra's IPC channel.
/// All calls gracefully return null/false when Penumbra is unavailable.
/// </summary>
public sealed class PenumbraIpcService : IDisposable
{
    private readonly IDalamudPluginInterface _pi;
    private readonly IPluginLog              _log;

    // ── IPC subscribers ───────────────────────────────────────────────────────
    // We cache the subscribers to avoid creating new objects on every call.

    private readonly ICallGateSubscriber<(int Breaking, int Features)>                                 _apiVersion;
    private readonly ICallGateSubscriber<string>                                                        _getModDirectory;
    private readonly ICallGateSubscriber<Dictionary<string, string>>                                    _getModList;
    private readonly ICallGateSubscriber<string, string, int>                                           _reloadMod;
    private readonly ICallGateSubscriber<string, int>                                                   _addMod;
    private readonly ICallGateSubscriber<string, string, (int, string, bool, bool)>                    _getModPath;
    private readonly ICallGateSubscriber<Dictionary<Guid, string>>                                            _getCollections;
    private readonly ICallGateSubscriber<int, (bool ObjectValid, bool IndividualSet, (Guid Id, string Name))> _getCollectionForObject;

    // ── Events ────────────────────────────────────────────────────────────────
    /// <summary>Raised when Penumbra signals it has fully initialised.</summary>
    public event Action? PenumbraInitialized;
    /// <summary>Raised when Penumbra is disposing.</summary>
    public event Action? PenumbraDisposed;

    private ICallGateSubscriber<object>? _initializedSub;
    private ICallGateSubscriber<object>? _disposedSub;

    public PenumbraIpcService(IDalamudPluginInterface pi, IPluginLog log)
    {
        _pi  = pi;
        _log = log;

        _apiVersion      = pi.GetIpcSubscriber<(int, int)>             ("Penumbra.ApiVersion.V5");
        _getModDirectory = pi.GetIpcSubscriber<string>                 ("Penumbra.GetModDirectory");
        _getModList      = pi.GetIpcSubscriber<Dictionary<string,string>>("Penumbra.GetModList");
        _reloadMod       = pi.GetIpcSubscriber<string, string, int>    ("Penumbra.ReloadMod.V5");
        _addMod          = pi.GetIpcSubscriber<string, int>            ("Penumbra.AddMod.V5");
        _getModPath      = pi.GetIpcSubscriber<string, string, (int, string, bool, bool)>("Penumbra.GetModPath.V5");
        _getCollections         = pi.GetIpcSubscriber<Dictionary<Guid, string>>("Penumbra.GetCollections.V5");
        _getCollectionForObject = pi.GetIpcSubscriber<int, (bool, bool, (Guid, string))>("Penumbra.GetCollectionForObject.V5");

        // Subscribe to lifecycle events
        try
        {
            _initializedSub = pi.GetIpcSubscriber<object>("Penumbra.Initialized");
            _initializedSub.Subscribe(OnPenumbraInitialized);
        }
        catch (Exception ex) { _log.Debug(ex, "[APIC] Could not subscribe to Penumbra.Initialized"); }

        try
        {
            _disposedSub = pi.GetIpcSubscriber<object>("Penumbra.Disposed");
            _disposedSub.Subscribe(OnPenumbraDisposed);
        }
        catch (Exception ex) { _log.Debug(ex, "[APIC] Could not subscribe to Penumbra.Disposed"); }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    private void OnPenumbraInitialized() => PenumbraInitialized?.Invoke();
    private void OnPenumbraDisposed()    => PenumbraDisposed?.Invoke();

    public void Dispose()
    {
        try { _initializedSub?.Unsubscribe(OnPenumbraInitialized); } catch { /* ignore */ }
        try { _disposedSub?.Unsubscribe(OnPenumbraDisposed); }       catch { /* ignore */ }
    }

    // ── Availability check ────────────────────────────────────────────────────

    /// <summary>Returns true when Penumbra is loaded and its IPC is reachable.</summary>
    public bool IsAvailable
    {
        get
        {
            try
            {
                var (breaking, _) = _apiVersion.InvokeFunc();
                return breaking == 5;
            }
            catch { return false; }
        }
    }

    // ── API calls ─────────────────────────────────────────────────────────────

    /// <summary>Returns Penumbra's configured mod root directory, or null on failure.</summary>
    public string? GetModDirectory()
    {
        try   { return _getModDirectory.InvokeFunc(); }
        catch (Exception ex) { _log.Warning(ex, "[APIC] GetModDirectory failed"); return null; }
    }

    /// <summary>Returns a dict of modDirectory → modName for all known mods.</summary>
    public Dictionary<string, string>? GetModList()
    {
        try   { return _getModList.InvokeFunc(); }
        catch (Exception ex) { _log.Warning(ex, "[APIC] GetModList failed"); return null; }
    }

    /// <summary>
    /// Asks Penumbra to reload a mod from disk.
    /// <paramref name="modDirectory"/> is the folder name under the Penumbra root (not a full path).
    /// Returns true on success.
    /// </summary>
    public bool ReloadMod(string modDirectory, string modName = "")
    {
        try
        {
            var rc = (PenumbraApiEc)_reloadMod.InvokeFunc(modDirectory, modName);
            if (rc != PenumbraApiEc.Success)
                _log.Warning($"[APIC] ReloadMod returned {rc} for '{modDirectory}'");
            return rc == PenumbraApiEc.Success;
        }
        catch (Exception ex) { _log.Warning(ex, "[APIC] ReloadMod failed"); return false; }
    }

    /// <summary>
    /// Registers a new mod folder (already created inside the Penumbra mod root) in
    /// Penumbra's mod list so it is immediately visible without a full rediscover.
    /// <paramref name="modDirectory"/> is the folder name only (not a full path).
    /// Returns true on success.
    /// </summary>
    public bool AddMod(string modDirectory)
    {
        try
        {
            var rc = (PenumbraApiEc)_addMod.InvokeFunc(modDirectory);
            if (rc != PenumbraApiEc.Success && rc != PenumbraApiEc.NothingDone)
                _log.Warning($"[APIC] AddMod returned {rc} for '{modDirectory}'");
            return rc == PenumbraApiEc.Success || rc == PenumbraApiEc.NothingDone;
        }
        catch (Exception ex) { _log.Warning(ex, "[APIC] AddMod failed"); return false; }
    }

    /// <summary>
    /// Retrieves the full on-disk path for a specific mod.
    /// Returns (success, fullPath).
    /// </summary>
    public (bool Success, string FullPath) GetModPath(string modDirectory, string modName = "")
    {
        try
        {
            var (ret, fullPath, _, _) = _getModPath.InvokeFunc(modDirectory, modName);
            return ((PenumbraApiEc)ret == PenumbraApiEc.Success, fullPath);
        }
        catch (Exception ex) { _log.Warning(ex, "[APIC] GetModPath failed"); return (false, string.Empty); }
    }

    /// <summary>
    /// Returns all Penumbra collections as (Guid Id, string Name) pairs, or null on failure.
    /// </summary>
    public Dictionary<Guid, string>? GetCollections()
    {
        try   { return _getCollections.InvokeFunc(); }
        catch (Exception ex) { _log.Debug(ex, "[APIC] GetCollections failed"); return null; }
    }

    /// <summary>
    /// Returns the collection currently assigned to a game object by table index.
    /// Index 0 is always the player character.
    /// Returns null when there is no valid object at that index or on failure.
    /// </summary>
    public (Guid Id, string Name)? GetCollectionForObject(int gameObjectIndex)
    {
        try
        {
            var (objectValid, _, collection) = _getCollectionForObject.InvokeFunc(gameObjectIndex);
            return objectValid ? collection : null;
        }
        catch (Exception ex)
        {
            _log.Debug(ex, "[APIC] GetCollectionForObject failed for index {0}", gameObjectIndex);
            return null;
        }
    }
}

/// <summary>Mirrors Penumbra's PenumbraApiEc enum (int values).</summary>
public enum PenumbraApiEc : int
{
    Success            = 0,
    NothingDone        = 1,
    InvalidArgument    = 2,
    ModMissing         = 3,
    CollectionMissing  = 4,
    OptionGroupMissing = 5,
    OptionMissing      = 6,
    PathMissing        = 7,
    Default            = 8,
    SystemCollection   = 9,
    Disposed           = 10,
    UnknownError       = 11,
}
