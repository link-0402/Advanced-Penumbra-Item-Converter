using Dalamud.Configuration;
using Dalamud.Plugin;
using System;

namespace AdvancedPenumbraItemConverter;

[Serializable]
public class Configuration : IPluginConfiguration
{
    public int Version { get; set; } = 1;

    /// <summary>Last-used mod directory path (the folder containing default_mod.json).</summary>
    public string LastModDirectory { get; set; } = string.Empty;

    /// <summary>Whether the preview window auto-refreshes when inputs change.</summary>
    public bool AutoRefreshPreview { get; set; } = true;

    /// <summary>When true, Apply creates a new mod instead of modifying in place.</summary>
    public bool CreateNewMod { get; set; } = false;

    /// <summary>Last-used new mod name (used as folder name and Penumbra display name).</summary>
    public string LastNewModName { get; set; } = string.Empty;

    public void Save()
    {
        Plugin.PluginInterface.SavePluginConfig(this);
    }
}
