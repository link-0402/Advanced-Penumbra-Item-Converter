using System;
using System.Numerics;
using Dalamud.Bindings.ImGui;
using Dalamud.Interface.Windowing;

namespace AdvancedPenumbraItemConverter.Windows;

public sealed class ConfigWindow : Window, IDisposable
{
    private readonly Plugin _plugin;

    public ConfigWindow(Plugin plugin) : base(
        "Advanced Penumbra Item Converter — Configuration###APICConfig",
        ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoScrollbar)
    {
        _plugin = plugin;
        Size          = new Vector2(400, 120);
        SizeCondition = ImGuiCond.Always;
    }

    public void Dispose() { }

    public override void Draw()
    {
        var cfg = _plugin.Configuration;

        bool autoRefresh = cfg.AutoRefreshPreview;
        if (ImGui.Checkbox("Auto-refresh preview when inputs change", ref autoRefresh))
        {
            cfg.AutoRefreshPreview = autoRefresh;
            cfg.Save();
        }
    }
}
