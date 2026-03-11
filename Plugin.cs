using System;
using Dalamud.Game.Command;
using Dalamud.IoC;
using Dalamud.Plugin;
using Dalamud.Plugin.Services;
using Dalamud.Interface.Windowing;
using AdvancedPenumbraItemConverter.Services;
using AdvancedPenumbraItemConverter.Windows;

namespace AdvancedPenumbraItemConverter;

public sealed class Plugin : IDalamudPlugin
{
    // ── Dalamud services ──────────────────────────────────────────────────────
    [PluginService] internal static IDalamudPluginInterface PluginInterface { get; private set; } = null!;
    [PluginService] internal static ICommandManager         CommandManager  { get; private set; } = null!;
    [PluginService] internal static IPluginLog              Log             { get; private set; } = null!;
    [PluginService] internal static IDataManager            DataManager     { get; private set; } = null!;

    // ── Plugin internals ──────────────────────────────────────────────────────
    internal Configuration        Configuration  { get; }
    internal PenumbraIpcService   PenumbraIpc    { get; }
    internal ModConverterService  Converter      { get; }
    internal GameDataService      GameData       { get; }

    public   readonly WindowSystem WindowSystem = new("AdvancedPenumbraItemConverter");
    private  ConfigWindow          ConfigWindow  { get; }
    private  MainWindow            MainWindow    { get; }

    private const string CommandName    = "/apic";
    private const string CommandConfig  = "/apicconfig";

    public Plugin()
    {
        Configuration = PluginInterface.GetPluginConfig() as Configuration ?? new Configuration();

        PenumbraIpc = new PenumbraIpcService(PluginInterface, Log);
        GameData    = new GameDataService(DataManager, Log);
        Converter   = new ModConverterService(Log, GameData);

        // ── Windows ───────────────────────────────────────────────────────────
        ConfigWindow = new ConfigWindow(this);
        MainWindow   = new MainWindow(this);

        WindowSystem.AddWindow(ConfigWindow);
        WindowSystem.AddWindow(MainWindow);

        // ── Commands ──────────────────────────────────────────────────────────
        CommandManager.AddHandler(CommandName, new CommandInfo(OnMainCommand)
        {
            HelpMessage = "Open the Advanced Penumbra Item Converter window."
        });
        CommandManager.AddHandler(CommandConfig, new CommandInfo(OnConfigCommand)
        {
            HelpMessage = "Open the Advanced Penumbra Item Converter configuration."
        });

        // ── UI hooks ──────────────────────────────────────────────────────────
        PluginInterface.UiBuilder.Draw          += WindowSystem.Draw;
        PluginInterface.UiBuilder.OpenConfigUi  += ToggleConfigUi;
        PluginInterface.UiBuilder.OpenMainUi    += ToggleMainUi;

        // ── Penumbra lifecycle ────────────────────────────────────────────────
        PenumbraIpc.PenumbraInitialized += OnPenumbraInitialized;
        PenumbraIpc.PenumbraDisposed    += OnPenumbraDisposed;

        Log.Information("[APIC] Advanced Penumbra Item Converter loaded.");
    }

    public void Dispose()
    {
        PluginInterface.UiBuilder.Draw         -= WindowSystem.Draw;
        PluginInterface.UiBuilder.OpenConfigUi -= ToggleConfigUi;
        PluginInterface.UiBuilder.OpenMainUi   -= ToggleMainUi;

        PenumbraIpc.PenumbraInitialized -= OnPenumbraInitialized;
        PenumbraIpc.PenumbraDisposed    -= OnPenumbraDisposed;

        PenumbraIpc.Dispose();

        WindowSystem.RemoveAllWindows();
        ConfigWindow.Dispose();
        MainWindow.Dispose();

        CommandManager.RemoveHandler(CommandName);
        CommandManager.RemoveHandler(CommandConfig);
    }

    // ── Command handlers ──────────────────────────────────────────────────────

    private void OnMainCommand   (string cmd, string args) => MainWindow.Toggle();
    private void OnConfigCommand (string cmd, string args) => ConfigWindow.Toggle();

    public void ToggleMainUi()   => MainWindow.Toggle();
    public void ToggleConfigUi() => ConfigWindow.Toggle();

    // ── Penumbra lifecycle callbacks ──────────────────────────────────────────

    private void OnPenumbraInitialized()
    {
        Log.Information("[APIC] Penumbra became available.");
        MainWindow.OnPenumbraStateChanged();
    }

    private void OnPenumbraDisposed()
    {
        Log.Warning("[APIC] Penumbra became unavailable.");
        MainWindow.OnPenumbraStateChanged();
    }
}
