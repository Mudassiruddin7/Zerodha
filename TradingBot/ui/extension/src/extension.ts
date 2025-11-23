/**
 * VS Code Extension for Zerodha Trading Bot
 * 
 * Provides:
 * - Trade preview with manual confirmation
 * - Real-time dashboard with P&L tracking
 * - Emergency kill switch
 * - Position monitoring
 */

import * as vscode from 'vscode';
import { TradePreviewPanel } from './tradePreviewPanel';
import { DashboardPanel } from './dashboardPanel';
import { TradingBotClient } from './tradingBotClient';

let tradingBotClient: TradingBotClient | undefined;
let statusBarItem: vscode.StatusBarItem;

export function activate(context: vscode.ExtensionContext) {
    console.log('Zerodha Trading Bot UI extension activated');

    // Initialize trading bot client (WebSocket connection to Python bot)
    tradingBotClient = new TradingBotClient();

    // Create status bar item
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBarItem.text = "$(pulse) Trading Bot: Inactive";
    statusBarItem.command = 'zerodha-trading-bot.showDashboard';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('zerodha-trading-bot.showTradePreview', () => {
            TradePreviewPanel.createOrShow(context.extensionUri, tradingBotClient!);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('zerodha-trading-bot.showDashboard', () => {
            DashboardPanel.createOrShow(context.extensionUri, tradingBotClient!);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('zerodha-trading-bot.emergencyKillSwitch', async () => {
            const answer = await vscode.window.showWarningMessage(
                'Are you sure you want to STOP ALL TRADING? This will close all positions and halt the bot.',
                { modal: true },
                'Yes, Stop Trading',
                'Cancel'
            );

            if (answer === 'Yes, Stop Trading') {
                await emergencyKillSwitch();
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('zerodha-trading-bot.approveSignal', (signalId: string) => {
            tradingBotClient?.approveSignal(signalId);
            vscode.window.showInformationMessage(`Trade signal ${signalId} approved`);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('zerodha-trading-bot.rejectSignal', (signalId: string) => {
            tradingBotClient?.rejectSignal(signalId);
            vscode.window.showInformationMessage(`Trade signal ${signalId} rejected`);
        })
    );

    // Listen for bot status updates
    tradingBotClient.on('status', (status: any) => {
        updateStatusBar(status);
    });

    // Listen for new trade signals
    tradingBotClient.on('signal', (signal: any) => {
        handleNewSignal(signal, context);
    });

    // Connect to trading bot
    tradingBotClient.connect();
}

export function deactivate() {
    tradingBotClient?.disconnect();
    statusBarItem?.dispose();
}

function updateStatusBar(status: any) {
    const { active, pnl, positions } = status;
    
    if (active) {
        const pnlColor = pnl >= 0 ? '$(arrow-up)' : '$(arrow-down)';
        statusBarItem.text = `${pnlColor} Trading Bot: ₹${pnl.toFixed(2)} | Positions: ${positions}`;
        statusBarItem.backgroundColor = undefined;
    } else {
        statusBarItem.text = "$(circle-slash) Trading Bot: Inactive";
        statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
    }
}

async function handleNewSignal(signal: any, context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('tradingBot');
    const autoApprove = config.get<boolean>('autoApprove', false);
    const notificationLevel = config.get<string>('notificationLevel', 'all');

    // Show notification
    if (notificationLevel === 'all' || notificationLevel === 'signals-only') {
        const action = await vscode.window.showInformationMessage(
            `New ${signal.direction} signal for ${signal.instrument} | Confidence: ${(signal.confidence * 100).toFixed(1)}% | Expected: ${signal.expected_profit_pct.toFixed(2)}%`,
            'Preview',
            'Approve',
            'Reject'
        );

        if (action === 'Preview') {
            TradePreviewPanel.createOrShow(context.extensionUri, tradingBotClient!);
        } else if (action === 'Approve') {
            tradingBotClient?.approveSignal(signal.id);
        } else if (action === 'Reject') {
            tradingBotClient?.rejectSignal(signal.id);
        }
    }

    // Auto-approve if enabled (DANGEROUS!)
    if (autoApprove && !signal.requires_manual_confirmation) {
        tradingBotClient?.approveSignal(signal.id);
    }
}

async function emergencyKillSwitch() {
    try {
        // Send kill switch command to bot
        await tradingBotClient?.emergencyStop();

        // Update UI
        statusBarItem.text = "$(error) Trading Bot: EMERGENCY STOPPED";
        statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');

        vscode.window.showErrorMessage(
            'EMERGENCY STOP ACTIVATED - All trading halted. Positions will be closed.',
            { modal: true }
        );
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to activate kill switch: ${error}`);
    }
}
