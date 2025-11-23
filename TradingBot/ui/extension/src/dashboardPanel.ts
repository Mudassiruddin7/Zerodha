/**
 * Dashboard Webview Panel - Real-time P&L, positions, and kill switch
 */

import * as vscode from 'vscode';
import { TradingBotClient } from './tradingBotClient';

export class DashboardPanel {
    public static currentPanel: DashboardPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;
    private readonly _extensionUri: vscode.Uri;
    private _disposables: vscode.Disposable[] = [];
    private _botClient: TradingBotClient;
    private _refreshInterval: NodeJS.Timeout | null = null;

    private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri, botClient: TradingBotClient) {
        this._panel = panel;
        this._extensionUri = extensionUri;
        this._botClient = botClient;

        this._update();
        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

        // Handle messages from the webview
        this._panel.webview.onDidReceiveMessage(
            message => {
                switch (message.command) {
                    case 'killSwitch':
                        this.activateKillSwitch();
                        return;
                    case 'refresh':
                        this._botClient.requestDashboardData();
                        return;
                }
            },
            null,
            this._disposables
        );

        // Listen for bot updates
        this._botClient.on('status', (status: any) => {
            this._panel.webview.postMessage({ type: 'status', data: status });
        });

        this._botClient.on('position_update', (positions: any) => {
            this._panel.webview.postMessage({ type: 'positions', data: positions });
        });

        this._botClient.on('pnl_update', (pnl: any) => {
            this._panel.webview.postMessage({ type: 'pnl', data: pnl });
        });

        // Request initial data
        this._botClient.requestDashboardData();

        // Auto-refresh
        const config = vscode.workspace.getConfiguration('tradingBot');
        const refreshInterval = config.get<number>('refreshInterval', 5000);
        this._refreshInterval = setInterval(() => {
            this._botClient.requestDashboardData();
        }, refreshInterval);
    }

    public static createOrShow(extensionUri: vscode.Uri, botClient: TradingBotClient) {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;

        if (DashboardPanel.currentPanel) {
            DashboardPanel.currentPanel._panel.reveal(column);
            return;
        }

        const panel = vscode.window.createWebviewPanel(
            'tradingDashboard',
            'Trading Dashboard',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                localResourceRoots: [vscode.Uri.joinPath(extensionUri, 'media')],
            }
        );

        DashboardPanel.currentPanel = new DashboardPanel(panel, extensionUri, botClient);
    }

    public dispose() {
        DashboardPanel.currentPanel = undefined;

        if (this._refreshInterval) {
            clearInterval(this._refreshInterval);
        }

        this._panel.dispose();

        while (this._disposables.length) {
            const x = this._disposables.pop();
            if (x) {
                x.dispose();
            }
        }
    }

    private async activateKillSwitch() {
        const answer = await vscode.window.showWarningMessage(
            'EMERGENCY STOP - Are you absolutely sure?',
            { modal: true },
            'Yes, STOP ALL TRADING',
            'Cancel'
        );

        if (answer === 'Yes, STOP ALL TRADING') {
            await this._botClient.emergencyStop();
            vscode.window.showErrorMessage('EMERGENCY STOP ACTIVATED');
        }
    }

    private _update() {
        const webview = this._panel.webview;
        this._panel.title = 'Trading Dashboard';
        this._panel.webview.html = this._getHtmlForWebview(webview);
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            padding: 20px;
            background: var(--vscode-editor-background);
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid var(--vscode-panel-border);
        }
        h1 {
            font-size: 28px;
        }
        .kill-switch {
            padding: 15px 30px;
            background: #d32f2f;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: background 0.3s;
        }
        .kill-switch:hover {
            background: #b71c1c;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            padding: 20px;
            background: var(--vscode-input-background);
            border-radius: 8px;
            border-left: 4px solid var(--vscode-focusBorder);
        }
        .metric-label {
            font-size: 14px;
            opacity: 0.7;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
        }
        .positive {
            color: #4caf50;
        }
        .negative {
            color: #f44336;
        }
        .section {
            margin-bottom: 30px;
        }
        .section-title {
            font-size: 20px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--vscode-panel-border);
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--vscode-panel-border);
        }
        th {
            background: var(--vscode-input-background);
            font-weight: bold;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-active {
            background: #4caf50;
        }
        .status-inactive {
            background: #f44336;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>
            <span class="status-indicator status-inactive" id="status-indicator"></span>
            Trading Bot Dashboard
        </h1>
        <button class="kill-switch" onclick="killSwitch()">⚠️ EMERGENCY STOP</button>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Total P&L</div>
            <div class="metric-value" id="total-pnl">₹0.00</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Daily P&L</div>
            <div class="metric-value" id="daily-pnl">₹0.00</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Open Positions</div>
            <div class="metric-value" id="open-positions">0</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value" id="win-rate">0%</div>
        </div>
    </div>

    <div class="section">
        <h2 class="section-title">Open Positions</h2>
        <table>
            <thead>
                <tr>
                    <th>Instrument</th>
                    <th>Direction</th>
                    <th>Quantity</th>
                    <th>Entry Price</th>
                    <th>Current Price</th>
                    <th>P&L</th>
                </tr>
            </thead>
            <tbody id="positions-table">
                <tr>
                    <td colspan="6" style="text-align: center; opacity: 0.5;">No open positions</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2 class="section-title">Today's Trades</h2>
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Instrument</th>
                    <th>Direction</th>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>P&L</th>
                </tr>
            </thead>
            <tbody id="trades-table">
                <tr>
                    <td colspan="6" style="text-align: center; opacity: 0.5;">No trades today</td>
                </tr>
            </tbody>
        </table>
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        window.addEventListener('message', event => {
            const message = event.data;
            switch (message.type) {
                case 'status':
                    updateStatus(message.data);
                    break;
                case 'positions':
                    updatePositions(message.data);
                    break;
                case 'pnl':
                    updatePnL(message.data);
                    break;
            }
        });

        function updateStatus(status) {
            const indicator = document.getElementById('status-indicator');
            indicator.className = 'status-indicator ' + (status.active ? 'status-active' : 'status-inactive');
            
            document.getElementById('open-positions').textContent = status.positions || 0;
        }

        function updatePnL(pnl) {
            const totalPnl = document.getElementById('total-pnl');
            const dailyPnl = document.getElementById('daily-pnl');
            const winRate = document.getElementById('win-rate');

            totalPnl.textContent = '₹' + (pnl.total || 0).toFixed(2);
            totalPnl.className = 'metric-value ' + (pnl.total >= 0 ? 'positive' : 'negative');

            dailyPnl.textContent = '₹' + (pnl.daily || 0).toFixed(2);
            dailyPnl.className = 'metric-value ' + (pnl.daily >= 0 ? 'positive' : 'negative');

            winRate.textContent = ((pnl.win_rate || 0) * 100).toFixed(1) + '%';
        }

        function updatePositions(positions) {
            const tbody = document.getElementById('positions-table');
            
            if (!positions || positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; opacity: 0.5;">No open positions</td></tr>';
                return;
            }

            tbody.innerHTML = positions.map(pos => \`
                <tr>
                    <td>\${pos.instrument}</td>
                    <td>\${pos.direction}</td>
                    <td>\${pos.quantity}</td>
                    <td>₹\${pos.entry_price.toFixed(2)}</td>
                    <td>₹\${pos.current_price.toFixed(2)}</td>
                    <td class="\${pos.pnl >= 0 ? 'positive' : 'negative'}">₹\${pos.pnl.toFixed(2)}</td>
                </tr>
            \`).join('');
        }

        function killSwitch() {
            if (confirm('ARE YOU SURE YOU WANT TO STOP ALL TRADING? This action cannot be undone.')) {
                vscode.postMessage({ command: 'killSwitch' });
            }
        }

        // Request initial data
        vscode.postMessage({ command: 'refresh' });
    </script>
</body>
</html>`;
    }
}
