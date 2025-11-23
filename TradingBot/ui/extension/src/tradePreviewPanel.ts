/**
 * Trade Preview Webview Panel
 */

import * as vscode from 'vscode';
import { TradingBotClient } from './tradingBotClient';

export class TradePreviewPanel {
    public static currentPanel: TradePreviewPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;
    private readonly _extensionUri: vscode.Uri;
    private _disposables: vscode.Disposable[] = [];
    private _botClient: TradingBotClient;

    private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri, botClient: TradingBotClient) {
        this._panel = panel;
        this._extensionUri = extensionUri;
        this._botClient = botClient;

        // Set the webview's initial html content
        this._update();

        // Listen for when the panel is disposed
        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

        // Handle messages from the webview
        this._panel.webview.onDidReceiveMessage(
            message => {
                switch (message.command) {
                    case 'approve':
                        this._botClient.approveSignal(message.signalId);
                        vscode.window.showInformationMessage('Trade approved');
                        return;
                    case 'reject':
                        this._botClient.rejectSignal(message.signalId);
                        vscode.window.showInformationMessage('Trade rejected');
                        return;
                }
            },
            null,
            this._disposables
        );

        // Listen for bot updates
        this._botClient.on('signal', (signal: any) => {
            this._panel.webview.postMessage({ type: 'signal', data: signal });
        });
    }

    public static createOrShow(extensionUri: vscode.Uri, botClient: TradingBotClient) {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;

        // If we already have a panel, show it.
        if (TradePreviewPanel.currentPanel) {
            TradePreviewPanel.currentPanel._panel.reveal(column);
            return;
        }

        // Otherwise, create a new panel.
        const panel = vscode.window.createWebviewPanel(
            'tradePreview',
            'Trade Preview',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                localResourceRoots: [vscode.Uri.joinPath(extensionUri, 'media')],
            }
        );

        TradePreviewPanel.currentPanel = new TradePreviewPanel(panel, extensionUri, botClient);
    }

    public dispose() {
        TradePreviewPanel.currentPanel = undefined;

        // Clean up our resources
        this._panel.dispose();

        while (this._disposables.length) {
            const x = this._disposables.pop();
            if (x) {
                x.dispose();
            }
        }
    }

    private _update() {
        const webview = this._panel.webview;
        this._panel.title = 'Trade Preview';
        this._panel.webview.html = this._getHtmlForWebview(webview);
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trade Preview</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            padding: 20px;
        }
        .signal-card {
            border: 1px solid var(--vscode-panel-border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            background: var(--vscode-editor-background);
        }
        .signal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .instrument {
            font-size: 24px;
            font-weight: bold;
        }
        .direction {
            padding: 5px 15px;
            border-radius: 4px;
            font-weight: bold;
        }
        .direction.buy {
            background: #4caf50;
            color: white;
        }
        .direction.sell {
            background: #f44336;
            color: white;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 20px 0;
        }
        .metric {
            text-align: center;
            padding: 15px;
            background: var(--vscode-input-background);
            border-radius: 4px;
        }
        .metric-label {
            font-size: 12px;
            opacity: 0.7;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 20px;
            font-weight: bold;
        }
        .confidence-high {
            color: #4caf50;
        }
        .confidence-medium {
            color: #ff9800;
        }
        .confidence-low {
            color: #f44336;
        }
        .actions {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-top: 20px;
        }
        button {
            padding: 12px 30px;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
            font-weight: bold;
        }
        .approve {
            background: #4caf50;
            color: white;
        }
        .approve:hover {
            background: #45a049;
        }
        .reject {
            background: #f44336;
            color: white;
        }
        .reject:hover {
            background: #da190b;
        }
        .details {
            margin-top: 20px;
            padding: 15px;
            background: var(--vscode-input-background);
            border-radius: 4px;
        }
        .detail-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid var(--vscode-panel-border);
        }
        .detail-row:last-child {
            border-bottom: none;
        }
    </style>
</head>
<body>
    <div id="signals-container">
        <p style="text-align: center; opacity: 0.7;">Waiting for trade signals...</p>
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        window.addEventListener('message', event => {
            const message = event.data;
            if (message.type === 'signal') {
                displaySignal(message.data);
            }
        });

        function displaySignal(signal) {
            const container = document.getElementById('signals-container');
            const confidenceClass = signal.confidence >= 0.85 ? 'confidence-high' : 
                                   signal.confidence >= 0.70 ? 'confidence-medium' : 'confidence-low';
            
            container.innerHTML = \`
                <div class="signal-card">
                    <div class="signal-header">
                        <span class="instrument">\${signal.instrument}</span>
                        <span class="direction \${signal.direction.toLowerCase()}">\${signal.direction}</span>
                    </div>

                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Confidence</div>
                            <div class="metric-value \${confidenceClass}">\${(signal.confidence * 100).toFixed(1)}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Expected Profit</div>
                            <div class="metric-value">\${signal.expected_profit_pct.toFixed(2)}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Quantity</div>
                            <div class="metric-value">\${signal.quantity}</div>
                        </div>
                    </div>

                    <div class="details">
                        <div class="detail-row">
                            <span>Entry Price</span>
                            <span>₹\${signal.entry_price.toFixed(2)}</span>
                        </div>
                        <div class="detail-row">
                            <span>Stop Loss</span>
                            <span>₹\${signal.stop_loss.toFixed(2)}</span>
                        </div>
                        <div class="detail-row">
                            <span>Take Profit</span>
                            <span>₹\${signal.take_profit.toFixed(2)}</span>
                        </div>
                        <div class="detail-row">
                            <span>Risk</span>
                            <span>₹\${((signal.entry_price - signal.stop_loss) * signal.quantity).toFixed(2)}</span>
                        </div>
                        <div class="detail-row">
                            <span>Expected Profit (INR)</span>
                            <span>₹\${(signal.entry_price * signal.quantity * signal.expected_profit_pct / 100).toFixed(2)}</span>
                        </div>
                    </div>

                    <div class="actions">
                        <button class="approve" onclick="approveSignal('\${signal.id}')">Approve Trade</button>
                        <button class="reject" onclick="rejectSignal('\${signal.id}')">Reject</button>
                    </div>
                </div>
            \`;
        }

        function approveSignal(signalId) {
            vscode.postMessage({
                command: 'approve',
                signalId: signalId
            });
        }

        function rejectSignal(signalId) {
            vscode.postMessage({
                command: 'reject',
                signalId: signalId
            });
        }
    </script>
</body>
</html>`;
    }
}
