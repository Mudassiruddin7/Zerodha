# VS Code Extension for Zerodha Trading Bot

## Overview

This VS Code extension provides a graphical user interface for the Zerodha algorithmic trading bot.

## Features

### 1. Trade Preview Panel
- Real-time trade signal preview
- Shows confidence, expected profit, risk metrics
- Manual approve/reject buttons
- Entry/exit price details

### 2. Trading Dashboard
- Real-time P&L tracking
- Open positions monitoring
- Today's trades history
- Win rate and performance metrics
- Status indicator (active/inactive)

### 3. Emergency Kill Switch
- Immediate stop all trading
- Close all positions
- Halt the bot
- Modal confirmation for safety

### 4. Status Bar Integration
- Shows bot status in VS Code status bar
- Current P&L display
- Number of open positions
- Click to open dashboard

## Installation

### From Source

```bash
cd ui/extension
npm install
npm run compile
```

### Install Extension

1. Press `F5` to launch extension development host
2. Or package and install:
   ```bash
   npm install -g @vscode/vsce
   vsce package
   code --install-extension zerodha-trading-bot-ui-1.0.0.vsix
   ```

## Usage

### Opening Trade Preview
1. Press `Ctrl+Shift+P` (Cmd+Shift+P on Mac)
2. Type "Show Trade Preview"
3. Or click status bar item

### Opening Dashboard
1. Press `Ctrl+Shift+P`
2. Type "Show Trading Dashboard"
3. Monitor P&L, positions, and trades in real-time

### Emergency Stop
1. Press `Ctrl+Shift+P`
2. Type "Emergency Kill Switch"
3. Confirm to stop all trading

## Configuration

Open VS Code settings and configure:

```json
{
  "tradingBot.autoApprove": false,  // NEVER enable in live trading
  "tradingBot.notificationLevel": "all",  // all, signals-only, alerts-only, none
  "tradingBot.refreshInterval": 5000  // Dashboard refresh in ms
}
```

## Communication

The extension communicates with the Python trading bot via WebSocket (default port 8765).

### Message Protocol

**From Bot to Extension:**
- `status`: Bot status updates (active, P&L, positions)
- `signal`: New trade signals requiring approval
- `position_update`: Position changes
- `pnl_update`: P&L updates
- `alert`: Critical alerts

**From Extension to Bot:**
- `approve_signal`: Approve trade execution
- `reject_signal`: Reject trade
- `emergency_stop`: Kill switch activated
- `get_dashboard_data`: Request dashboard data

## Security

- Manual confirmation required for all trades
- Emergency kill switch for immediate halt
- Auto-approve disabled by default
- WebSocket connection on localhost only

## Development

### Structure

```
ui/extension/
├── src/
│   ├── extension.ts           # Main extension entry point
│   ├── tradingBotClient.ts    # WebSocket client
│   ├── tradePreviewPanel.ts   # Trade preview webview
│   └── dashboardPanel.ts      # Dashboard webview
├── package.json               # Extension manifest
└── tsconfig.json             # TypeScript config
```

### Building

```bash
npm run compile
```

### Watching

```bash
npm run watch
```

## Requirements

- VS Code 1.85.0 or higher
- Python trading bot running with WebSocket server
- WebSocket server on `ws://localhost:8765`

## Known Issues

- WebSocket reconnection may take a few seconds
- Dashboard updates with 5-second default interval

## Release Notes

### 1.0.0
- Initial release
- Trade preview panel
- Trading dashboard
- Emergency kill switch
- Status bar integration
