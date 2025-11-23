/**
 * WebSocket client to communicate with Python trading bot
 */

import * as WebSocket from 'ws';
import { EventEmitter } from 'events';

export class TradingBotClient extends EventEmitter {
    private ws: WebSocket | null = null;
    private reconnectInterval: NodeJS.Timeout | null = null;
    private readonly wsUrl: string = 'ws://localhost:8765';

    constructor() {
        super();
    }

    connect() {
        try {
            this.ws = new WebSocket(this.wsUrl);

            this.ws.on('open', () => {
                this.emit('connected');
                this.clearReconnect();
            });

            this.ws.on('message', (data: WebSocket.Data) => {
                try {
                    const message = JSON.parse(data.toString());
                    this.handleMessage(message);
                } catch (error) {
                    console.error('Failed to parse message:', error);
                }
            });

            this.ws.on('close', () => {
                this.emit('disconnected');
                this.scheduleReconnect();
            });

            this.ws.on('error', (error) => {
                console.error('WebSocket error:', error);
                this.emit('error', error);
            });
        } catch (error) {
            console.error('Failed to connect:', error);
            this.scheduleReconnect();
        }
    }

    disconnect() {
        this.clearReconnect();
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    private handleMessage(message: any) {
        switch (message.type) {
            case 'status':
                this.emit('status', message.data);
                break;
            case 'signal':
                this.emit('signal', message.data);
                break;
            case 'position_update':
                this.emit('position_update', message.data);
                break;
            case 'pnl_update':
                this.emit('pnl_update', message.data);
                break;
            case 'alert':
                this.emit('alert', message.data);
                break;
            default:
                console.warn('Unknown message type:', message.type);
        }
    }

    private scheduleReconnect() {
        if (this.reconnectInterval) {
            return;
        }
        this.reconnectInterval = setTimeout(() => {
            this.connect();
        }, 5000);
    }

    private clearReconnect() {
        if (this.reconnectInterval) {
            clearTimeout(this.reconnectInterval);
            this.reconnectInterval = null;
        }
    }

    send(message: any) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        }
    }

    approveSignal(signalId: string) {
        this.send({
            type: 'approve_signal',
            signal_id: signalId,
        });
    }

    rejectSignal(signalId: string) {
        this.send({
            type: 'reject_signal',
            signal_id: signalId,
        });
    }

    async emergencyStop() {
        this.send({
            type: 'emergency_stop',
        });
    }

    requestDashboardData() {
        this.send({
            type: 'get_dashboard_data',
        });
    }
}
