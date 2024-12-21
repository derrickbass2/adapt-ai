import {store} from '../store';
import {updateMetricValue, updateRealTimeMetrics} from '../store/slices/dashboardSlice';
import {addAlert} from '../store/slices/alertsSlice';
import {MetricUpdate, WebSocketMessage} from '../types';

export class WebSocketService {
    private static instance: WebSocketService;
    private ws: WebSocket | null = null;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;
    private reconnectTimeout = 3000;

    private constructor() {
    }

    public static getInstance(): WebSocketService {
        if (!WebSocketService.instance) {
            WebSocketService.instance = new WebSocketService();
        }
        return WebSocketService.instance;
    }

    public connect(organizationId: string): void {
        const wsUrl = `${process.env.REACT_APP_WS_URL}/organizations/${organizationId}/metrics/stream`;

        this.ws = new WebSocket(wsUrl);
        this.setupEventHandlers();
    }

    public sendMessage(message: any): void {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        }
    }

    public disconnect(): void {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    private setupEventHandlers(): void {
        if (!this.ws) return;

        this.ws.onopen = this.handleOpen.bind(this);
        this.ws.onmessage = this.handleMessage.bind(this);
        this.ws.onclose = this.handleClose.bind(this);
        this.ws.onerror = this.handleError.bind(this);
    }

    private handleOpen(): void {
        console.log('WebSocket connection established');
        this.reconnectAttempts = 0;

        // Send authentication message
        this.sendMessage({
            type: 'AUTH',
            payload: {
                token: localStorage.getItem('authToken'),
            },
        });
    }

    private handleMessage(event: MessageEvent): void {
        try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.processMessage(message);
        } catch (error) {
            console.error('Error processing WebSocket message:', error);
        }
    }

    private processMessage(message: WebSocketMessage): void {
        switch (message.type) {
            case 'METRIC_UPDATE':
                this.handleMetricUpdate(message.payload as MetricUpdate);
                break;
            case 'ALERT':
                store.dispatch(addAlert(message.payload));
                break;
            case 'BATCH_UPDATE':
                store.dispatch(updateRealTimeMetrics(message.payload));
                break;
            case 'ERROR':
                this.handleError(message.payload);
                break;
            default:
                console.warn('Unknown message type:', message.type);
        }
    }

    private handleMetricUpdate(update: MetricUpdate): void {
        store.dispatch(updateMetricValue({
            path: update.path,
            value: update.value,
            timestamp: update.timestamp,
        }));
    }

    private handleClose(event: CloseEvent): void {
        console.log('WebSocket connection closed:', event.reason);
        this.attemptReconnect();
    }

    private handleError(error: any): void {
        console.error('WebSocket error:', error);
        store.dispatch({
            type: 'WS_ERROR',
            payload: error,
        });
    }

    private attemptReconnect(): void {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            return;
        }

        setTimeout(() => {
            this.reconnectAttempts++;
            this.connect(store.getState().dashboard.organizationId);
        }, this.reconnectTimeout * this.reconnectAttempts);
    }
}
