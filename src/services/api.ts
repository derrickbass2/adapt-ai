import axios from 'axios';
import {MetricData, UpdateMetricPayload} from '../types/dashboard';

const api = axios.create({
    baseURL: process.env.REACT_APP_API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const dashboardService = {
    // Fetch dashboard metrics
    async getMetrics(
        organizationId: string,
        timeRange: string,
        filters: object
    ): Promise<MetricData> {
        const response = await api.get(`/organizations/${organizationId}/metrics`, {
            params: {timeRange, ...filters},
        });
        return response.data;
    },

    // Update metrics
    async updateMetrics(
        organizationId: string,
        payload: UpdateMetricPayload
    ): Promise<void> {
        await api.post(`/organizations/${organizationId}/metrics`, payload);
    },

    // Stream real-time updates
    subscribeToUpdates(organizationId: string, callback: (data: MetricData) => void) {
        const ws = new WebSocket(
            `${process.env.REACT_APP_WS_URL}/organizations/${organizationId}/metrics/stream`
        );

        ws.onmessage = (event) => {
            callback(JSON.parse(event.data));
        };

        return () => ws.close();
    },
};
