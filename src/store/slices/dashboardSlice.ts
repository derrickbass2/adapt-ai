import {handleWebSocketReconnection} from '../../services/websockets';

import {dashboardService} from '../../services/api';
import {DashboardState, FilterOptions, MetricData} from '../../types';

const initialState: DashboardState = {
    metrics: null,
    loading: false,
    error: null,
    timeRange: '7d',
    filters: {},
    lastUpdated: null,
};

// Async thunks
export const fetchMetrics = createAsyncThunk(
    'dashboard/fetchMetrics',
    async ({
               organizationId,
               timeRange,
               filters
           }: {
        organizationId: string;
        timeRange: string;
        filters: Record<string, any>;
    }) => {
        try {
            const response = await dashboardService.getMetrics(
                organizationId,
                timeRange,
                filters
            );
            return response;
        } catch (error: any) {
            throw new Error(error.message || 'Failed to fetch metrics');
        }
    }
);

export const updateMetrics = createAsyncThunk(
    'dashboard/updateMetrics',
    async ({
               organizationId,
               metrics
           }: {
        organizationId: string;
        metrics: Partial<MetricData>;
    }) => {
        try {
            const response = await dashboardService.updateMetrics(organizationId, metrics);

            return response;
        } catch (error: any) {
            throw new Error(error.message || 'Failed to update metrics');
        }
    )

    ).

        export
        const subscribeToUpdates = (organizationId: string, token: string) => {
            const ws = new WebSocket(process.env.REACT_APP_WS_URL);

            let isReconnecting = false;

            const handleOpen = () => {
                ws.send(JSON.stringify({type: 'AUTH', token}));
            };
            ws.addEventListener('open', handleOpen);

            handleWebSocketReconnection(ws, {
                onReconnecting: () => (isReconnecting = true),
                onReconnected: () => {
                    isReconnecting = false;
                    ws.send(JSON.stringify({type: 'AUTH', token}));
                },
            });

            const cleanup = () => {
                ws.removeEventListener('open', handleOpen);
                ws.close();
            };

            return cleanup;
        };

        const dashboardSlice = createSlice({
            name: 'dashboard',
            initialState,
            reducers: {
                setTimeRange: (state, action: PayloadAction<string>) => {
                    state.timeRange = action.payload;
                },
                setFilters: (state, action: PayloadAction<FilterOptions>) => {
                    state.filters = action.payload;
                },
                updateMetricValue: (
                    state,
                    action: PayloadAction<{
                        path: string[];
                        value: any;
                    }>
                ) => {
                    const {path, value} = action.payload;
                    let current = state.metrics;
                    for (let i = 0; i < path.length - 1; i++) {
                        current = current[path[i]];
                    }
                    current[path[path.length - 1]] = value;
                },
                resetDashboard: (state) => {
                    return initialState;
                },
            },
            extraReducers: (builder) => {
                builder
                    .addCase(fetchMetrics.pending, (state) => {
                        state.loading = true;
                        state.error = null;
                    })
                    .addCase(fetchMetrics.fulfilled, (state, action) => {
                        state.loading = false;
                        state.metrics = action.payload;
                        state.lastUpdated = new Date().toISOString();
                    })
                    .addCase(fetchMetrics.rejected, (state, action) => {
                        state.loading = false;
                        state.error = action.error.message || 'Failed to fetch metrics';
                    })
                    .addCase(updateMetrics.fulfilled, (state, action) => {
                        state.metrics = {...state.metrics, ...action.payload};
                        state.lastUpdated = new Date().toISOString();
                    });
            },
        });

        if (!process.env.REACT_APP_API_URL || !process.env.REACT_APP_WS_URL) {
            throw new Error('Required environment variables are not properly configured.');
        }

        export const {
            setTimeRange,
            setFilters,
            updateMetricValue,
            resetDashboard
        } = dashboardSlice.actions;

        export default dashboardSlice.reducer;

        export const dashboardSlice = createSlice({
            update this to prevent duplication
            initialState,
            reducers: {
                // ... existing reducers

                updateRealTimeMetrics: (state, action: PayloadAction<MetricUpdate[]>) => {
                    action.payload.forEach(update => {
                        let current = state.metrics;
                        for (let i = 0; i < update.path.length - 1; i++) {
                            current = current[update.path[i]];
                        }
                        current[update.path[update.path.length - 1]] = update.value;
                    });

                    state.lastUpdated = new Date().toISOString();
                },

                addMetricDataPoint: (state, action: PayloadAction<MetricUpdate>) => {
                    const {path, value, timestamp} = action.payload;
                    if (!state.realTimeData) {
                        state.realTimeData = {};
                    }

                    const key = path.join('.');
                    if (!state.realTimeData[key]) {
                        state.realTimeData[key] = [];
                    }

                    state.realTimeData[key].push({value, timestamp});

                    // Keep only last 100 data points
                    if (state.realTimeData[key].length > 100) {
                        state.realTimeData[key].shift();
                    }
                },
            },
        });
