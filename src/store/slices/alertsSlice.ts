import {createAsyncThunk, createSlice, PayloadAction} from '@reduxjs/toolkit';
import {Alert, AlertState} from '../../types';
import {dashboardService} from "../../services/api";

const initialState: AlertState = {
    alerts: [],
    unreadCount: 0,
};

export const fetchAlerts = createAsyncThunk(
    'alerts/fetchAlerts',
    async (organizationId: string, {rejectWithValue}) => {
        try {
            const response = await dashboardService.getAlerts(organizationId);
            return response;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to fetch alerts');
        }
    }
);

const alertsSlice = createSlice({
    name: 'alerts',
    initialState,
    reducers: {
        addAlert: (state, action: PayloadAction<Alert>) => {
            state.alerts.unshift(action.payload);
            state.unreadCount += 1;
        },
        markAlertAsRead: (state, action: PayloadAction<string>) => {
            const alert = state.alerts.find(a => a.id === action.payload);
            if (alert && !alert.read) {
                alert.read = true;
                state.unreadCount -= 1;
            }
        },
        clearAlerts: (state) => {
            state.alerts = [];
            state.unreadCount = 0;
        },
    },
    extraReducers: (builder) => {
        builder
            .addCase(fetchAlerts.fulfilled, (state, action) => {
                state.alerts = action.payload;
                state.unreadCount = action.payload.filter(
                    (alert: Alert) => !alert.read
                ).length;
            });
    },
});

export const {addAlert, markAlertAsRead, clearAlerts} = alertsSlice.actions;
export default alertsSlice.reducer;

