import {configureStore} from '@reduxjs/toolkit';
import {TypedUseSelectorHook, useDispatch, useSelector} from 'react-redux';
import dashboardReducer from './slices/dashboardSlice';
import alertsReducer from './slices/alertsSlice';
import userReducer from './slices/userSlice';
import integrationReducer from './slices/integrationSlice';

export const store = configureStore({
    reducer: {
        dashboard: dashboardReducer,
        alerts: alertsReducer,
        user: userReducer,
        integration: integrationReducer,
    },
    middleware: (getDefaultMiddleware) =>
        getDefaultMiddleware({
            serializableCheck: false,
            thunk: true,
        }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Typed hooks
export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;

