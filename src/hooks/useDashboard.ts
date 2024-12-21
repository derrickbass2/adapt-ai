import {useEffect} from 'react';
import {useAppDispatch, useAppSelector} from '../store';
import {fetchMetrics, setFilters, setTimeRange} from '../store/slices/dashboardSlice';

export const useDashboard = (organizationId: string) => {
    const dispatch = useAppDispatch();
    const {
        metrics,
        loading,
        error,
        timeRange,
        filters,
        lastUpdated,
    } = useAppSelector(state => state.dashboard);

    useEffect(() => {
        dispatch(fetchMetrics({organizationId, timeRange, filters}));
    }, [dispatch, organizationId, timeRange, filters]);

    const updateTimeRange = (newTimeRange: string) => {
        dispatch(setTimeRange(newTimeRange));
    };

    const updateFilters = (newFilters: any) => {
        dispatch(setFilters(newFilters));
    };

    return {
        metrics,
        loading,
        error,
        timeRange,
        filters,
        lastUpdated,
        updateTimeRange,
        updateFilters,
    };
};

