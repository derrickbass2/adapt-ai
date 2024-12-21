import React, {useEffect, useState} from 'react';
import {useAppSelector} from '../../store';
import {useWebSocket} from '../../hooks/useWebSocket';
import {MetricCard} from '../MetricCard';
import {RealTimeChart} from '../RealTimeChart';
import {MetricUpdate} from '../../types';

interface RealTimeMetricsProps {
    organizationId: string;
    metricKeys: string[];
}

export const RealTimeMetrics: React.FC<RealTimeMetricsProps> = ({
                                                                    organizationId,
                                                                    metricKeys,
                                                                }) => {
    const {metrics} = useAppSelector(state => state.dashboard);
    const [realTimeData, setRealTimeData] = useState<MetricUpdate[]>([]);
    const {sendMessage} = useWebSocket(organizationId);

    useEffect(() => {
        // Subscribe to specific metrics
        sendMessage({
            type: 'SUBSCRIBE',
            payload: {
                metrics: metricKeys,
            },
        });

        return () => {
            sendMessage({
                type: 'UNSUBSCRIBE',
                payload: {
                    metrics: metricKeys,
                },
            });
        };
    }, [sendMessage, metricKeys]);

    return (
        <div className="real-time-metrics">
            {metricKeys.map(key => (
                <MetricCard
                    key={key}
                    title={key}
                    value={metrics[key]}
                    realTimeData={realTimeData.filter(update => update.path.includes(key))}
                />
            ))}
            <RealTimeChart data={realTimeData}/>
        </div>
    );
};

