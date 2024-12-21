import React, {useMemo} from 'react';
import {CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,} from 'recharts';
import {MetricUpdate} from '../../types';

interface RealTimeChartProps {
    data: MetricUpdate[];
    timeWindowMinutes?: number; // in minutes
    height?: number; // optional height for ResponsiveContainer
}

timeWindowMinutes ? : number; // in minutes
}

export const RealTimeChart: React.FC<RealTimeChartProps> = ({
                                                                data,
                                                                timeWindowMinutes = 5,
                                                                height = 300,
                                                            }) => {

    const chartData = useMemo(() => {
        const now = Date.now();
        const windowStart = now - timeWindowMinutes * 60 * 1000;

        return (Array.isArray(data) ? data : [])
            .filter(update =>
                update &&
                typeof update.timestamp === 'string' &&
                typeof update.value === 'number' &&
                !Number.isNaN(new Date(update.timestamp).getTime()) &&
                new Date(update.timestamp).getTime() > windowStart
            )
            .map(update => ({
                timestamp: new Date(update.timestamp).toLocaleTimeString(),
                value: update.value,
            }));
    }, [data, timeWindowMinutes]);

    return (
        <ResponsiveContainer width="100%" height={height}>
            <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3"/>
                <XAxis
                    dataKey="timestamp"
                    tick={{fontSize: 12}}
                />
                <YAxis/>
                <Tooltip/>
                <Legend/>
                <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#8884d8"
                    dot={false}
                    isAnimationActive={false}
                />
            </LineChart>
        </ResponsiveContainer>
    );
};

