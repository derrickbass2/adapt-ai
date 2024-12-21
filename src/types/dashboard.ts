export interface MetricData {
    organizationName: string;
    lastUpdated: string;
    usageAnalytics: UsageAnalytics;
    adoptionRates: AdoptionRates;
    resistanceMetrics: ResistanceMetrics;
    successMetrics: SuccessMetrics;
    alerts: Alert[];
    detailedMetrics: DetailedMetric[];
    departments: Department[];
    thresholds: Thresholds;
    targets: Targets;
}
