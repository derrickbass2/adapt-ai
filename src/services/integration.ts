export class IntegrationService {
    private static instance: IntegrationService;
    private integrations: Map<string, Integration>;

    public async syncData(organizationId: string): Promise<void> {
        for (const integration of this.integrations.values()) {
            await integration.sync(organizationId);
        }
    }

    public registerIntegration(integration: Integration): void {
        this.integrations.set(integration.name, integration);
    }
}
