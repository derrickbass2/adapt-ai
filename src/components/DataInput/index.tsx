import React, {useState} from "react";
import {dashboardService} from "../../services/api";

export const DataInputForm: React.FC<DataInputFormProps> = ({
                                                                organizationId,
                                                                onSubmit,
                                                            }) => {
    const [formData, setFormData] = useState<InputFormData>(initialFormData);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            await dashboardService.updateMetrics(organizationId, formData);
            onSubmit?.(formData);
        } catch (error) {
            console.error('Error updating metrics:', error);
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            {/* Form fields */}
        </form>
    );
};
