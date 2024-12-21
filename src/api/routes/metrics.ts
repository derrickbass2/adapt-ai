import express from 'express';
import {MetricsController} from '../controllers/MetricsController';
import {validateMetricUpdate} from '../middleware/validation';
import {authenticate} from '../middleware/auth';

const router = express.Router();

router.get(
    '/organizations/:organizationId/metrics',
    authenticate,
    MetricsController.getMetrics
);

router.post(
    '/organizations/:organizationId/metrics',
    authenticate,
    validateMetricUpdate,
    MetricsController.updateMetrics
);

router.ws(
    '/organizations/:organizationId/metrics/stream',
    authenticate,
    MetricsController.streamMetrics
);

export default router;

