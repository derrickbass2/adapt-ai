import {Box, Grid, Paper} from '@mui/material';
import {styled} from '@mui/system';

const DashboardContainer = styled(Box)(({theme}) => ({
    padding: theme.spacing(3),
    height: '100vh',
    overflow: 'hidden',
    backgroundColor: theme.palette.background.default,
}));

const DashboardGrid = styled(Grid)(({theme}) => ({
    height: 'calc(100% - 64px)', // Adjust for header
    overflow: 'auto',
    gap: theme.spacing(2),
}));

const MetricCard = styled(Paper)(({theme}) => ({
    padding: theme.spacing(2),
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    position: 'relative',
    '&:hover': {
        boxShadow: theme.shadows[4],
    },
}));
