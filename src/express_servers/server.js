const express = require('express');
const axios = require('axios');
const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

app.post('/api/data', async (req, res) => {
    const {file_path, feature_cols, num_clusters} = req.body;

    try {
        const preprocessResponse = await axios.post('http://localhost:3001/api/spark-engine/preprocess', {
            file_path,
            feature_cols
        });
        const clusterResponse = await axios.post('http://localhost:3001/api/spark-engine/cluster', {
            file_path,
            num_clusters
        });
        const aaGenomeResponse = await axios.post('http://localhost:3002/api/aa-genome/process', clusterResponse.data);
        const neurotechResponse = await axios.post('http://localhost:3003/api/neurotech-network/analyze', aaGenomeResponse.data);

        res.send(neurotechResponse.data);
    } catch (error) {
        res.status(500).send(error.message);
    }
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
