const express = require('express');
const {exec} = require('child_process');
const bodyParser = require('body-parser');
const app = express();
const port = process.env.PORT || 3001;

app.use(bodyParser.json());

// Helper function to execute a Python script and handle errors
const executePythonScript = (scriptPath, args, res) => {
    const command = `python ${scriptPath} ${args}`;
    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error executing script: ${stderr}`);
            res.status(500).send(`Error: ${stderr}`);
            return;
        }
        res.send(stdout);
    });
};

// Endpoint to preprocess data
app.post('/api/spark-engine/preprocess', (req, res) => {
    const {file_path, feature_cols} = req.body;

    // Properly format feature_cols array as a comma-separated string
    const featureColsString = feature_cols.map(col => `"${col}"`).join(',');

    const args = `--file_path "${file_path}" --feature_cols "${featureColsString}"`;
    executePythonScript('/Users/derrickbass/Public/adaptai/src/modular_learning_system/spark_engine/preprocess.py', args, res);
});

// Endpoint to perform clustering
app.post('/api/spark-engine/cluster', (req, res) => {
    const {file_path, num_clusters} = req.body;
    const args = `--file_path "${file_path}" --num_clusters ${num_clusters}`;
    executePythonScript('/Users/derrickbass/Public/adaptai/src/modular_learning_system/spark_engine/cluster.py', args, res);
});

app.listen(port, () => {
    console.log(`Spark Engine API running on port ${port}`);
});
