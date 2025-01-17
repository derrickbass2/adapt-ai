const express = require('express');
const {exec} = require('child_process');
const bodyParser = require('body-parser');
const app = express();
const port = process.env.PORT || 3003;

app.use(bodyParser.json());

app.post('/api/neurotech-network/analyze', (req, res) => {
    const data = req.body;
    exec(`python /Users/derrickbass/Public/adaptai/src/modular_learning_system/neurotech_network/analyze.py`, {input: JSON.stringify(data)}, (error, stdout, stderr) => {
        if (error) {
            res.status(500).send(stderr);
            return;
        }
        res.send(stdout);
    });
});

app.listen(port, () => {
    console.log(`Neurotech Network API running on port ${port}`);
});
