const express = require('express');
const cors = require('cors');
const path = require('path');
const dotenv = require('dotenv');

dotenv.config();

const app = express();
const port = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// Serve static files from the 'public' directory
app.use(express.static(path.resolve(__dirname, '../../public'))); // Adjust path if necessary

// Example route for the API
app.get('/api/data', (req, res) => {
    res.json({message: 'Welcome to ADAPT AI Web App'});
});

// Serve index.html for all other routes (development)
app.get('*', (req, res) => {
    res.sendFile(path.resolve(__dirname, '../../public', 'index.html')); // Adjust path if necessary
});

// Start the server
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});

module.exports = app;
