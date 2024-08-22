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
app.use(express.static(path.join(__dirname, 'public')));

// Example route
app.get('/api/data', (req, res) => {
  res.json({ message: 'Welcome to ADAPT AI Web App' });
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});

module.exports = app;