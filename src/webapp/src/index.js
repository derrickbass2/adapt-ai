import React from 'react';
import ReactDOM from 'react-dom';
import App from './webapp/src/app';
import './webapp/src/index.css';

// Render the App component into the root element in your HTML
ReactDOM.render(
    <React.StrictMode>
        <App/>
    </React.StrictMode>,
    document.getElementById('root') // Assumes you have a <div id="root"></div> in your HTML
);
